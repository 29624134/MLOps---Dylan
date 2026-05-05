"""
models/cnn_lstm_model.py
═══════════════════════════════════════════════════════════════════════════════
CNN-LSTM model for multi-step RUL prediction.

Architecture
────────────
1D CNN  →  LSTM  →  Attention  →  Condition Embedding  →  FC head

The model receives a sequence of feature vectors (seq_len, n_features) rather
than a single aggregated vector. This lets the temporal structure of bearing
degradation inform the prediction — the CNN extracts local patterns across
features, the LSTM captures degradation trends over the window, and the
attention mechanism weights which timesteps matter most.

Condition embedding encodes the bearing group (1/2/3) so the model knows which
operating environment it is predicting for. At live inference the group is
inferred from the bearing name (Bearing2_x → group 2 → condition index 1).

Interface
─────────
CNNLSTMModel mirrors MLPModel exactly so RULTrainerPHM can use either
without modification. The key difference is that:
  - MLPModel.train() receives (X_flat, y) where X is (n_rows, n_features)
  - CNNLSTMModel.train() receives (X_seq, y, conditions) where X_seq is
    (n_windows, seq_len, n_features) — the trainer builds these sequences.

Checkpoint format
─────────────────
Saved as a .pt file containing:
    model_state_dict  : PyTorch state dict
    scaler_bytes      : joblib-serialised StandardScaler (fitted on raw features)
    hyperparameters   : dict (includes model_type="CNN_LSTM" for routing)
    model_type        : "CNN_LSTM"   ← used by LivePredictor to route correctly

The model_type key at the top level allows LivePredictor.from_path() to detect
which architecture to reconstruct without inspecting the state dict.
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import logging
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SequenceRULDataset(Dataset):
    """
    Sliding-window dataset for sequence-based RUL prediction.

    Each sample is a (seq_len, n_features) input window paired with
    a (horizon,) target vector of future normalised RUL values.

    y_norm  : RUL values already normalised per-bearing to [0, 1].
    condition: integer 1/2/3 broadcast across all windows for this bearing.
    """

    def __init__(
        self,
        X: np.ndarray,           # (n_bursts, n_features) — already scaled
        y_norm: np.ndarray,      # (n_bursts,) — normalised to [0, 1]
        condition: int,          # 1, 2, or 3
        seq_len: int,
        horizon: int,
    ):
        self.X         = torch.tensor(X,      dtype=torch.float32)
        self.y         = torch.tensor(y_norm, dtype=torch.float32)
        self.condition = torch.tensor(condition - 1, dtype=torch.long)  # 0-indexed
        self.seq_len   = seq_len
        self.horizon   = horizon
        self.n         = len(X) - seq_len - horizon + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]                         # (seq_len, n_features)
        y_seq = self.y[idx + self.seq_len : idx + self.seq_len + self.horizon]  # (horizon,)
        return x_seq, y_seq, self.condition


# ─────────────────────────────────────────────────────────────────────────────
# Attention module
# ─────────────────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    """
    Scaled dot-product attention over LSTM timesteps.
    Learns which timesteps in the window are most informative for RUL.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq, hidden)
        scores  = self.attn(lstm_out).squeeze(-1)    # (batch, seq)
        weights = F.softmax(scores, dim=1)            # (batch, seq)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class _CNNLSTMNet(nn.Module):
    """
    1D CNN → LSTM → Attention → FC head.

    The condition embedding is concatenated to the attended LSTM output so
    the model can adapt its RUL estimate to the bearing's operating environment.

    Input : x          (batch, seq_len, n_features)
            condition  (batch,) — integer, 0-indexed (0=cond1, 1=cond2, 2=cond3)
    Output: (batch, horizon) — normalised RUL in [0, 1]
    """

    def __init__(
        self,
        n_features:       int,
        horizon:          int,
        seq_len:          int,
        cnn_channels:     List[int],
        cnn_kernel:       int,
        cnn_pool:         int,
        lstm_hidden:      int,
        lstm_layers:      int,
        fc_hidden:        int,
        condition_emb_dim: int,
        dropout:          float,
        n_conditions:     int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len

        # ── CNN ───────────────────────────────────────────────────────────────
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel,
                          padding=cnn_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        cnn_layers.append(nn.MaxPool1d(kernel_size=cnn_pool))
        self.cnn = nn.Sequential(*cnn_layers)

        # ── LSTM ──────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = cnn_channels[-1],
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = _Attention(lstm_hidden)

        # ── Condition embedding ───────────────────────────────────────────────
        self.cond_emb = nn.Embedding(n_conditions, condition_emb_dim)

        # ── FC head ───────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + condition_emb_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
            nn.Sigmoid(),   # output in [0, 1] — matches per-bearing normalised RUL
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)                # (batch, n_features, seq_len)
        x = self.cnn(x)                        # (batch, cnn_channels[-1], seq_len//pool)
        x = x.permute(0, 2, 1)                # (batch, seq_len//pool, cnn_channels[-1])
        lstm_out, _ = self.lstm(x)             # (batch, seq_len//pool, lstm_hidden)
        context  = self.attention(lstm_out)    # (batch, lstm_hidden)
        cond_vec = self.cond_emb(condition)    # (batch, condition_emb_dim)
        x = torch.cat([context, cond_vec], dim=1)  # (batch, lstm_hidden + emb_dim)
        return self.head(x)                    # (batch, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Public model class — mirrors MLPModel interface
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMModel:
    """
    CNN-LSTM model for multi-step RUL prediction.

    Mirrors the MLPModel interface so RULTrainerPHM can use either model
    without branching logic inside the trainer's core methods.

    Key difference from MLPModel
    ────────────────────────────
    train() expects sequence data: X shaped (n_windows, seq_len, n_features)
    and a parallel conditions array (n_windows,) of integers (1/2/3).
    RULTrainerPHM builds this via _prepare_sequence_data() before calling train().

    predict() and save()/load() follow the same contract as MLPModel.
    """

    def __init__(self, **kwargs):
        self.hyperparameters = kwargs
        self.model: Optional[_CNNLSTMNet] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_model_name(cls) -> str:
        return "CNN_LSTM"

    # ── Hyperparameter helpers ────────────────────────────────────────────────

    def _params(self) -> Dict[str, Any]:
        defaults = {
            "horizon":           10,
            "seq_len":           40,
            "cnn_channels":      [64, 128],
            "cnn_kernel":        3,
            "cnn_pool":          2,
            "lstm_hidden":       128,
            "lstm_layers":       2,
            "fc_hidden":         64,
            "condition_emb_dim": 8,
            "dropout":           0.2,
            "rul_scale":         30000.0,
            "epochs":            300,
            "lr":                1e-3,
            "batch_size":        32,
            "patience":          60,
            "seed":              42,
            "n_conditions":      3,
        }
        return {**defaults, **self.hyperparameters}

    def _build_net(self, n_features: int) -> _CNNLSTMNet:
        p = self._params()
        return _CNNLSTMNet(
            n_features        = n_features,
            horizon           = p["horizon"],
            seq_len           = p["seq_len"],
            cnn_channels      = p["cnn_channels"],
            cnn_kernel        = p["cnn_kernel"],
            cnn_pool          = p["cnn_pool"],
            lstm_hidden       = p["lstm_hidden"],
            lstm_layers       = p["lstm_layers"],
            fc_hidden         = p["fc_hidden"],
            condition_emb_dim = p["condition_emb_dim"],
            dropout           = p["dropout"],
            n_conditions      = p["n_conditions"],
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,      # (n_windows, seq_len, n_features)
        y_train: np.ndarray,      # (n_windows,) — normalised [0,1]
        X_val:   np.ndarray,      # (n_val_windows, seq_len, n_features)
        y_val:   np.ndarray,      # (n_val_windows,) — raw seconds (for logging)
        conditions_train: np.ndarray = None,  # (n_windows,) ints 1/2/3
        conditions_val:   np.ndarray = None,  # (n_val_windows,) ints 1/2/3
    ) -> "CNNLSTMModel":
        """
        Train the CNN-LSTM on sequence data.

        Parameters
        ──────────
        X_train          : (n_windows, seq_len, n_features) — scaled sequences
        y_train          : (n_windows,) — per-bearing normalised RUL [0, 1]
        X_val            : (n_val_windows, seq_len, n_features) — scaled sequences
        y_val            : (n_val_windows,) — raw seconds (for loss monitoring)
        conditions_train : (n_windows,) int array of condition codes 1/2/3.
                           If None, defaults to condition 1 for all windows.
        conditions_val   : (n_val_windows,) int array. If None → condition 1.
        """
        p = self._params()
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])

        # Default conditions to 1 if not provided
        if conditions_train is None:
            conditions_train = np.ones(len(X_train), dtype=np.int64)
        if conditions_val is None:
            conditions_val = np.ones(len(X_val), dtype=np.int64)

        n_features = X_train.shape[2]
        self.model = self._build_net(n_features).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=p["lr"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )
        criterion = nn.HuberLoss(delta=0.1)

        # Build datasets — y_val is kept as raw seconds just for logging;
        # the val loss is computed on normalised values (y_val / y_val.max()).
        y_val_max  = float(y_val.max()) if y_val.max() > 0 else p["rul_scale"]
        y_val_norm = (y_val / y_val_max).astype(np.float32)

        train_ds = _SequenceRULDataset(
            X_train, y_train, condition=1,  # condition passed per-sample below
            seq_len=0, horizon=p["horizon"]
        )
        # We build the dataset manually to support per-window conditions
        X_tr_t   = torch.tensor(X_train,          dtype=torch.float32)
        y_tr_t   = torch.tensor(y_train,          dtype=torch.float32)
        c_tr_t   = torch.tensor(conditions_train - 1, dtype=torch.long)  # 0-indexed
        X_va_t   = torch.tensor(X_val,            dtype=torch.float32)
        y_va_t   = torch.tensor(y_val_norm,       dtype=torch.float32)
        c_va_t   = torch.tensor(conditions_val - 1,   dtype=torch.long)

        train_loader = DataLoader(
            list(zip(X_tr_t, y_tr_t, c_tr_t)),
            batch_size=p["batch_size"], shuffle=True,
        )
        val_loader = DataLoader(
            list(zip(X_va_t, y_va_t, c_va_t)),
            batch_size=p["batch_size"], shuffle=False,
        )

        best_val_loss = float("inf")
        best_state    = None
        wait          = 0

        logger.info(
            f"[CNNLSTMModel] Training on {self.device} | "
            f"params={sum(p_.numel() for p_ in self.model.parameters()):,} | "
            f"X_train={X_train.shape} X_val={X_val.shape}"
        )

        for epoch in range(1, p["epochs"] + 1):
            self.model.train()
            t_losses = []
            for xb, yb, cb in train_loader:
                xb, yb, cb = xb.to(self.device), yb.to(self.device), cb.to(self.device)
                optimizer.zero_grad()
                # yb from the zip is shape (batch,) — we need (batch, 1) for
                # the loss to broadcast against (batch, horizon).
                # Actually the target for each window is a single RUL value;
                # we replicate it across the horizon so the model learns to
                # predict the step-0 RUL at every output head simultaneously.
                yb_rep = yb.unsqueeze(1).expand(-1, p["horizon"])
                loss = criterion(self.model(xb, cb), yb_rep)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                t_losses.append(loss.item())

            self.model.eval()
            v_losses = []
            with torch.no_grad():
                for xb, yb, cb in val_loader:
                    xb, yb, cb = xb.to(self.device), yb.to(self.device), cb.to(self.device)
                    yb_rep = yb.unsqueeze(1).expand(-1, p["horizon"])
                    v_losses.append(criterion(self.model(xb, cb), yb_rep).item())

            t_loss = float(np.mean(t_losses))
            v_loss = float(np.mean(v_losses))
            scheduler.step(v_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"[CNNLSTMModel] Epoch {epoch:4d}/{p['epochs']}  "
                    f"train={t_loss:.5f}  val={v_loss:.5f}"
                )

            if v_loss < best_val_loss - 1e-5:
                best_val_loss = v_loss
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= p["patience"]:
                    logger.info(
                        f"[CNNLSTMModel] Early stopping at epoch {epoch} "
                        f"(best val loss: {best_val_loss:.5f})"
                    )
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_trained = True
        logger.info("[CNNLSTMModel] Training complete.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, conditions: np.ndarray = None) -> np.ndarray:
        """
        Sliding-window inference. Returns (n_windows, horizon) in seconds.

        X          : (n_bursts, n_features) — already scaled
        conditions : (n_windows,) int array 1/2/3. Defaults to 1 if None.

        Note: this is the batch predict used during training evaluation.
        For single-burst live inference, LivePredictor calls _forward_single()
        on the raw net directly — see live_predictor.py.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        p = self._params()
        self.model.eval()

        seq_len   = p["seq_len"]
        horizon   = p["horizon"]
        rul_scale = p["rul_scale"]
        n_windows = len(X) - seq_len - horizon + 1

        preds = []
        with torch.no_grad():
            for i in range(max(n_windows, 0)):
                x_seq = torch.tensor(
                    X[i : i + seq_len][np.newaxis], dtype=torch.float32
                ).to(self.device)                           # (1, seq_len, n_features)
                if conditions is not None:
                    cond = torch.tensor(
                        [int(conditions[i]) - 1], dtype=torch.long
                    ).to(self.device)
                else:
                    cond = torch.tensor([0], dtype=torch.long).to(self.device)
                out = self.model(x_seq, cond).cpu().numpy().reshape(-1)  # (horizon,)
                preds.append(out * rul_scale)

        return np.array(preds) if preds else np.empty((0, horizon))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, filepath: str) -> str:
        """
        Save checkpoint to filepath.

        The checkpoint includes a top-level 'model_type' key = "CNN_LSTM"
        so LivePredictor.from_path() can route to the correct architecture
        without inspecting the state dict.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        buf = io.BytesIO()
        joblib.dump(self.scaler, buf)

        torch.save(
            {
                "model_type":       "CNN_LSTM",       # ← routing key for LivePredictor
                "model_state_dict": self.model.state_dict(),
                "scaler_bytes":     buf.getvalue(),
                "hyperparameters":  self.hyperparameters,
            },
            filepath,
        )
        logger.info(f"[CNNLSTMModel] Saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "CNNLSTMModel":
        """Load a CNN-LSTM checkpoint and return a ready-to-predict instance."""
        checkpoint = torch.load(filepath, map_location="cpu")
        hp         = checkpoint.get("hyperparameters", {})
        instance   = cls(**hp)

        buf = io.BytesIO(checkpoint["scaler_bytes"])
        instance.scaler = joblib.load(buf)

        state     = checkpoint["model_state_dict"]
        # Infer n_features from the first Conv1d weight: shape (out_ch, n_features, kernel)
        first_conv_w = next(v for k, v in state.items() if "cnn" in k and "weight" in k)
        n_features = first_conv_w.shape[1]

        instance.model = instance._build_net(n_features)
        instance.model.load_state_dict(state)
        instance.model.to(instance.device)
        instance.model.eval()
        instance.is_trained = True
        logger.info(f"[CNNLSTMModel] Loaded from {filepath}")
        return instance