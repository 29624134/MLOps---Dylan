# =============================================================================
# 1D CNN-LSTM Multi-Step RUL Predictor  —  PHM 2012 Bearing Dataset
#
# Improvements over v1:
#   1. Log-transform on wavelet features before StandardScaler (tames large values)
#   2. Per-bearing RUL normalisation — each bearing scaled to [0,1] individually
#      so short-life bearings (e.g. Bearing1_4, 339 s) aren't drowned out by
#      long-life ones during training
#   3. Attention over LSTM timesteps — model learns WHICH bursts matter most
#   4. Condition embedding — tells the model which load condition (1/2/3) it's in
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET       = "RUL_s"
SEQ_LEN      = 40
HORIZON      = 10
BURST_PERIOD = 10.0
BATCH_SIZE   = 32
EPOCHS       = 300
LR           = 1e-3
DROPOUT      = 0.2
PATIENCE     = 60
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# RUL labels are normalised per-bearing to [0, 1] during training.
# At inference time we scale back using each bearing's max RUL_s.
# RUL_SCALE below is only used as a fallback for bearings without a known max.
RUL_SCALE    = 30_000.0

# Set True once feature_extraction.py has been run with wavelet extraction
USE_WAVELET_FEATURES = True

# CNN
CNN_CHANNELS = [64, 128]
CNN_KERNEL   = 3
CNN_POOL     = 2

# LSTM
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2

# FC head
FC_HIDDEN    = 64

# Condition embedding (conditions 1, 2, 3 → embed to this size)
CONDITION_EMB_DIM = 8

META_COLS = ["file_id", "burst_idx", "time_s", TARGET, "RUL_norm"]

# Bearing → load condition mapping
BEARING_CONDITION = {
    "Bearing1_1": 1, "Bearing1_2": 1, "Bearing1_3": 1,
    "Bearing1_4": 1, "Bearing1_5": 1, "Bearing1_6": 1, "Bearing1_7": 1,
    "Bearing2_1": 2, "Bearing2_2": 2, "Bearing2_3": 2,
    "Bearing2_4": 2, "Bearing2_5": 2, "Bearing2_6": 2, "Bearing2_7": 2,
    "Bearing3_1": 3, "Bearing3_2": 3, "Bearing3_3": 3,
}

# Ground-truth RULs (PHM 2012 Table 3) — ONLY for Er% scoring
ACTUAL_RUL_S = {
    "Bearing1_3": 5730,
    "Bearing1_4":  339,
    "Bearing1_5": 1610,
    "Bearing1_6": 1460,
    "Bearing1_7": 7570,
    "Bearing2_3": 7530,
    "Bearing2_4": 1390,
    "Bearing2_5": 3090,
    "Bearing2_6": 1290,
    "Bearing2_7":  580,
    "Bearing3_3":  820,
}

torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

TIME_DOMAIN_FEATURES = [
    "h_max", "h_min", "h_mean", "h_sd", "h_rms",
    "h_skew", "h_kurt", "h_crest", "h_form",
    "v_max", "v_min", "v_mean", "v_sd", "v_rms",
    "v_skew", "v_kurt", "v_crest", "v_form",
]

WAVELET_FEATURES = (
    [f"h_wp_energy_{i}"  for i in range(8)] +
    [f"v_wp_energy_{i}"  for i in range(8)] +
    [f"h_wp_entropy_{i}" for i in range(8)] +
    [f"v_wp_entropy_{i}" for i in range(8)]
)

WAVELET_ENERGY_COLS = (
    [f"h_wp_energy_{i}" for i in range(8)] +
    [f"v_wp_energy_{i}" for i in range(8)]
)


def get_feature_cols(df: pd.DataFrame) -> list:
    base = [c for c in TIME_DOMAIN_FEATURES if c in df.columns]
    if USE_WAVELET_FEATURES:
        wav = [c for c in WAVELET_FEATURES if c in df.columns]
        if wav:
            base = base + wav
        else:
            print("[WARN] USE_WAVELET_FEATURES=True but no wavelet columns found — "
                  "using time-domain features only.")
    return base


def log_transform_wavelet_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to wavelet energy columns only.
    Energy values can be 1e6+ while time-domain features are O(1).
    Log transform brings them to a similar scale before StandardScaler.
    Entropy columns are already O(1) so they are left unchanged.
    """
    df = df.copy()
    for col in WAVELET_ENERGY_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0.0))
    return df


# =============================================================================
# DATASET
# =============================================================================

class SequenceRULDataset(Dataset):
    """
    Sliding-window dataset for a single bearing.
    y is normalised to [0, 1] using that bearing's own max RUL_s.
    condition is an integer (1/2/3) broadcast across the whole sequence.
    """

    def __init__(self, X: np.ndarray, y_norm: np.ndarray,
                 condition: int, seq_len: int, horizon: int):
        self.X         = torch.tensor(X,      dtype=torch.float32)
        self.y         = torch.tensor(y_norm, dtype=torch.float32)
        self.condition = torch.tensor(condition - 1, dtype=torch.long)  # 0-indexed
        self.seq_len   = seq_len
        self.horizon   = horizon
        self.n         = len(X) - seq_len - horizon + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x_seq, y_seq, self.condition


# =============================================================================
# MODEL — CNN-LSTM with Attention + Condition Embedding
# =============================================================================

class Attention(nn.Module):
    """
    Scaled dot-product attention over LSTM timesteps.
    Learns which timesteps in the window are most informative.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq, hidden)
        scores  = self.attn(lstm_out).squeeze(-1)   # (batch, seq)
        weights = F.softmax(scores, dim=1)           # (batch, seq)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context


class CNNLSTMRULNet(nn.Module):
    """
    1D CNN → LSTM → Attention → FC head.
    Condition embedding is concatenated to the attended LSTM output.

    Input : x (batch, seq_len, n_features), condition (batch,) int
    Output: (batch, horizon) — normalised RUL in [0, 1]
    """

    def __init__(self, n_features: int, horizon: int, n_conditions: int = 3):
        super().__init__()

        # CNN
        cnn_layers = []
        in_ch = n_features
        for out_ch in CNN_CHANNELS:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=CNN_KERNEL,
                          padding=CNN_KERNEL // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ]
            in_ch = out_ch
        cnn_layers.append(nn.MaxPool1d(kernel_size=CNN_POOL))
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM
        self.lstm = nn.LSTM(
            input_size  = CNN_CHANNELS[-1],
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS,
            batch_first = True,
            dropout     = DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        # Attention
        self.attention = Attention(LSTM_HIDDEN)

        # Condition embedding
        self.cond_emb = nn.Embedding(n_conditions, CONDITION_EMB_DIM)

        # FC head — takes attended LSTM output + condition embedding
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN + CONDITION_EMB_DIM, FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN, horizon),
            nn.Sigmoid(),   # output in [0, 1] — matches per-bearing normalised RUL
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)              # (batch, n_features, seq_len)
        x = self.cnn(x)                      # (batch, CNN_CHANNELS[-1], seq_len//pool)
        x = x.permute(0, 2, 1)              # (batch, seq_len//pool, CNN_CHANNELS[-1])
        lstm_out, _ = self.lstm(x)           # (batch, seq_len//pool, hidden)
        context = self.attention(lstm_out)   # (batch, hidden)
        cond    = self.cond_emb(condition)   # (batch, emb_dim)
        x       = torch.cat([context, cond], dim=1)  # (batch, hidden + emb_dim)
        return self.head(x)                  # (batch, horizon)


# =============================================================================
# DATA LOADING
# =============================================================================

def bearing_name_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def load_bearing(csv_path: str, feature_cols: list = None):
    """
    Load one bearing CSV.
    Returns (X_raw, y_raw_seconds, feature_cols).
    y is NOT scaled — caller handles per-bearing normalisation.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found, skipping: {csv_path}")
        return None, None, None

    df = pd.read_csv(csv_path)

    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' not found in {csv_path}")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Log-transform wavelet energy BEFORE determining feature cols
    df = log_transform_wavelet_energy(df)

    if feature_cols is None:
        feature_cols = get_feature_cols(df)

    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        print(f"[WARN] {csv_path}: missing cols "
              f"{set(feature_cols) - set(available)}")

    if len(df) < SEQ_LEN + HORIZON:
        print(f"[WARN] Too few rows ({len(df)}) in {csv_path}, skipping.")
        return None, None, None

    X = df[available].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    return X, y, available


def prepare_train_val(train_files: list, val_files: list):
    """
    Per-bearing RUL normalisation:
      Each bearing's RUL_s is divided by its own max RUL_s → [0, 1].
      This prevents long-life bearings from drowning short-life ones in the loss.

    Scaler is fitted on all concatenated train features after log-transform.
    """
    feature_cols = None

    # ── Load all train bearings individually (need per-bearing max for norm) ──
    train_datasets = []
    all_train_X    = []

    for f in train_files:
        X, y, feature_cols = load_bearing(f, feature_cols)
        if X is None:
            continue
        bname     = bearing_name_from_path(f)
        condition = BEARING_CONDITION.get(bname, 1)
        rul_max   = float(y.max()) if y.max() > 0 else RUL_SCALE
        y_norm    = y / rul_max
        train_datasets.append((X, y_norm, y, rul_max, condition, bname))
        all_train_X.append(X)

    val_datasets = []
    all_val_X    = []

    for f in val_files:
        X, y, _ = load_bearing(f, feature_cols)
        if X is None:
            continue
        bname     = bearing_name_from_path(f)
        condition = BEARING_CONDITION.get(bname, 1)
        rul_max   = float(y.max()) if y.max() > 0 else RUL_SCALE
        y_norm    = y / rul_max
        val_datasets.append((X, y_norm, y, rul_max, condition, bname))
        all_val_X.append(X)

    if not train_datasets:
        raise RuntimeError("No training data could be loaded.")
    if not val_datasets:
        raise RuntimeError("No validation data could be loaded.")

    # Fit scaler on all train feature rows concatenated
    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_train_X, axis=0))

    # Scale features in every split
    scaled_train = []
    for (X, y_norm, y_raw, rul_max, cond, bname) in train_datasets:
        scaled_train.append((scaler.transform(X), y_norm, y_raw, rul_max, cond, bname))

    scaled_val = []
    for (X, y_norm, y_raw, rul_max, cond, bname) in val_datasets:
        scaled_val.append((scaler.transform(X), y_norm, y_raw, rul_max, cond, bname))

    n_feat = scaled_train[0][0].shape[1]
    print(f"\n{'='*60}")
    print(f"Features  : {n_feat}  ({', '.join(feature_cols)})")
    print(f"Train bearings: {[d[5] for d in scaled_train]}")
    print(f"Val   bearings: {[d[5] for d in scaled_val]}")
    print(f"Per-bearing RUL normalisation enabled")
    print(f"{'='*60}\n")

    return scaled_train, scaled_val, scaler, feature_cols


def build_dataloaders(train_datasets, val_datasets):
    """Build DataLoaders from per-bearing dataset tuples."""

    def make_ds(datasets):
        all_ds = []
        for (X, y_norm, _, _, cond, _) in datasets:
            ds = SequenceRULDataset(X, y_norm, cond, SEQ_LEN, HORIZON)
            if len(ds) > 0:
                all_ds.append(ds)
        return torch.utils.data.ConcatDataset(all_ds)

    train_ds = make_ds(train_datasets)
    val_ds   = make_ds(val_datasets)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_dl, val_dl


def load_test_bearing(csv_path: str, scaler: StandardScaler, feature_cols: list):
    """Load and scale a test bearing. Returns y in raw seconds."""
    X, y, _ = load_bearing(csv_path, feature_cols)
    if X is None:
        return None, None, None
    bname     = bearing_name_from_path(csv_path)
    condition = BEARING_CONDITION.get(bname, 1)
    return scaler.transform(X), y, condition


# =============================================================================
# TRAINING
# =============================================================================

def train_model(train_datasets, val_datasets, n_features):

    train_dl, val_dl = build_dataloaders(train_datasets, val_datasets)

    model     = CNNLSTMRULNet(n_features=n_features, horizon=HORIZON).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )
    criterion = nn.HuberLoss(delta=0.1)

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    train_hist, val_hist = [], []

    print(f"Training on {DEVICE}  |  "
          f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        t_losses = []
        for xb, yb, cb in train_dl:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, cb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_losses.append(loss.item())

        # Validate
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb, cb in val_dl:
                xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
                v_losses.append(criterion(model(xb, cb), yb).item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        train_hist.append(t_loss)
        val_hist.append(v_loss)
        scheduler.step(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{EPOCHS}  "
                  f"train={t_loss:.5f}  val={v_loss:.5f}")

        if v_loss < best_val_loss - 1e-5:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model: CNNLSTMRULNet, X: np.ndarray, y_raw: np.ndarray, condition: int):
    """
    Sliding-window inference.
    Model outputs normalised [0,1] RUL — we rescale using each bearing's own
    max RUL_s (y_raw.max()) to get predictions back in seconds.
    """
    model.eval()
    preds, y_true = [], []

    rul_max  = float(y_raw.max()) if y_raw.max() > 0 else RUL_SCALE
    cond_t   = torch.tensor([condition - 1], dtype=torch.long).to(DEVICE)
    n        = len(X) - SEQ_LEN - HORIZON + 1

    with torch.no_grad():
        for i in range(max(n, 0)):
            x_in = torch.tensor(
                X[i : i + SEQ_LEN][np.newaxis], dtype=torch.float32
            ).to(DEVICE)
            out = model(x_in, cond_t).cpu().numpy().reshape(-1) * rul_max
            preds.append(out)
            y_true.append(y_raw[i + SEQ_LEN : i + SEQ_LEN + HORIZON])

    if not preds:
        return None, None, None, None

    preds  = np.array(preds)
    y_true = np.array(y_true)
    mae    = mean_absolute_error(y_true.flatten(), preds.flatten())
    rmse   = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()))
    return mae, rmse, preds, y_true


def er_score(predicted_rul_s: float, actual_rul_s: float) -> float:
    return abs(predicted_rul_s - actual_rul_s) / actual_rul_s * 100.0


# =============================================================================
# PLOTTING
# =============================================================================

def plot_loss(train_hist, val_hist):
    plt.figure(figsize=(8, 4))
    plt.plot(train_hist, label="Train loss")
    plt.plot(val_hist,   label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Huber Loss (normalised RUL)")
    plt.title("CNN-LSTM Training Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig("cnn_lstm_loss.png", dpi=150)
    plt.show()
    print("Loss curve saved → cnn_lstm_loss.png")


def plot_bearing(preds, y_true, bearing_name, actual_rul_s=None):
    pred_step0 = preds[:, 0]
    true_step0 = y_true[:, 0]
    t = np.arange(len(pred_step0)) * BURST_PERIOD

    plt.figure(figsize=(10, 4))
    plt.plot(t, true_step0 / 3600, label="True RUL (h)",      color="steelblue")
    plt.plot(t, pred_step0 / 3600, label="Predicted RUL (h)", color="tomato",
             linestyle="--")
    if actual_rul_s is not None:
        er = er_score(float(pred_step0[-1]), actual_rul_s)
        plt.title(f"{bearing_name}  |  Er = {er:.1f}%  "
                  f"|  Final pred = {pred_step0[-1]/3600:.2f} h  "
                  f"|  GT = {actual_rul_s/3600:.2f} h")
    else:
        plt.title(bearing_name)
    plt.xlabel("Time (s)"); plt.ylabel("RUL (h)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"cnn_lstm_{bearing_name}.png", dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    BASE = r"D:\Data + Models\Data\ieee-phm-2012-data-challenge-dataset-master\All_Test_Sets"

    def feat(bearing):
        return os.path.join(BASE, bearing, "features.csv")

    TRAIN_FILES = [
        feat("Bearing1_1"), feat("Bearing1_2"),
        feat("Bearing2_1"), feat("Bearing2_2"),
        feat("Bearing3_1"), feat("Bearing3_2"),
    ]

    VAL_FILES = [
        feat("Bearing1_3"),
        feat("Bearing2_5"),
    ]

    TEST_FILES = [
        feat("Bearing1_3"), feat("Bearing1_4"),
        feat("Bearing1_5"), feat("Bearing1_6"),
        feat("Bearing1_7"), feat("Bearing2_3"),
        feat("Bearing2_4"), feat("Bearing2_5"),
        feat("Bearing2_6"), feat("Bearing2_7"),
        feat("Bearing3_3"),
    ]

    # ── Prepare ───────────────────────────────────────────────────────────────
    train_datasets, val_datasets, scaler, feature_cols = prepare_train_val(
        TRAIN_FILES, VAL_FILES
    )
    n_features = train_datasets[0][0].shape[1]

    # ── Train ─────────────────────────────────────────────────────────────────
    model, train_hist, val_hist = train_model(
        train_datasets, val_datasets, n_features
    )
    plot_loss(train_hist, val_hist)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler":           scaler,
        "feature_cols":     feature_cols,
        "hyperparameters": {
            "n_features":       n_features,
            "horizon":          HORIZON,
            "seq_len":          SEQ_LEN,
            "cnn_channels":     CNN_CHANNELS,
            "cnn_kernel":       CNN_KERNEL,
            "cnn_pool":         CNN_POOL,
            "lstm_hidden":      LSTM_HIDDEN,
            "lstm_layers":      LSTM_LAYERS,
            "fc_hidden":        FC_HIDDEN,
            "condition_emb_dim": CONDITION_EMB_DIM,
        },
    }, "cnn_lstm_rul.pt")
    print("Model saved → cnn_lstm_rul.pt")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Bearing':<14} {'MAE (s)':>10} {'RMSE (s)':>10} {'Er%':>8}")
    print(f"{'-'*60}")

    all_er = []
    for fpath in TEST_FILES:
        bname = bearing_name_from_path(fpath)
        X_t, y_t, cond = load_test_bearing(fpath, scaler, feature_cols)
        if X_t is None:
            print(f"{bname:<14}  — skipped (file not found)")
            continue

        mae, rmse, preds, y_true = evaluate(model, X_t, y_t, cond)
        if preds is None:
            print(f"{bname:<14}  — skipped (too few rows)")
            continue

        final_pred = float(preds[-1, 0])
        actual     = ACTUAL_RUL_S.get(bname)
        er         = er_score(final_pred, actual) if actual else float("nan")
        all_er.append(er)

        print(f"{bname:<14} {mae:>10.0f} {rmse:>10.0f} {er:>7.1f}%")
        plot_bearing(preds, y_true, bname, actual_rul_s=actual)

    print(f"{'-'*60}")
    print(f"{'Mean Er%':<14} {'':>10} {'':>10} {np.nanmean(all_er):>7.1f}%")
    print(f"{'='*60}")