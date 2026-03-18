import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sympy import false
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class _RULDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, horizon: int):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.horizon + 1

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx:idx + self.horizon]


class RULNetModel:
    """
    Multi-layer perceptron for multi-step RUL prediction.
    Accepts a config-style dict of hyperparameters on construction.
    """

    def __init__(self, **kwargs):
        self.hyperparameters = kwargs
        self.model      = None
        self.scaler     = None
        self.is_trained = False
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_model_name(cls) -> str:
        return "RULNet_MLP"

    def _params(self) -> Dict[str, Any]:
        defaults = {
            "horizon":      10,
            "hidden_units": [128, 64],
            "dropout":      0.25,
            "rul_scale":    30000.0,
            "epochs":       600,
            "lr":           1e-4,
            "batch_size":   32,
            "patience":     30,
            "seed":         42,
        }
        return {**defaults, **self.hyperparameters}

    def _build_net(self, input_dim: int) -> nn.Module:
        p            = self._params()
        hidden_units = p["hidden_units"]
        dropout      = p["dropout"]
        horizon      = p["horizon"]
        layers, prev = [], input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, horizon))
        return nn.Sequential(*layers)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray) -> "RULNetModel":
        p = self._params()
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])

        self.model = self._build_net(X_train.shape[1]).to(self.device)
        opt     = torch.optim.Adam(self.model.parameters(), lr=p["lr"])
        loss_fn = nn.SmoothL1Loss(beta=0.1)

        loader = DataLoader(
            _RULDataset(X_train, y_train, p["horizon"]),
            batch_size=p["batch_size"], shuffle=True,
        )

        best_val_mae, best_state, no_improve = float("inf"), None, 0

        for epoch in range(1, p["epochs"] + 1):
            self.model.train()
            total_loss = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(Xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(Xb)

            val_mae = self._val_mae(X_val, y_val, p["horizon"], p["rul_scale"])

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>3}/{p['epochs']} | "
                    f"Loss: {total_loss / len(loader.dataset):.5f} | "
                    f"Val MAE: {val_mae:.1f} s"
                )

            if no_improve >= p["patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_trained = True
        logger.info(f"Training complete. Best Val MAE: {best_val_mae:.1f} s")
        return self

    def _val_mae(self, X_val, y_val, horizon, rul_scale) -> float:
        """y_val must be in raw seconds (unscaled)."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(len(X_val) - horizon + 1):
                x_in = torch.tensor(X_val[i:i + 1], dtype=torch.float32).to(self.device)
                preds.append(self.model(x_in).cpu().numpy().reshape(-1))

        preds = np.array(preds)  # scaled output (0–1 range)
        y_true = np.array([y_val[i:i + horizon]
                           for i in range(len(y_val) - horizon + 1)])  # raw seconds

        # Explicit: model output is scaled, y_true is seconds — convert to same space
        preds_seconds = preds * rul_scale
        return float(mean_absolute_error(y_true.flatten(), preds_seconds.flatten()))


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows, horizon) array in raw seconds."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        p = self._params()
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(len(X) - p["horizon"] + 1):
                x_in = torch.tensor(X[i:i+1], dtype=torch.float32).to(self.device)
                preds.append(self.model(x_in).cpu().numpy().reshape(-1) * p["rul_scale"])
        return np.array(preds)

    def save(self, filepath: str) -> str:
        import io, joblib
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        buf = io.BytesIO()
        joblib.dump(self.scaler, buf)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "scaler_bytes":     buf.getvalue(),
            "hyperparameters":  self.hyperparameters,
        }, filepath)
        logger.info(f"Saved RULNetModel to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "RULNetModel":
        import io, joblib
        checkpoint = torch.load(filepath, map_location="cpu")
        instance   = cls(**checkpoint.get("hyperparameters", {}))
        buf        = io.BytesIO(checkpoint["scaler_bytes"])
        instance.scaler = joblib.load(buf)
        state      = checkpoint["model_state_dict"]
        input_dim  = next(iter(state.values())).shape[1]
        instance.model = instance._build_net(input_dim)
        instance.model.load_state_dict(state)
        instance.model.to(instance.device)
        instance.model.eval()
        instance.is_trained = True
        logger.info(f"Loaded RULNetModel from {filepath}")
        return instance