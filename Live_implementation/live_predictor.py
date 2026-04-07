"""
Live Predictor
==============
Loads the deployed RULNetModel from the ModelRegistry and wraps it with
the StandardScaler that was saved alongside the model at training time.

Usage
-----
    predictor = LivePredictor.from_registry(run_id="__deployed__")
    # or load directly:
    predictor = LivePredictor.from_path("workflow_data/.../rul_model.pt")

    rul_s = predictor.predict(feature_vector)   # float, seconds
"""

import io
import logging
from typing import Optional

import joblib
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight fallback — used when RULNetModel cannot be imported
# (e.g. running tests outside the full project tree)
# ---------------------------------------------------------------------------

class _StandaloneModel:
    """Wraps a raw nn.Module state dict so LivePredictor can run without
    the full project's models/ package."""

    def __init__(self, state_dict: dict, input_dim: int, hp: dict):
        horizon      = hp.get("horizon", 10)
        hidden_units = hp.get("hidden_units", [128, 64])
        dropout      = hp.get("dropout", 0.0)
        self._rul_scale = hp.get("rul_scale", 30000.0)
        self._horizon   = horizon

        layers, prev = [], input_dim
        for h in hidden_units:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(),
                       torch.nn.Dropout(dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, horizon))
        net = torch.nn.Sequential(*layers)
        net.load_state_dict(state_dict)
        net.eval()
        self.model      = net
        self.is_trained = True

    def _params(self) -> dict:
        return {"horizon": self._horizon, "rul_scale": self._rul_scale}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_windows, horizon) in raw seconds."""
        horizon = self._horizon
        preds = []
        with torch.no_grad():
            for i in range(len(X) - horizon + 1):
                x_in = torch.tensor(X[i:i+1], dtype=torch.float32)
                out  = self.model(x_in).detach().numpy().reshape(-1) * self._rul_scale
                preds.append(out)
        if not preds:
            # Single-row input — predict once
            x_in = torch.tensor(X[:1], dtype=torch.float32)
            preds.append(self.model(x_in).detach().numpy().reshape(-1) * self._rul_scale)
        return np.array(preds)


class LivePredictor:
    """
    Wraps a trained RULNetModel + its StandardScaler for single-vector
    inference in the live serving pipeline.

    The model outputs a (1, horizon) array of predicted RUL values in
    seconds. predict() returns the step-0 value (immediate next prediction)
    clipped to >= 0.
    """

    def __init__(self, model, scaler, rul_scale: float = 30000.0):
        """
        Prefer the factory methods below over calling this directly.

        Parameters
        ----------
        model   : RULNetModel — trained, is_trained=True
        scaler  : sklearn StandardScaler fitted on training features
        rul_scale : float — must match the rul_scale used during training
        """
        self._model = model
        self._scaler = scaler
        self._rul_scale = rul_scale

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        target_feature: str = "RUL_s",
        rul_scale: float = 30000.0,
    ) -> "LivePredictor":
        """
        Load the currently deployed model from the global ModelRegistry.

        Parameters
        ----------
        target_feature : feature the model predicts (default "RUL_s")
        rul_scale      : must match the rul_scale used during training
        """
        from utils.model_registry import ModelRegistry

        registry = ModelRegistry()
        entry = registry.get_deployed_model(target_feature)
        if entry is None:
            raise RuntimeError(
                f"No deployed model found for target_feature='{target_feature}' "
                f"in the global ModelRegistry."
            )

        logger.info(
            f"Loading deployed model: {entry['model_id']} "
            f"({entry['model_type']}) from {entry['model_path']}"
        )
        return cls.from_path(entry["model_path"], rul_scale=rul_scale)

    @classmethod
    def from_path(
        cls,
        model_path: str,
        rul_scale: float = 30000.0,
    ) -> "LivePredictor":
        """
        Load a RULNetModel checkpoint directly from a .pt file.

        The checkpoint is expected to contain:
            model_state_dict  : PyTorch state dict
            scaler_bytes      : joblib-serialised StandardScaler
            hyperparameters   : dict forwarded to RULNetModel.__init__
        """
        checkpoint = torch.load(model_path, map_location="cpu")

        # Restore scaler
        scaler = joblib.load(io.BytesIO(checkpoint["scaler_bytes"]))

        hp = checkpoint.get("hyperparameters", {})
        state = checkpoint["model_state_dict"]

        # Infer input_dim from the first weight tensor
        first_weight = next(v for k, v in state.items() if "weight" in k)
        input_dim = first_weight.shape[1]

        # Try to use RULNetModel from the project; fall back to a raw nn.Module
        # so that tests (and environments without the full project tree) still work.
        try:
            from models.rul_net_model import RULNetModel
            model = RULNetModel(**hp)
            model.model = model._build_net(input_dim)
            model.model.load_state_dict(state)
            model.model.eval()
            model.is_trained = True
        except ImportError:
            # Lightweight fallback: wrap the raw state dict in a plain nn.Module
            model = _StandaloneModel(state, input_dim, hp)

        logger.info(
            f"Loaded RULNetModel from {model_path} "
            f"| input_dim={input_dim} | hp={hp}"
        )
        return cls(model, scaler, rul_scale=rul_scale)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward_single(self, vec_scaled: np.ndarray) -> np.ndarray:
        """
        Run one forward pass through the underlying nn.Module for a single
        row, returning a (horizon,) array in raw seconds.

        RULNetModel.predict() uses a horizon-sized sliding window and requires
        len(X) >= horizon rows — it returns an empty array for a single row.
        In live mode we always have exactly one feature vector per burst, so
        we call the nn.Module directly instead.
        """
        p = self._model._params()
        rul_scale = p.get("rul_scale", self._rul_scale)

        net = self._model.model          # the underlying nn.Sequential
        net.eval()
        with torch.no_grad():
            x_in = torch.tensor(vec_scaled, dtype=torch.float32)  # (1, n_features)
            out  = net(x_in).detach().cpu().numpy().reshape(-1)    # (horizon,)
        return out * rul_scale

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Predict RUL for a single feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray of shape (n_features,) or (1, n_features)
            The rolling feature vector produced by LiveFeatureBuffer.push_burst().

        Returns
        -------
        float — predicted remaining useful life in seconds (>= 0).
        """
        vec = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        vec_scaled = self._scaler.transform(vec)
        preds = self._forward_single(vec_scaled)   # (horizon,)
        return float(np.clip(preds[0], 0.0, None))

    def predict_horizon(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Return the full horizon of predictions rather than just step-0.

        Returns
        -------
        np.ndarray of shape (horizon,) in seconds.
        """
        vec = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        vec_scaled = self._scaler.transform(vec)
        preds = self._forward_single(vec_scaled)   # (horizon,)
        return np.clip(preds, 0.0, None)

    @property
    def horizon(self) -> int:
        """Number of steps the model predicts ahead."""
        return self._model._params().get("horizon", 10)