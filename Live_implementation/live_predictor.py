"""
Live Predictor
==============
Loads the deployed model from the ModelRegistry (or directly from a .pt file)
and wraps it with the StandardScaler saved alongside the model at training time.

Model type routing
──────────────────
LivePredictor.from_path() inspects the 'model_type' key in the checkpoint:
    "CNN_LSTM"  →  CNNLSTMPredictor (sequence-based forward pass)
    anything else (or key absent) → MLPPredictor  (single-vector forward pass)

Both return a LivePredictor-compatible object with the same public API:
    predictor.predict(feature_vector)          → float (RUL in seconds)
    predictor.predict_horizon(feature_vector)  → np.ndarray (horizon,)

CNN-LSTM live inference
───────────────────────
The CNN-LSTM requires a (1, seq_len, n_features) sequence tensor rather than
a single (1, n_features) vector.  At live inference time the ServingPipeline
calls fe.get_window_matrix() to get the raw (window_size, 19) base-feature
matrix from the Feature Engineering stage, scales it, and passes it here.

The Inference stage (serving_pipeline/inference.py) detects the model type
from the champion JSON / ModelRegistry record and calls either:
    predictor.predict(feature_vector)          → MLP path
    predictor.predict_sequence(window_matrix)  → CNN-LSTM path

Usage
-----
    predictor = LivePredictor.from_registry(target_feature="RUL_s")
    # or load directly:
    predictor = LivePredictor.from_path("model_registry/models/rul_model_cnn_lstm_....pt")

    # MLP:
    rul_s = predictor.predict(feature_vector)          # (n_features,)

    # CNN-LSTM (via Inference stage):
    rul_s = predictor.predict_sequence(window_matrix)  # (window_size, 19)
"""

import io
import logging
from typing import Optional

import joblib
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MLP fallback wrapper — used when RULNetModel cannot be imported
# ─────────────────────────────────────────────────────────────────────────────

class _StandaloneMLPModel:
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
            x_in = torch.tensor(X[:1], dtype=torch.float32)
            preds.append(self.model(x_in).detach().numpy().reshape(-1) * self._rul_scale)
        return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# CNN-LSTM fallback wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _StandaloneCNNLSTMModel:
    """Wraps a raw _CNNLSTMNet state dict for environments without the full
    models/ package. Reconstructs the network from hyperparameters."""

    def __init__(self, state_dict: dict, n_features: int, hp: dict):
        from models.cnn_lstm_model import _CNNLSTMNet
        self._rul_scale = hp.get("rul_scale", 30000.0)
        self._horizon   = hp.get("horizon", 10)

        net = _CNNLSTMNet(
            n_features        = n_features,
            horizon           = hp.get("horizon", 10),
            seq_len           = hp.get("seq_len", 40),
            cnn_channels      = hp.get("cnn_channels", [64, 128]),
            cnn_kernel        = hp.get("cnn_kernel", 3),
            cnn_pool          = hp.get("cnn_pool", 2),
            lstm_hidden       = hp.get("lstm_hidden", 128),
            lstm_layers       = hp.get("lstm_layers", 2),
            fc_hidden         = hp.get("fc_hidden", 64),
            condition_emb_dim = hp.get("condition_emb_dim", 8),
            dropout           = hp.get("dropout", 0.2),
            n_conditions      = hp.get("n_conditions", 3),
        )
        net.load_state_dict(state_dict)
        net.eval()
        self.model      = net
        self.is_trained = True

    def _params(self) -> dict:
        return {"horizon": self._horizon, "rul_scale": self._rul_scale}


# ─────────────────────────────────────────────────────────────────────────────
# LivePredictor — unified interface for MLP and CNN-LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LivePredictor:
    """
    Wraps a trained model + its StandardScaler for single-vector (MLP) or
    single-sequence (CNN-LSTM) inference in the live serving pipeline.

    Do not instantiate directly — use the factory methods:
        LivePredictor.from_path(model_path)
        LivePredictor.from_registry(target_feature)
    """

    def __init__(
        self,
        model,
        scaler,
        rul_scale:  float = 30000.0,
        model_type: str   = "MLP",
        condition:  int   = 1,
    ):
        """
        Parameters
        ----------
        model      : trained model instance (MLPModel or CNNLSTMModel)
        scaler     : sklearn StandardScaler fitted on training features
        rul_scale  : fallback scale factor (models store their own internally)
        model_type : "MLP" or "CNN_LSTM" — controls which forward pass to use
        condition  : bearing condition integer (1/2/3) for CNN-LSTM embedding.
                     Set by the Inference stage from the bearing's group.
        """
        self._model      = model
        self._scaler     = scaler
        self._rul_scale  = rul_scale
        self._model_type = model_type.upper()
        self._condition  = condition   # may be updated per-burst by Inference stage

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        target_feature: str = "RUL_s",
        rul_scale:      float = 30000.0,
        condition:      int   = 1,
    ) -> "LivePredictor":
        """
        Load the currently deployed model from the global ModelRegistry.

        Parameters
        ----------
        target_feature : feature the model predicts (default "RUL_s")
        rul_scale      : fallback unscale factor
        condition      : bearing condition 1/2/3 for CNN-LSTM embedding
        """
        from utils.model_registry import ModelRegistry

        registry = ModelRegistry()
        entry    = registry.get_deployed_model(target_feature)
        if entry is None:
            raise RuntimeError(
                f"No deployed model found for target_feature='{target_feature}' "
                f"in the global ModelRegistry."
            )

        logger.info(
            f"Loading deployed model: {entry['model_id']} "
            f"({entry['model_type']}) from {entry['model_path']}"
        )
        return cls.from_path(entry["model_path"], rul_scale=rul_scale, condition=condition)

    @classmethod
    def from_path(
        cls,
        model_path: str,
        rul_scale:  float = 30000.0,
        condition:  int   = 1,
    ) -> "LivePredictor":
        """
        Load a model checkpoint directly from a .pt file.

        The checkpoint must contain:
            model_type        : str  — "MLP" or "CNN_LSTM"
            model_state_dict  : PyTorch state dict
            scaler_bytes      : joblib-serialised StandardScaler
            hyperparameters   : dict

        The model_type key is used to reconstruct the correct architecture.
        If absent (legacy MLP checkpoints), defaults to "MLP".
        """
        checkpoint = torch.load(model_path, map_location="cpu")

        # Detect model type from checkpoint
        model_type = checkpoint.get("model_type", "MLP").upper()
        scaler     = joblib.load(io.BytesIO(checkpoint["scaler_bytes"]))
        hp         = checkpoint.get("hyperparameters", {})
        state      = checkpoint["model_state_dict"]

        if model_type == "CNN_LSTM":
            model = cls._load_cnn_lstm(state, hp)
            logger.info(
                f"Loaded CNN_LSTM model from {model_path} | hp={hp}"
            )
        else:
            model = cls._load_mlp(state, hp)
            logger.info(
                f"Loaded MLP model from {model_path} | hp={hp}"
            )

        return cls(model, scaler, rul_scale=rul_scale, model_type=model_type,
                   condition=condition)

    @staticmethod
    def _load_mlp(state: dict, hp: dict):
        """Reconstruct an MLP model from its state dict."""
        # Infer input_dim from the first weight tensor
        first_weight = next(v for k, v in state.items() if "weight" in k)
        input_dim    = first_weight.shape[1]
        try:
            from models.mlp_model import MLPModel
            model = MLPModel(**hp)
            model.model = model._build_net(input_dim)
            model.model.load_state_dict(state)
            model.model.eval()
            model.is_trained = True
        except ImportError:
            model = _StandaloneMLPModel(state, input_dim, hp)
        return model

    @staticmethod
    def _load_cnn_lstm(state: dict, hp: dict):
        """Reconstruct a CNN-LSTM model from its state dict."""
        try:
            from models.cnn_lstm_model import CNNLSTMModel
            model = CNNLSTMModel(**hp)
            # Infer n_features from the first Conv1d weight: (out_ch, n_features, kernel)
            first_conv_w = next(v for k, v in state.items() if "cnn" in k and "weight" in k)
            n_features   = first_conv_w.shape[1]
            model.model  = model._build_net(n_features)
            model.model.load_state_dict(state)
            model.model.eval()
            model.is_trained = True
        except ImportError:
            first_conv_w = next(v for k, v in state.items() if "cnn" in k and "weight" in k)
            n_features   = first_conv_w.shape[1]
            model = _StandaloneCNNLSTMModel(state, n_features, hp)
        return model

    # ── Public inference API ──────────────────────────────────────────────────

    def set_condition(self, condition: int) -> None:
        """
        Update the condition integer used by the CNN-LSTM embedding.
        Called by the Inference stage when the bearing group is known.
        """
        self._condition = condition

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Predict RUL for a single feature vector (MLP path).

        Parameters
        ----------
        feature_vector : np.ndarray of shape (n_features,) or (1, n_features)
            The rolling feature vector produced by ServingFeatureEngineer.

        Returns
        -------
        float — predicted remaining useful life in seconds (>= 0).
        """
        vec        = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        vec_scaled = self._scaler.transform(vec)
        preds      = self._forward_single_mlp(vec_scaled)   # (horizon,)
        return float(np.clip(preds[0], 0.0, None))

    def predict_horizon(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Return the full horizon of predictions (MLP path).

        Returns
        -------
        np.ndarray of shape (horizon,) in seconds.
        """
        vec        = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        vec_scaled = self._scaler.transform(vec)
        preds      = self._forward_single_mlp(vec_scaled)   # (horizon,)
        return np.clip(preds, 0.0, None)

    def predict_sequence(self, window_matrix: np.ndarray) -> float:
        """
        Predict RUL from a raw window matrix (CNN-LSTM path).

        Parameters
        ----------
        window_matrix : np.ndarray of shape (window_size, n_base_features)
            The raw per-burst base-feature matrix from get_window_matrix().
            n_base_features = 19 (18 time-domain + RUL_norm placeholder).

        Returns
        -------
        float — predicted remaining useful life in seconds (>= 0).
        """
        preds = self._forward_sequence(window_matrix)   # (horizon,)
        return float(np.clip(preds[0], 0.0, None))

    def predict_sequence_horizon(self, window_matrix: np.ndarray) -> np.ndarray:
        """
        Return the full horizon of predictions (CNN-LSTM path).

        Returns
        -------
        np.ndarray of shape (horizon,) in seconds.
        """
        preds = self._forward_sequence(window_matrix)
        return np.clip(preds, 0.0, None)

    @property
    def model_type(self) -> str:
        """Return "MLP" or "CNN_LSTM"."""
        return self._model_type

    @property
    def horizon(self) -> int:
        """Number of steps the model predicts ahead."""
        return self._model._params().get("horizon", 10)

    # ── Internal forward passes ───────────────────────────────────────────────

    def _forward_single_mlp(self, vec_scaled: np.ndarray) -> np.ndarray:
        """
        Single-vector MLP forward pass.
        vec_scaled : (1, n_features) — already scaled.
        Returns    : (horizon,) in raw seconds.
        """
        p         = self._model._params()
        rul_scale = p.get("rul_scale", self._rul_scale)
        net       = self._model.model
        net.eval()
        with torch.no_grad():
            x_in = torch.tensor(vec_scaled, dtype=torch.float32)  # (1, n_features)
            out  = net(x_in).detach().cpu().numpy().reshape(-1)    # (horizon,)
        return out * rul_scale

    def _forward_sequence(self, window_matrix: np.ndarray) -> np.ndarray:
        """
        CNN-LSTM sequence forward pass.

        window_matrix : (window_size, n_base_features) — raw, unscaled.
            The scaler was fitted on the 19 base features column-wise, so
            we scale the matrix row-by-row (each row = one burst's features).
        Returns       : (horizon,) in raw seconds.
        """
        p         = self._model._params()
        rul_scale = p.get("rul_scale", self._rul_scale)

        # Scale the raw window matrix.
        # self._scaler was fitted on the 19-column base feature space.
        # window_matrix shape: (window_size, 19)
        mat        = np.array(window_matrix, dtype=np.float32)      # (seq, 19)
        mat_scaled = self._scaler.transform(mat)                    # (seq, 19)

        # Build input tensor: (1, seq_len, n_features)
        x_in   = torch.tensor(mat_scaled[np.newaxis], dtype=torch.float32)
        # Condition tensor: (1,) — 0-indexed
        cond_t = torch.tensor([self._condition - 1], dtype=torch.long)

        net = self._model.model
        net.eval()
        with torch.no_grad():
            out = net(x_in, cond_t).detach().cpu().numpy().reshape(-1)  # (horizon,) normalised

        # CNN-LSTM outputs normalised [0,1] — scale to seconds.
        # Per-bearing normalisation means rul_scale is an approximation here;
        # the model's own rul_scale (30000 s) is the common output scale.
        return out * rul_scale