"""
models/model_trainer.py
═══════════════════════════════════════════════════════════════════════════════
Unified trainer for PHM 2012 RUL models.

Pipeline:
    load_data() -> train() -> evaluate() -> save_model() -> register_model() -> run(run_id)

Model type routing
──────────────────
The trainer reads 'model_type' from its config dict and dispatches to either:
    "MLP"      → MLPModel    (default, Group 1 and Group 3)
    "CNN_LSTM" → CNNLSTMModel (Group 2)

This is controlled by workflow.yaml so no code changes are needed to switch
a group's model — just update the group_model_overrides section in the yaml.

Data sources
────────────
Primary  : DataFrames loaded from MongoDB factory_features by the orchestrator
           and passed in via config keys train_dataframes / val_dataframes /
           test_dataframes.  The orchestrator never passes CSV file paths for
           training data — it reads from MongoDB first.

Fallback : CSV file paths via train_files / val_files / test_files — kept
           for backward compatibility with external scripts (IEEE original
           code, standalone notebooks) that still run directly from disk.

CNN-LSTM feature space
──────────────────────
The CNN-LSTM trains on the 19 RAW base features per burst:

    18 time-domain features (h_max..v_form) + RUL_norm placeholder = 19

It does NOT consume the rolling mean/std/slope features the MLP uses —
the LSTM learns temporal patterns itself, so pre-computed rolling stats
are redundant and (worse) lead to a scaler/feature-count mismatch at
serve time, where ServingFeatureEngineer.get_window_matrix() returns the
raw (window_size, 19) matrix.

To enforce this, _load_data_cnn_lstm() projects the input DataFrames down
to the 19 _CNN_LSTM_BASE_COLS columns in the exact same order used by
ServingFeatureEngineer, then fits a fresh StandardScaler on those 19
columns only. That scaler is the one persisted alongside the model
checkpoint, so live inference and training operate in identical feature
space.

CNN-LSTM sequence preparation
──────────────────────────────
For CNN_LSTM, load_data() returns per-bearing (X, y, rul_max, condition)
tuples rather than a single concatenated array. The trainer then calls
_prepare_sequence_data() to build the sliding-window (seq_len, n_features)
tensors and per-bearing normalised RUL targets required by CNNLSTMModel.train().

The condition integer (1/2/3) is derived from the bearing group field in
bearings.json, which the orchestrator passes via the 'bearing_conditions' key
in the training config. If not provided, condition defaults to 1.

Metrics computed and stored in ModelRegistry
────────────────────────────────────────────
    mae_s        — Mean Absolute Error in seconds (primary comparison metric)
    rmse_s       — Root Mean Square Error in seconds (penalises large errors)
    mape         — Mean Absolute Percentage Error (relative, bearing-agnostic)
    mean_abs_pct — Alias for mape (backward compatibility)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

from utils.model_registry import ModelRegistry
from models.mlp_model import MLPModel
from models.cnn_lstm_model import CNNLSTMModel

logger = logging.getLogger(__name__)

# GT RULs — used for point-in-time evaluation at last recorded burst.
ACTUAL_RUL_S = {
    "Bearing1_3": 5730,  "Bearing1_4":  339,  "Bearing1_5": 1610,
    "Bearing1_6": 1460,  "Bearing1_7": 7570,  "Bearing2_3": 7530,
    "Bearing2_4": 1390,  "Bearing2_5": 3090,  "Bearing2_6": 1290,
    "Bearing2_7":  580,  "Bearing3_3":  820,
}

TARGET    = "RUL_s"
DROP_COLS = ["file_id", "burst_idx", "time_s", TARGET]

# Bearing → load condition mapping (matches CNN-LSTM.py original)
BEARING_CONDITION = {
    "Bearing1_1": 1, "Bearing1_2": 1, "Bearing1_3": 1,
    "Bearing1_4": 1, "Bearing1_5": 1, "Bearing1_6": 1, "Bearing1_7": 1,
    "Bearing2_1": 2, "Bearing2_2": 2, "Bearing2_3": 2,
    "Bearing2_4": 2, "Bearing2_5": 2, "Bearing2_6": 2, "Bearing2_7": 2,
    "Bearing3_1": 3, "Bearing3_2": 3, "Bearing3_3": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# CNN-LSTM feature space
# ─────────────────────────────────────────────────────────────────────────────
# These MUST match serving_pipeline/feature_engineering.py::_ROLLING_COLS
# exactly (same names, same order). ServingFeatureEngineer.get_window_matrix()
# returns a (window_size, 19) matrix in this column order at live inference
# time, so the scaler fitted here at training time must expect the same.
_CNN_LSTM_BASE_COLS = [
    "h_max", "h_min", "h_mean", "h_sd", "h_rms",
    "h_skew", "h_kurt", "h_crest", "h_form",
    "v_max", "v_min", "v_mean", "v_sd", "v_rms",
    "v_skew", "v_kurt", "v_crest", "v_form",
    "RUL_norm",
]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RULTrainerPHM:
    """
    Unified trainer for PHM 2012 RUL models.

    Config keys — MongoDB path (preferred, used by orchestrator)
    ─────────────────────────────────────────────────────────────
    train_dataframes    : list[pd.DataFrame]              — training bearing DataFrames
    val_dataframes      : list[pd.DataFrame]              — validation bearing DataFrames
    test_dataframes     : list[tuple[str, pd.DataFrame]]  — (bearing_name, df) pairs
    bearing_conditions  : dict[str, int]                  — bearing_name → condition (1/2/3)
                          Used by CNN-LSTM to build condition tensors.
                          If not provided, defaults to BEARING_CONDITION lookup.
    output_location     : str
    window_size         : int
    rul_scale           : float
    model_type          : str   — "MLP" (default) or "CNN_LSTM"
    model_params        : dict
    state_location      : str   (optional)
    log_path            : str   (optional)

    Config keys — CSV fallback (external scripts / backward compatibility)
    ──────────────────────────────────────────────────────────────────────
    train_files      : list[str]  — feature CSV paths for training bearings
    val_files        : list[str]  — feature CSV paths for validation bearings
    test_files       : list[str]  — feature CSV paths for test bearings
    """

    def __init__(self, config: dict):
        # ── MongoDB DataFrame path ────────────────────────────────────────────
        self.train_dataframes   = config.get("train_dataframes",   [])
        self.val_dataframes     = config.get("val_dataframes",     [])
        self.test_dataframes    = config.get("test_dataframes",    [])  # list[(str, df)]
        self.bearing_conditions = config.get("bearing_conditions", {})  # name → int

        # ── Legacy CSV path ───────────────────────────────────────────────────
        self.train_files = config.get("train_files", [])
        self.val_files   = config.get("val_files",   [])
        self.test_files  = config.get("test_files",  [])

        self.output_location = config.get("output_location", "workflow_data/models")
        self.window_size     = config.get("window_size", 40)
        self.rul_scale       = config.get("rul_scale", 30000.0)
        self.model_type      = config.get("model_type", "MLP").upper().strip()
        self.model_params    = config.get("model_params", {})
        self.state_location  = config.get("state_location")
        self.log_path        = config.get("log_path")

        # Inject rul_scale so the model knows how to unscale predictions
        if "rul_scale" not in self.model_params:
            self.model_params["rul_scale"] = self.rul_scale

        # For CNN-LSTM, also inject seq_len to match window_size
        if self.model_type == "CNN_LSTM" and "seq_len" not in self.model_params:
            self.model_params["seq_len"] = self.window_size

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"[RULTrainerPHM] model_type='{self.model_type}'  "
            f"window_size={self.window_size}  rul_scale={self.rul_scale}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _condition_for_bearing(self, bearing_name: str) -> int:
        """
        Return the condition integer (1/2/3) for a bearing.
        Priority: bearing_conditions config dict → BEARING_CONDITION lookup → 1.
        """
        if bearing_name in self.bearing_conditions:
            return int(self.bearing_conditions[bearing_name])
        return BEARING_CONDITION.get(bearing_name, 1)

    def _add_rolling_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add rolling mean/std/slope features matching the serving pipeline."""
        if df is None or len(df) < self.window_size:
            return None
        if TARGET not in df.columns:
            self.logger.warning(f"  No '{TARGET}' column — skipping DataFrame.")
            return None

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        keep_cols    = list(dict.fromkeys(
            numeric_cols + [c for c in DROP_COLS if c in df.columns]
        ))
        df = df[keep_cols].copy()
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)

        if len(df) < self.window_size:
            return None

        base_cols = [
            c for c in df.columns
            if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        new_cols = {}
        for col in base_cols:
            new_cols[f"{col}_mean"]  = df[col].rolling(self.window_size).mean()
            new_cols[f"{col}_std"]   = df[col].rolling(self.window_size).std()
            new_cols[f"{col}_slope"] = df[col].rolling(self.window_size).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
            )
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df.dropna().reset_index(drop=True)

    def _load_bearing(self, path: str) -> Optional[pd.DataFrame]:
        """Load a bearing features.csv from disk and add rolling features (CSV path)."""
        if not os.path.exists(path):
            self.logger.warning(f"  File not found: {path}")
            return None
        df = pd.read_csv(path)
        if TARGET not in df.columns:
            self.logger.warning(f"  No '{TARGET}' column in {path} — skipping.")
            return None
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        if len(df) < self.window_size:
            self.logger.warning(
                f"  {path}: only {len(df)} rows < window_size={self.window_size} — skipping."
            )
            return None
        return self._add_rolling_features(df)

    def _get_xy(self, df: pd.DataFrame):
        drop = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=drop).values, df[TARGET].values

    @staticmethod
    def _parse_bearing_name(path: str) -> str:
        for part in reversed(Path(path).parts):
            if part.startswith("Bearing"):
                return part
        return Path(path).stem

    # ── CNN-LSTM base-feature projection ──────────────────────────────────────

    def _project_to_cnn_lstm_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of `df` containing only the 19 base columns the CNN-LSTM
        expects, in the canonical order matching ServingFeatureEngineer.

        Missing columns are filled with 0.0 — this lets the trainer accept
        DataFrames that may or may not include 'RUL_norm' as an input feature
        (it's a placeholder at serve time anyway).
        """
        out = pd.DataFrame(index=df.index)
        for col in _CNN_LSTM_BASE_COLS:
            if col in df.columns:
                out[col] = df[col].astype(np.float32)
            else:
                # RUL_norm is the only column legitimately allowed to be missing
                # in upstream data — fill with 0.0 to match live inference.
                if col != "RUL_norm":
                    self.logger.warning(
                        f"  Column '{col}' missing — filling with 0.0. "
                        f"Check upstream feature extraction."
                    )
                out[col] = 0.0
        return out

    # ── CNN-LSTM sequence preparation ─────────────────────────────────────────

    def _prepare_sequence_data(
        self,
        bearing_datasets: List[Tuple[str, np.ndarray, np.ndarray]],
        scaler: StandardScaler,
        horizon: int,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sliding-window sequence arrays for CNN-LSTM training.

        For each bearing, this method:
          1. Scales features with the fitted scaler.
          2. Normalises RUL per-bearing to [0, 1] (avoids short-life bearings
             being drowned by long-life ones during training).
          3. Slides a (seq_len, n_features) window with step=1 across the
             bearing, producing one training sample per position.
          4. Records the condition integer (1/2/3) for each window.

        Parameters
        ──────────
        bearing_datasets : list of (bearing_name, X_raw, y_raw_seconds)
                           X_raw must already be projected to the 19 base
                           columns expected by the CNN-LSTM.
        scaler           : fitted StandardScaler (fitted on 19-column space)
        horizon          : number of future steps to predict
        seq_len          : length of each input sequence window

        Returns
        ───────
        X_seq        : (total_windows, seq_len, n_features)
        y_seq        : (total_windows,) — normalised RUL [0, 1]
        conditions   : (total_windows,) — int array of condition codes 1/2/3
        """
        all_X, all_y, all_cond = [], [], []

        for bearing_name, X_raw, y_raw in bearing_datasets:
            X_scaled = scaler.transform(X_raw)
            rul_max  = float(y_raw.max()) if y_raw.max() > 0 else self.rul_scale
            y_norm   = y_raw / rul_max
            cond     = self._condition_for_bearing(bearing_name)
            n_windows = len(X_scaled) - seq_len - horizon + 1

            for i in range(max(n_windows, 0)):
                all_X.append(X_scaled[i : i + seq_len])         # (seq_len, n_features)
                all_y.append(y_norm[i + seq_len])                # scalar (step-0 RUL)
                all_cond.append(cond)

        if not all_X:
            raise RuntimeError(
                "No sequence windows could be built — check that bearings have "
                f"at least seq_len+horizon={seq_len + horizon} rows."
            )

        return (
            np.array(all_X,    dtype=np.float32),   # (N, seq_len, n_features)
            np.array(all_y,    dtype=np.float32),   # (N,)
            np.array(all_cond, dtype=np.int64),     # (N,)
        )

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def load_data(self) -> tuple:
        """
        Load and prepare training and validation data.

        For MLP:
            Returns X_train, y_train_scaled, X_val, y_val, scaler
            where X arrays are (n_rows, n_features) flat feature vectors with
            rolling mean/std/slope augmentation (76 features total).

        For CNN-LSTM:
            Returns bearing_datasets_train, bearing_datasets_val, scaler, feature_cols
            where each bearing_dataset is a list of (bearing_name, X_raw, y_raw)
            and X_raw is the 19-column base feature matrix (no rolling stats).
            The trainer's train() method calls _prepare_sequence_data() next.
        """
        # ── CNN-LSTM path branches early ──────────────────────────────────────
        # It does NOT need rolling features; it needs only the 19 raw base
        # columns matching the serving FE's get_window_matrix() output.
        if self.model_type == "CNN_LSTM":
            return self._load_data_cnn_lstm()

        # ── MLP path: load + rolling FE + flat scaler ─────────────────────────
        return self._load_data_mlp()

    # ── MLP data loading ──────────────────────────────────────────────────────

    def _load_data_mlp(self) -> tuple:
        """
        Load data for the MLP path. Returns:
            X_train, y_train_scaled, X_val, y_val, scaler
        """
        # Load raw DataFrames
        if self.train_dataframes:
            self.logger.info(
                f"[MLP] Loading {len(self.train_dataframes)} train DataFrame(s) "
                f"and {len(self.val_dataframes)} val DataFrame(s) [MongoDB path]."
            )
            train_dfs = []
            for df in self.train_dataframes:
                result = self._add_rolling_features(df.copy())
                if result is not None and len(result) > 0:
                    train_dfs.append(result)
            val_dfs = []
            for df in self.val_dataframes:
                result = self._add_rolling_features(df.copy())
                if result is not None and len(result) > 0:
                    val_dfs.append(result)
        else:
            self.logger.info(
                f"[MLP] Loading {len(self.train_files)} train CSV(s) [CSV fallback path]."
            )
            train_dfs = [
                df for f in self.train_files
                if (df := self._load_bearing(f)) is not None
            ]
            val_dfs = [
                df for f in self.val_files
                if (df := self._load_bearing(f)) is not None
            ]

        if not train_dfs:
            raise RuntimeError("No training data could be loaded.")
        if not val_dfs:
            raise RuntimeError("No validation data could be loaded.")

        train_combined = pd.concat(train_dfs, axis=0).reset_index(drop=True)
        val_combined   = pd.concat(val_dfs,   axis=0).reset_index(drop=True)

        drop         = [c for c in DROP_COLS if c in train_combined.columns]
        feature_cols = [c for c in train_combined.columns if c not in drop]

        X_all_train = train_combined[feature_cols].values.astype(np.float32)
        scaler      = StandardScaler()
        scaler.fit(X_all_train)

        X_train, y_train = self._get_xy(train_combined)
        X_val,   y_val   = self._get_xy(val_combined)

        y_train_scaled = y_train / self.rul_scale
        X_train        = scaler.transform(pd.DataFrame(X_train, columns=feature_cols))
        X_val          = scaler.transform(pd.DataFrame(X_val,   columns=feature_cols))

        self.logger.info(f"[MLP] Training features ({len(feature_cols)}): {feature_cols}")
        self.logger.info(f"[MLP] Train: {X_train.shape} | Val: {X_val.shape}")
        self.logger.info(
            f"[MLP] RUL_s train range (scaled): "
            f"{y_train_scaled.min():.4f} -> {y_train_scaled.max():.4f}"
        )
        self.logger.info(
            f"[MLP] RUL_s val range: {y_val.min():.0f} s -> {y_val.max():.0f} s"
        )
        return X_train, y_train_scaled, X_val, y_val, scaler

    # ── CNN-LSTM data loading ─────────────────────────────────────────────────

    def _load_data_cnn_lstm(self) -> tuple:
        """
        Load data for the CNN-LSTM path.

        Returns:
            train_datasets : list of (bearing_name, X_raw_19col, y_raw_seconds)
            val_datasets   : list of (bearing_name, X_raw_19col, y_raw_seconds)
            scaler         : StandardScaler fitted on the 19-column space
            feature_cols   : list[str] — _CNN_LSTM_BASE_COLS

        Notes
        ─────
        * No rolling features are added — the CNN-LSTM learns temporal
          patterns itself via the LSTM.
        * The scaler is fit ONLY on the 19 base columns so it matches the
          (window_size, 19) matrix that ServingFeatureEngineer.get_window_matrix()
          produces at live inference time.
        * RUL_norm is filled with 0.0 if absent (placeholder, matching the
          serving pipeline's behaviour).
        """
        # Load raw DataFrames (no rolling FE — CNN-LSTM doesn't use it)
        if self.train_dataframes:
            self.logger.info(
                f"[CNN-LSTM] Loading {len(self.train_dataframes)} train DataFrame(s) "
                f"and {len(self.val_dataframes)} val DataFrame(s) [MongoDB path]."
            )
            train_raw_dfs = [df.copy() for df in self.train_dataframes]
            val_raw_dfs   = [df.copy() for df in self.val_dataframes]
        else:
            self.logger.info(
                f"[CNN-LSTM] Loading {len(self.train_files)} train CSV(s) "
                f"[CSV fallback path]."
            )
            train_raw_dfs = [pd.read_csv(f) for f in self.train_files if os.path.exists(f)]
            val_raw_dfs   = [pd.read_csv(f) for f in self.val_files   if os.path.exists(f)]

        if not train_raw_dfs:
            raise RuntimeError("No training data could be loaded.")
        if not val_raw_dfs:
            raise RuntimeError("No validation data could be loaded.")

        feature_cols = list(_CNN_LSTM_BASE_COLS)   # 19 cols, canonical order

        def _bname_of(df: pd.DataFrame) -> str:
            if "dataset_id" in df.columns and len(df) > 0:
                return str(df["dataset_id"].iloc[0])
            if "file_id" in df.columns and len(df) > 0:
                return str(df["file_id"].iloc[0])
            return "unknown"

        def _extract_bearing_datasets(dfs):
            """
            Project each DataFrame to the 19 base columns, drop rows with NaN
            in RUL_s, and return (bearing_name, X_19col, y_seconds) tuples.
            """
            datasets = []
            for df in dfs:
                if df is None or len(df) == 0:
                    continue
                if TARGET not in df.columns:
                    self.logger.warning(f"  No '{TARGET}' column — skipping DataFrame.")
                    continue

                bname  = _bname_of(df)
                df_use = df.dropna(subset=[TARGET]).reset_index(drop=True)
                if len(df_use) < (self.window_size + 1):
                    self.logger.warning(
                        f"  [{bname}] only {len(df_use)} rows < window_size+1 "
                        f"({self.window_size + 1}) — skipping."
                    )
                    continue

                X_19  = self._project_to_cnn_lstm_cols(df_use).values.astype(np.float32)
                y_raw = df_use[TARGET].values.astype(np.float32)
                datasets.append((bname, X_19, y_raw))
            return datasets

        train_datasets = _extract_bearing_datasets(train_raw_dfs)
        val_datasets   = _extract_bearing_datasets(val_raw_dfs)

        if not train_datasets:
            raise RuntimeError("[CNN-LSTM] No training bearings survived projection.")
        if not val_datasets:
            raise RuntimeError("[CNN-LSTM] No validation bearings survived projection.")

        # Fit scaler on the 19-column space only — concatenate all train rows.
        X_all_train = np.concatenate([X for _, X, _ in train_datasets], axis=0)
        scaler      = StandardScaler()
        scaler.fit(X_all_train)

        self.logger.info(
            f"[CNN-LSTM] Scaler fit on 19 base columns. "
            f"Train rows={X_all_train.shape[0]}  "
            f"feature_cols={feature_cols}"
        )
        self.logger.info(
            f"[CNN-LSTM] Train bearings={[b for b, _, _ in train_datasets]}  "
            f"Val bearings={[b for b, _, _ in val_datasets]}"
        )

        return train_datasets, val_datasets, scaler, feature_cols

    def train(self, *args) -> Union[MLPModel, CNNLSTMModel]:
        """
        Train the appropriate model based on self.model_type.

        MLP path:
            train(X_train, y_train, X_val, y_val) → MLPModel

        CNN-LSTM path:
            train(bearing_datasets_train, bearing_datasets_val, scaler, feature_cols)
            → CNNLSTMModel
            (scaler and feature_cols are needed here to build the sequences)
        """
        if self.model_type == "CNN_LSTM":
            return self._train_cnn_lstm(*args)
        else:
            return self._train_mlp(*args)

    def _train_mlp(self, X_train, y_train, X_val, y_val) -> MLPModel:
        """Train and return an MLPModel."""
        self.logger.info(f"Training MLPModel with params: {self.model_params}")
        model = MLPModel(**self.model_params)
        model.train(X_train, y_train, X_val, y_val)
        return model

    def _train_cnn_lstm(
        self,
        bearing_datasets_train: List[Tuple[str, np.ndarray, np.ndarray]],
        bearing_datasets_val:   List[Tuple[str, np.ndarray, np.ndarray]],
        scaler:       StandardScaler,
        feature_cols: List[str],
    ) -> CNNLSTMModel:
        """Train and return a CNNLSTMModel."""
        self.logger.info(f"Training CNNLSTMModel with params: {self.model_params}")

        p       = {**{"horizon": 10, "seq_len": self.window_size}, **self.model_params}
        horizon = p["horizon"]
        seq_len = p["seq_len"]

        # Build sliding-window sequences — per-bearing normalisation + conditions
        X_train_seq, y_train_seq, cond_train = self._prepare_sequence_data(
            bearing_datasets_train, scaler, horizon, seq_len
        )

        # For validation we use the same sequence builder.
        # Validation loss is computed on normalised values (CNNLSTMModel handles this).
        X_val_seq, y_val_seq, cond_val = self._prepare_sequence_data(
            bearing_datasets_val, scaler, horizon, seq_len
        )

        # y_val for CNNLSTMModel.train() should be raw seconds (for loss logging).
        y_val_raw = np.array([
            y_raw[seq_len + i]
            for bname, X_raw, y_raw in bearing_datasets_val
            for i in range(max(len(X_raw) - seq_len - horizon + 1, 0))
        ], dtype=np.float32)

        self.logger.info(
            f"[CNN-LSTM] Sequence shapes: X_train={X_train_seq.shape} "
            f"X_val={X_val_seq.shape} | conditions unique={np.unique(cond_train)}"
        )

        model = CNNLSTMModel(**self.model_params)
        model.train(
            X_train       = X_train_seq,
            y_train       = y_train_seq,
            X_val         = X_val_seq,
            y_val         = y_val_raw,
            conditions_train = cond_train,
            conditions_val   = cond_val,
        )
        model.scaler = scaler
        return model

    def evaluate(
        self,
        model: Union[MLPModel, CNNLSTMModel],
        scaler: StandardScaler,
    ) -> Dict[str, Any]:
        """
        Evaluate on test bearings. Returns per-bearing results and mean metrics.

        Metrics computed per bearing (at final recorded burst):
            mae_s     — |predicted_RUL - gt_RUL| in seconds
            rmse_s    — sqrt(mean(errors^2)) across all test bearings
            mape      — mean absolute percentage error

        Routes to _evaluate_mlp or _evaluate_cnn_lstm based on model type.
        """
        if isinstance(model, CNNLSTMModel):
            return self._evaluate_cnn_lstm(model, scaler)
        return self._evaluate_mlp(model, scaler)

    def _resolve_eval_items_mlp(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        """
        Resolve the list of (bearing_name, df) pairs to evaluate the MLP on.
        DataFrames are augmented with rolling features.
        """
        if self.test_dataframes:
            self.logger.info(
                f"Evaluating on {len(self.test_dataframes)} test DataFrame(s) [MongoDB path]."
            )
            return [
                (name, self._add_rolling_features(df.copy()))
                for name, df in self.test_dataframes
            ]
        elif self.test_files:
            self.logger.info(
                f"Evaluating on {len(self.test_files)} test CSV(s) [CSV fallback path]."
            )
            return [
                (self._parse_bearing_name(f), self._load_bearing(f))
                for f in self.test_files
            ]
        elif self.val_dataframes:
            self.logger.info("No test data — falling back to val DataFrames for evaluation.")
            return [
                (f"val_{i}", self._add_rolling_features(df.copy()))
                for i, df in enumerate(self.val_dataframes)
            ]
        else:
            self.logger.info("No test files — falling back to val bearings for evaluation.")
            return [
                (self._parse_bearing_name(f), self._load_bearing(f))
                for f in self.val_files
            ]

    # Backward-compat alias — old callers used this single method name.
    # MLP eval keeps the old behaviour; CNN-LSTM eval uses its own resolver below.
    def _resolve_eval_items(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        return self._resolve_eval_items_mlp()

    def _resolve_eval_items_cnn_lstm(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        """
        Resolve the list of (bearing_name, df) pairs to evaluate the CNN-LSTM on.
        DataFrames are NOT augmented with rolling features — they're returned
        as-is. _evaluate_cnn_lstm projects them to the 19 base columns before
        scaling, mirroring _load_data_cnn_lstm().
        """
        if self.test_dataframes:
            self.logger.info(
                f"[CNN-LSTM eval] {len(self.test_dataframes)} test DataFrame(s) [MongoDB path]."
            )
            return [(name, df.copy()) for name, df in self.test_dataframes]
        elif self.test_files:
            self.logger.info(
                f"[CNN-LSTM eval] {len(self.test_files)} test CSV(s) [CSV fallback path]."
            )
            out = []
            for f in self.test_files:
                if not os.path.exists(f):
                    self.logger.warning(f"  File not found: {f}")
                    continue
                out.append((self._parse_bearing_name(f), pd.read_csv(f)))
            return out
        elif self.val_dataframes:
            self.logger.info("[CNN-LSTM eval] No test data — falling back to val DataFrames.")
            return [(f"val_{i}", df.copy()) for i, df in enumerate(self.val_dataframes)]
        else:
            self.logger.info("[CNN-LSTM eval] No test files — falling back to val bearings.")
            out = []
            for f in self.val_files:
                if not os.path.exists(f):
                    continue
                out.append((self._parse_bearing_name(f), pd.read_csv(f)))
            return out

    def _evaluate_mlp(self, model: MLPModel, scaler: StandardScaler) -> Dict[str, Any]:
        """Evaluate MLPModel on test bearings."""
        eval_items = self._resolve_eval_items_mlp()

        if not eval_items:
            return {"mae_s": None, "rmse_s": None, "mape": None, "mean_abs_pct": None, "summary_rows": []}

        summary_rows = []
        for bearing_name, df_test in eval_items:
            if df_test is None or len(df_test) == 0:
                self.logger.warning(f"  [{bearing_name}] Could not load — skipping.")
                continue

            drop     = [c for c in DROP_COLS if c in df_test.columns]
            X_test   = df_test.drop(columns=drop).values
            y_test   = df_test[TARGET].values
            X_scaled = scaler.transform(X_test)

            preds = model.predict(X_scaled)
            if len(preds) == 0:
                self.logger.warning(f"  [{bearing_name}] Too few rows to predict — skipping.")
                continue

            final_pred  = float(preds[-1, 0]) * model._params().get("rul_scale", self.rul_scale)
            gt_rul      = ACTUAL_RUL_S.get(bearing_name, float(y_test[-1]))
            error_s     = final_pred - gt_rul
            abs_err_s   = abs(error_s)
            abs_pct_err = abs_err_s / gt_rul * 100.0 if gt_rul != 0 else 0.0

            summary_rows.append({
                "bearing":     bearing_name,
                "pred_rul_s":  final_pred,
                "gt_rul_s":    gt_rul,
                "error_s":     error_s,
                "abs_err_s":   abs_err_s,
                "abs_pct_err": abs_pct_err,
            })
            self.logger.info(
                f"  [{bearing_name}] pred={final_pred:.0f}s  gt={gt_rul:.0f}s  "
                f"err={abs_err_s:.0f}s ({abs_pct_err:.1f}%)"
            )

        return self._aggregate_metrics(summary_rows)

    def _evaluate_cnn_lstm(self, model: CNNLSTMModel, scaler: StandardScaler) -> Dict[str, Any]:
        """
        Evaluate CNNLSTMModel on test bearings.

        Mirrors _load_data_cnn_lstm: projects each test DataFrame to the 19
        base columns before scaling, so the scaler dimension matches.
        """
        eval_items = self._resolve_eval_items_cnn_lstm()

        if not eval_items:
            return {"mae_s": None, "rmse_s": None, "mape": None, "mean_abs_pct": None, "summary_rows": []}

        p         = model._params()
        horizon   = p["horizon"]
        seq_len   = p["seq_len"]
        rul_scale = p["rul_scale"]

        summary_rows = []
        for bearing_name, df_test in eval_items:
            if df_test is None or len(df_test) == 0:
                self.logger.warning(f"  [{bearing_name}] Could not load — skipping.")
                continue
            if TARGET not in df_test.columns:
                self.logger.warning(f"  [{bearing_name}] No '{TARGET}' column — skipping.")
                continue

            df_use   = df_test.dropna(subset=[TARGET]).reset_index(drop=True)
            X_19     = self._project_to_cnn_lstm_cols(df_use).values.astype(np.float32)
            y_test   = df_use[TARGET].values.astype(np.float32)

            if len(X_19) < seq_len + horizon:
                self.logger.warning(
                    f"  [{bearing_name}] only {len(X_19)} rows < seq_len+horizon "
                    f"({seq_len + horizon}) — skipping."
                )
                continue

            X_scaled    = scaler.transform(X_19)
            cond        = self._condition_for_bearing(bearing_name)
            n_windows   = len(X_scaled) - seq_len - horizon + 1
            conditions  = np.full(n_windows, cond, dtype=np.int64)

            preds = model.predict(X_scaled, conditions)
            if len(preds) == 0:
                self.logger.warning(f"  [{bearing_name}] Too few rows to predict — skipping.")
                continue

            # preds is (n_windows, horizon) — already in raw seconds
            # (CNNLSTMModel.predict() multiplies by rul_scale internally).
            final_pred  = float(preds[-1, 0])
            gt_rul      = ACTUAL_RUL_S.get(bearing_name, float(y_test[-1]))
            error_s     = final_pred - gt_rul
            abs_err_s   = abs(error_s)
            abs_pct_err = abs_err_s / gt_rul * 100.0 if gt_rul != 0 else 0.0

            summary_rows.append({
                "bearing":     bearing_name,
                "pred_rul_s":  final_pred,
                "gt_rul_s":    gt_rul,
                "error_s":     error_s,
                "abs_err_s":   abs_err_s,
                "abs_pct_err": abs_pct_err,
            })
            self.logger.info(
                f"  [{bearing_name}] pred={final_pred:.0f}s  gt={gt_rul:.0f}s  "
                f"err={abs_err_s:.0f}s ({abs_pct_err:.1f}%)"
            )

        return self._aggregate_metrics(summary_rows)

    def _aggregate_metrics(self, summary_rows: List[dict]) -> Dict[str, Any]:
        """Aggregate per-bearing rows into overall metrics."""
        if not summary_rows:
            self.logger.warning(
                "No evaluation data available — all metrics will be None. "
                "Check that test data is configured and features exist."
            )
            return {
                "mae_s":        None,
                "rmse_s":       None,
                "mape":         None,
                "mean_abs_pct": None,
                "summary_rows": [],
            }

        errors   = np.array([r["error_s"]   for r in summary_rows])
        abs_errs = np.array([r["abs_err_s"] for r in summary_rows])
        pcts     = np.array([r["abs_pct_err"] for r in summary_rows])

        mae_s  = float(np.mean(abs_errs))
        rmse_s = float(np.sqrt(np.mean(errors ** 2)))
        mape   = float(np.mean(pcts))

        self.logger.info(
            f"  {'='*75}\n"
            f"  MAE  : {mae_s:.0f} s ({mae_s/60:.1f} min)\n"
            f"  RMSE : {rmse_s:.0f} s ({rmse_s/60:.1f} min)\n"
            f"  MAPE : {mape:.1f}%\n"
        )
        return {
            "mae_s":        mae_s,
            "rmse_s":       rmse_s,
            "mape":         mape,
            "mean_abs_pct": mape,
            "summary_rows": summary_rows,
        }

    def save_model(self, model: Union[MLPModel, CNNLSTMModel], scaler: StandardScaler) -> str:
        """Save model checkpoint (includes scaler). Returns saved path."""
        model_dir  = os.path.join("model_registry", "models")
        os.makedirs(model_dir, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type_tag = self.model_type.lower()
        model_path = os.path.join(model_dir, f"rul_model_{model_type_tag}_{timestamp}.pt")
        model.scaler = scaler
        model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        return model_path

    def save_results(self, metrics: Dict) -> str:
        """Save per-bearing results CSV. Returns saved path."""
        os.makedirs(self.output_location, exist_ok=True)
        results_csv = os.path.join(self.output_location, "rul_test_results.csv")
        pd.DataFrame(metrics.get("summary_rows", [])).to_csv(results_csv, index=False)
        self.logger.info(f"Results saved to {results_csv}")
        return results_csv

    def register_model(
        self,
        run_id:     str,
        model:      Union[MLPModel, CNNLSTMModel],
        model_path: str,
        metrics:    Dict,
    ) -> str:
        """Register trained model in ModelRegistry. Returns model_id."""
        registry = ModelRegistry()

        if self.train_dataframes:
            source_desc = f"{len(self.train_dataframes)} bearing(s) from MongoDB factory_features"
            num_train   = len(self.train_dataframes)
            num_val     = len(self.val_dataframes)
            num_test    = len(self.test_dataframes)
        else:
            source_desc = self.train_files
            num_train   = len(self.train_files)
            num_val     = len(self.val_files)
            num_test    = len(self.test_files)

        training_data_info = {
            "num_train_files": num_train,
            "num_val_files":   num_val,
            "num_test_files":  num_test,
            "window_size":     self.window_size,
            "source":          source_desc,
            "model_type":      self.model_type,
        }

        model_id = registry.register_model(
            model_path         = model_path,
            model_type         = model.get_model_name(),
            target_feature     = TARGET,
            metrics            = {
                "mae_s":        metrics["mae_s"],
                "rmse_s":       metrics["rmse_s"],
                "mape":         metrics["mape"],
                "mean_abs_pct": metrics["mean_abs_pct"],
            },
            training_data_info = training_data_info,
            metadata           = {
                "run_id":          run_id,
                "hyperparameters": self.model_params,
                "model_type":      self.model_type,
            },
        )

        self.logger.info(f"Model registered with ID: {model_id}")
        return model_id

    # ── Orchestrator entry point ──────────────────────────────────────────────

    def run(self, run_id: str) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Returns dict with model_id, model_path, results_csv, and all metrics.
        """
        self.logger.info(
            f"[{run_id}] Starting {self.model_type} training pipeline."
        )

        if self.model_type == "CNN_LSTM":
            return self._run_cnn_lstm(run_id)
        return self._run_mlp(run_id)

    def _run_mlp(self, run_id: str) -> Dict[str, Any]:
        """MLP training pipeline."""
        X_train, y_train, X_val, y_val, scaler = self.load_data()
        model   = self.train(X_train, y_train, X_val, y_val)
        metrics = self.evaluate(model, scaler)

        model_path  = self.save_model(model, scaler)
        results_csv = self.save_results(metrics)
        model_id    = self.register_model(run_id, model, model_path, metrics)

        self.logger.info(f"[{run_id}] MLPModel training complete — model_id={model_id}")
        return {
            "model_id":    model_id,
            "model_path":  model_path,
            "results_csv": results_csv,
            **metrics,
        }

    def _run_cnn_lstm(self, run_id: str) -> Dict[str, Any]:
        """CNN-LSTM training pipeline."""
        train_datasets, val_datasets, scaler, feature_cols = self.load_data()
        model   = self.train(train_datasets, val_datasets, scaler, feature_cols)
        metrics = self.evaluate(model, scaler)

        model_path  = self.save_model(model, scaler)
        results_csv = self.save_results(metrics)
        model_id    = self.register_model(run_id, model, model_path, metrics)

        self.logger.info(f"[{run_id}] CNNLSTMModel training complete — model_id={model_id}")
        return {
            "model_id":    model_id,
            "model_path":  model_path,
            "results_csv": results_csv,
            **metrics,
        }