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

Metrics computed and stored in ModelRegistry
────────────────────────────────────────────
Per the Model Registry policy, ONLY these three metrics are recorded:

    mae_s   — Mean Absolute Error in seconds (primary champion-selection metric)
    rmse_s  — Root Mean Square Error in seconds (penalises large errors more
              heavily; important since late RUL predictions are more
              consequential than small early ones)
    mcra    — Mean Cumulative Relative Accuracy. Trajectory-based formulation
              inspired by Lei et al. (2018):
                  RA_k  = max(0, 1 - |l_k - l̂_k| / l_k)
                  CRA_i = (1/N_i) * Σ_k RA_k                  (per bearing)
                  MCRA  = (1/M)   * Σ_i CRA_i                 (across bearings)
              For each test bearing the full prediction trajectory is used —
              not just the final burst — so MCRA captures how consistently
              the model tracks the bearing's decline over time.

Data sources
────────────
Primary  : DataFrames loaded from MongoDB factory_features by the orchestrator
           and passed in via config keys train_dataframes / val_dataframes /
           test_dataframes.

Fallback : CSV file paths via train_files / val_files / test_files — kept
           for backward compatibility with external scripts.

CNN-LSTM feature space
──────────────────────
The CNN-LSTM trains on the 19 RAW base features per burst:
    18 time-domain features (h_max..v_form) + RUL_norm placeholder = 19

CNN-LSTM sequence preparation
──────────────────────────────
For CNN_LSTM, load_data() returns per-bearing (X, y, rul_max, condition)
tuples. The trainer then calls _prepare_sequence_data() to build sliding-window
tensors and per-bearing normalised RUL targets required by CNNLSTMModel.train().
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
# exactly (same names, same order).
_CNN_LSTM_BASE_COLS = [
    "h_max", "h_min", "h_mean", "h_sd", "h_rms",
    "h_skew", "h_kurt", "h_crest", "h_form",
    "v_max", "v_min", "v_mean", "v_sd", "v_rms",
    "v_skew", "v_kurt", "v_crest", "v_form",
    "RUL_norm",
]


# ─────────────────────────────────────────────────────────────────────────────
# MCRA helpers — Mean Cumulative Relative Accuracy (Lei et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────

def _cra_for_bearing(
    y_true_s: np.ndarray,
    y_pred_s: np.ndarray,
    eps:      float = 1.0,
) -> Optional[float]:
    """
    Cumulative Relative Accuracy for a single bearing's full prediction
    trajectory.

        RA_k  = max(0, 1 - |l_k - l̂_k| / l_k)
        CRA_i = (1/N_i) * Σ_k RA_k

    Parameters
    ----------
    y_true_s : np.ndarray
        True RUL trajectory in seconds (length N_i).
    y_pred_s : np.ndarray
        Predicted RUL trajectory in seconds (length N_i, same alignment).
    eps : float
        Minimum true-RUL value (seconds) considered for inclusion in the
        average. Points where l_k < eps are skipped to avoid division-by-near-
        zero blow-ups in the last fraction of life. Default 1.0 s.

    Returns
    -------
    float in [0, 1]   — bearing-level CRA. Higher is better.
    None              — if no valid points remain after filtering.
    """
    y_true = np.asarray(y_true_s, dtype=np.float64)
    y_pred = np.asarray(y_pred_s, dtype=np.float64)

    if y_true.size == 0 or y_pred.size == 0:
        return None

    # Align lengths defensively (trajectories should already match)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    valid = y_true >= eps
    if not np.any(valid):
        return None

    ra = 1.0 - np.abs(y_true[valid] - y_pred[valid]) / y_true[valid]
    ra = np.clip(ra, 0.0, 1.0)   # RA cannot be negative or > 1
    return float(np.mean(ra))


def _mcra(per_bearing_cras: List[Optional[float]]) -> Optional[float]:
    """
    Mean CRA across bearings (M test units):

        MCRA = (1/M) * Σ_i CRA_i

    Returns None if no bearings produced a valid CRA.
    """
    vals = [c for c in per_bearing_cras if c is not None]
    if not vals:
        return None
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RULTrainerPHM:
    """
    Unified trainer for PHM 2012 RUL models.

    Config keys — MongoDB path (preferred, used by orchestrator)
    ─────────────────────────────────────────────────────────────
    train_dataframes    : list[pd.DataFrame]
    val_dataframes      : list[pd.DataFrame]
    test_dataframes     : list[tuple[str, pd.DataFrame]]  — (bearing_name, df)
    bearing_conditions  : dict[str, int]                  — bearing_name → condition

    Config keys — CSV fallback
    ──────────────────────────
    train_files / val_files / test_files : list[str]
    """

    def __init__(self, config: dict):
        # ── MongoDB DataFrame path ────────────────────────────────────────────
        self.train_dataframes   = config.get("train_dataframes",   [])
        self.val_dataframes     = config.get("val_dataframes",     [])
        self.test_dataframes    = config.get("test_dataframes",    [])
        self.bearing_conditions = config.get("bearing_conditions", {})

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

        if "rul_scale" not in self.model_params:
            self.model_params["rul_scale"] = self.rul_scale

        if self.model_type == "CNN_LSTM" and "seq_len" not in self.model_params:
            self.model_params["seq_len"] = self.window_size

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"[RULTrainerPHM] model_type='{self.model_type}'  "
            f"window_size={self.window_size}  rul_scale={self.rul_scale}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _condition_for_bearing(self, bearing_name: str) -> int:
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

    @staticmethod
    def _parse_bearing_name(path: str) -> str:
        """Extract bearing name from a feature CSV path like .../BearingX_Y/features.csv"""
        try:
            return Path(path).parent.name
        except Exception:
            return "unknown"

    def _get_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        drop = [c for c in DROP_COLS if c in df.columns]
        X = df.drop(columns=drop).values
        y = df[TARGET].values
        return X, y

    def _project_to_cnn_lstm_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project DataFrame to the 19 base columns in canonical order."""
        out = pd.DataFrame()
        for col in _CNN_LSTM_BASE_COLS:
            if col in df.columns:
                out[col] = df[col]
            elif col == "RUL_norm":
                out[col] = 0.0   # placeholder, matches serving FE
            else:
                out[col] = 0.0
        return out

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self):
        if self.model_type == "CNN_LSTM":
            return self._load_data_cnn_lstm()
        return self._load_data_mlp()

    def _load_data_mlp(self) -> tuple:
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
        return X_train, y_train_scaled, X_val, y_val, scaler

    def _load_data_cnn_lstm(self) -> tuple:
        if self.train_dataframes:
            self.logger.info(
                f"[CNN-LSTM] Loading {len(self.train_dataframes)} train DataFrame(s) "
                f"and {len(self.val_dataframes)} val DataFrame(s) [MongoDB path]."
            )
            train_raw_dfs = [df.copy() for df in self.train_dataframes]
            val_raw_dfs   = [df.copy() for df in self.val_dataframes]
        else:
            self.logger.info(
                f"[CNN-LSTM] Loading {len(self.train_files)} train CSV(s) [CSV fallback path]."
            )
            train_raw_dfs = [pd.read_csv(f) for f in self.train_files if os.path.exists(f)]
            val_raw_dfs   = [pd.read_csv(f) for f in self.val_files   if os.path.exists(f)]

        if not train_raw_dfs:
            raise RuntimeError("No training data could be loaded.")
        if not val_raw_dfs:
            raise RuntimeError("No validation data could be loaded.")

        feature_cols = list(_CNN_LSTM_BASE_COLS)

        def _bname_of(df: pd.DataFrame) -> str:
            if "dataset_id" in df.columns and len(df) > 0:
                return str(df["dataset_id"].iloc[0])
            if "file_id" in df.columns and len(df) > 0:
                return str(df["file_id"].iloc[0])
            return "unknown"

        def _extract_bearing_datasets(dfs):
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

        X_all_train = np.concatenate([X for _, X, _ in train_datasets], axis=0)
        scaler      = StandardScaler()
        scaler.fit(X_all_train)

        return train_datasets, val_datasets, scaler, feature_cols

    # ── Sequence builder (CNN-LSTM) ───────────────────────────────────────────

    def _prepare_sequence_data(
        self,
        bearing_datasets: List[Tuple[str, np.ndarray, np.ndarray]],
        scaler:           StandardScaler,
        horizon:          int,
        seq_len:          int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs, ys, conds = [], [], []
        for bname, X_raw, y_raw in bearing_datasets:
            if len(X_raw) < seq_len + horizon:
                continue
            X_scaled = scaler.transform(X_raw)
            cond     = self._condition_for_bearing(bname)
            rul_max  = float(np.max(y_raw)) if len(y_raw) else 1.0
            if rul_max <= 0:
                rul_max = 1.0
            for i in range(len(X_scaled) - seq_len - horizon + 1):
                Xs.append(X_scaled[i:i + seq_len])
                # Scalar target — RUL at the burst immediately after the window.
                # CNNLSTMModel replicates this across the horizon internally so all
                # output heads are supervised by the same value (matches the
                # original CNN-LSTM design from Lei et al.).
                target = y_raw[i + seq_len] / rul_max
                ys.append(target)
                conds.append(cond)
        if not Xs:
            return (
                np.zeros((0, seq_len, 19), dtype=np.float32),
                np.zeros((0, horizon), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.array(Xs,    dtype=np.float32),
            np.array(ys,    dtype=np.float32),
            np.array(conds, dtype=np.int64),
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, *args) -> Union[MLPModel, CNNLSTMModel]:
        if self.model_type == "CNN_LSTM":
            return self._train_cnn_lstm(*args)
        return self._train_mlp(*args)

    def _train_mlp(self, X_train, y_train, X_val, y_val) -> MLPModel:
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
        self.logger.info(f"Training CNNLSTMModel with params: {self.model_params}")

        p       = {**{"horizon": 10, "seq_len": self.window_size}, **self.model_params}
        horizon = p["horizon"]
        seq_len = p["seq_len"]

        X_train_seq, y_train_seq, cond_train = self._prepare_sequence_data(
            bearing_datasets_train, scaler, horizon, seq_len
        )
        X_val_seq, y_val_seq, cond_val = self._prepare_sequence_data(
            bearing_datasets_val, scaler, horizon, seq_len
        )

        y_val_raw = np.array([
            y_raw[seq_len + i]
            for bname, X_raw, y_raw in bearing_datasets_val
            for i in range(max(len(X_raw) - seq_len - horizon + 1, 0))
        ], dtype=np.float32)

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

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model: Union[MLPModel, CNNLSTMModel],
        scaler: StandardScaler,
    ) -> Dict[str, Any]:
        """
        Evaluate on test bearings. Routes to MLP or CNN-LSTM evaluator.

        Per-bearing trajectory is captured (not just the final burst) so
        MCRA can be computed properly across the full degradation path.

        Returned dict shape:
            {
                "mae_s":        float | None,
                "rmse_s":       float | None,
                "mcra":         float | None,
                "summary_rows": list[dict],
            }
        """
        if isinstance(model, CNNLSTMModel):
            return self._evaluate_cnn_lstm(model, scaler)
        return self._evaluate_mlp(model, scaler)

    def _resolve_eval_items_mlp(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
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

    def _resolve_eval_items(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        return self._resolve_eval_items_mlp()

    def _resolve_eval_items_cnn_lstm(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
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
        """
        Evaluate MLPModel on test bearings.

        For each bearing we collect the full predicted-vs-true trajectory so
        we can compute trajectory-based CRA, then aggregate to MCRA across
        bearings. The single-point final-burst row (for MAE/RMSE summary CSV)
        is also retained for the per-bearing report.
        """
        eval_items = self._resolve_eval_items_mlp()

        if not eval_items:
            return {"mae_s": None, "rmse_s": None, "mcra": None, "summary_rows": []}

        summary_rows  = []
        per_bearing_cra: List[Optional[float]] = []
        rul_scale     = model._params().get("rul_scale", self.rul_scale)

        for bearing_name, df_test in eval_items:
            if df_test is None or len(df_test) == 0:
                self.logger.warning(f"  [{bearing_name}] Could not load — skipping.")
                continue

            drop     = [c for c in DROP_COLS if c in df_test.columns]
            X_test   = df_test.drop(columns=drop).values
            y_test   = df_test[TARGET].values.astype(np.float64)
            X_scaled = scaler.transform(X_test)

            preds = model.predict(X_scaled)
            if len(preds) == 0:
                self.logger.warning(
                    f"  [{bearing_name}] Too few rows to predict — skipping."
                )
                continue

            # Trajectory of predictions in seconds (use horizon step 0)
            preds_traj_s = preds[:, 0].astype(np.float64) * rul_scale
            # True trajectory — align lengths (preds has len = N (no horizon offset
            # for column 0); y_test was the underlying RUL_s of df_test).
            n = min(len(preds_traj_s), len(y_test))
            y_traj_s     = y_test[:n]
            preds_traj_s = preds_traj_s[:n]

            cra = _cra_for_bearing(y_traj_s, preds_traj_s)
            per_bearing_cra.append(cra)

            # Point-in-time row for MAE/RMSE (final burst, as before)
            final_pred = float(preds_traj_s[-1])
            gt_rul     = ACTUAL_RUL_S.get(bearing_name, float(y_test[-1]))
            error_s    = final_pred - gt_rul
            abs_err_s  = abs(error_s)
            abs_pct    = abs_err_s / gt_rul * 100.0 if gt_rul != 0 else 0.0

            summary_rows.append({
                "bearing":     bearing_name,
                "pred_rul_s":  final_pred,
                "gt_rul_s":    gt_rul,
                "error_s":     error_s,
                "abs_err_s":   abs_err_s,
                "abs_pct_err": abs_pct,
                "cra":         cra,
            })
            self.logger.info(
                f"  [{bearing_name}] pred={final_pred:.0f}s  gt={gt_rul:.0f}s  "
                f"err={abs_err_s:.0f}s  CRA={cra if cra is not None else 'N/A'}"
            )

        return self._aggregate_metrics(summary_rows, per_bearing_cra)

    def _evaluate_cnn_lstm(self, model: CNNLSTMModel, scaler: StandardScaler) -> Dict[str, Any]:
        eval_items = self._resolve_eval_items_cnn_lstm()

        if not eval_items:
            return {"mae_s": None, "rmse_s": None, "mcra": None, "summary_rows": []}

        p         = model._params()
        horizon   = p["horizon"]
        seq_len   = p["seq_len"]

        summary_rows   = []
        per_bearing_cra: List[Optional[float]] = []

        for bearing_name, df_test in eval_items:
            if df_test is None or len(df_test) == 0:
                self.logger.warning(f"  [{bearing_name}] Could not load — skipping.")
                continue
            if TARGET not in df_test.columns:
                self.logger.warning(f"  [{bearing_name}] No '{TARGET}' column — skipping.")
                continue

            df_use = df_test.dropna(subset=[TARGET]).reset_index(drop=True)
            X_19   = self._project_to_cnn_lstm_cols(df_use).values.astype(np.float32)
            y_test = df_use[TARGET].values.astype(np.float64)

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

            preds = model.predict(X_scaled, conditions)   # raw seconds, shape (n_windows, horizon)
            if len(preds) == 0:
                self.logger.warning(f"  [{bearing_name}] Too few rows to predict — skipping.")
                continue

            # Trajectory: take horizon-step 0 as the "current burst" prediction
            preds_traj_s = preds[:, 0].astype(np.float64)
            # Align true RUL at the corresponding burst index (i + seq_len for each window i)
            y_traj_s     = y_test[seq_len: seq_len + len(preds_traj_s)]
            n = min(len(preds_traj_s), len(y_traj_s))
            preds_traj_s = preds_traj_s[:n]
            y_traj_s     = y_traj_s[:n]

            cra = _cra_for_bearing(y_traj_s, preds_traj_s)
            per_bearing_cra.append(cra)

            final_pred = float(preds_traj_s[-1])
            gt_rul     = ACTUAL_RUL_S.get(bearing_name, float(y_test[-1]))
            error_s    = final_pred - gt_rul
            abs_err_s  = abs(error_s)
            abs_pct    = abs_err_s / gt_rul * 100.0 if gt_rul != 0 else 0.0

            summary_rows.append({
                "bearing":     bearing_name,
                "pred_rul_s":  final_pred,
                "gt_rul_s":    gt_rul,
                "error_s":     error_s,
                "abs_err_s":   abs_err_s,
                "abs_pct_err": abs_pct,
                "cra":         cra,
            })
            self.logger.info(
                f"  [{bearing_name}] pred={final_pred:.0f}s  gt={gt_rul:.0f}s  "
                f"err={abs_err_s:.0f}s  CRA={cra if cra is not None else 'N/A'}"
            )

        return self._aggregate_metrics(summary_rows, per_bearing_cra)

    def _aggregate_metrics(
        self,
        summary_rows:    List[dict],
        per_bearing_cra: List[Optional[float]],
    ) -> Dict[str, Any]:
        """Aggregate per-bearing rows into the three whitelisted metrics."""
        if not summary_rows:
            self.logger.warning(
                "No evaluation data available — all metrics will be None."
            )
            return {
                "mae_s":        None,
                "rmse_s":       None,
                "mcra":         None,
                "summary_rows": [],
            }

        errors   = np.array([r["error_s"]   for r in summary_rows])
        abs_errs = np.array([r["abs_err_s"] for r in summary_rows])

        mae_s  = float(np.mean(abs_errs))
        rmse_s = float(np.sqrt(np.mean(errors ** 2)))
        mcra   = _mcra(per_bearing_cra)

        self.logger.info(
            f"  {'='*75}\n"
            f"  MAE_s : {mae_s:.0f} s  ({mae_s/60:.1f} min)\n"
            f"  RMSE_s: {rmse_s:.0f} s  ({rmse_s/60:.1f} min)\n"
            f"  MCRA  : {('%.4f' % mcra) if mcra is not None else 'N/A'}\n"
        )
        return {
            "mae_s":        mae_s,
            "rmse_s":       rmse_s,
            "mcra":         mcra,
            "summary_rows": summary_rows,
        }

    # ── Save / Register ───────────────────────────────────────────────────────

    def save_model(self, model: Union[MLPModel, CNNLSTMModel], scaler: StandardScaler) -> str:
        model_dir = os.path.join("model_registry", "models")
        os.makedirs(model_dir, exist_ok=True)
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type_tag = self.model_type.lower()
        model_path     = os.path.join(model_dir, f"rul_model_{model_type_tag}_{timestamp}.pt")
        model.scaler   = scaler
        model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        return model_path

    def save_results(self, metrics: Dict) -> str:
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
        """
        Register trained model in ModelRegistry.

        Only the three whitelisted metrics are persisted (mae_s, rmse_s, mcra) —
        anything else in `metrics` is dropped by ModelRegistry.register_model().
        """
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
                "mae_s":  metrics.get("mae_s"),
                "rmse_s": metrics.get("rmse_s"),
                "mcra":   metrics.get("mcra"),
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
        self.logger.info(
            f"[{run_id}] Starting {self.model_type} training pipeline."
        )
        if self.model_type == "CNN_LSTM":
            return self._run_cnn_lstm(run_id)
        return self._run_mlp(run_id)

    def _run_mlp(self, run_id: str) -> Dict[str, Any]:
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