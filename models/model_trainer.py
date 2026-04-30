"""
models/model_trainer.py
═══════════════════════════════════════════════════════════════════════════════
Unified trainer for PHM 2012 RUL models.

Pipeline:
    load_data() -> train() -> evaluate() -> save_model() -> register_model() -> run(run_id)

Data sources
────────────
Primary  : DataFrames loaded from MongoDB factory_features by the orchestrator
           and passed in via config keys train_dataframes / val_dataframes /
           test_dataframes.  The orchestrator never passes CSV file paths for
           training data — it reads from MongoDB first.

Fallback : CSV file paths via train_files / val_files / test_files — kept
           for backward compatibility with external scripts (IEEE original
           code, standalone notebooks) that still run directly from disk.

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
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

from utils.model_registry import ModelRegistry
from models.rul_net_model import RULNetModel

logger = logging.getLogger(__name__)

# GT RULs — used for point-in-time evaluation at last recorded burst.
# For train/val bearings these are derived from the failure point in the
# feature data.  For test bearings these are known from when the recording
# was stopped.
ACTUAL_RUL_S = {
    "Bearing1_3": 5730,  "Bearing1_4":  339,  "Bearing1_5": 1610,
    "Bearing1_6": 1460,  "Bearing1_7": 7570,  "Bearing2_3": 7530,
    "Bearing2_4": 1390,  "Bearing2_5": 3090,  "Bearing2_6": 1290,
    "Bearing2_7":  580,  "Bearing3_3":  820,
}

TARGET    = "RUL_s"
DROP_COLS = ["file_id", "burst_idx", "time_s", TARGET]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RULTrainerPHM:
    """
    Unified trainer for PHM 2012 RUL models.

    Config keys — MongoDB path (preferred, used by orchestrator)
    ─────────────────────────────────────────────────────────────
    train_dataframes : list[pd.DataFrame]         — training bearing DataFrames
    val_dataframes   : list[pd.DataFrame]         — validation bearing DataFrames
    test_dataframes  : list[tuple[str, pd.DataFrame]] — (bearing_name, df) pairs
    output_location  : str
    window_size      : int
    rul_scale        : float
    model_params     : dict
    state_location   : str   (optional)
    log_path         : str   (optional)

    Config keys — CSV fallback (external scripts / backward compatibility)
    ──────────────────────────────────────────────────────────────────────
    train_files      : list[str]  — feature CSV paths for training bearings
    val_files        : list[str]  — feature CSV paths for validation bearings
    test_files       : list[str]  — feature CSV paths for test bearings
    """

    def __init__(self, config: dict):
        # ── MongoDB DataFrame path ────────────────────────────────────────────
        # list[pd.DataFrame] for train/val; list[(name, df)] for test
        self.train_dataframes = config.get("train_dataframes", [])
        self.val_dataframes   = config.get("val_dataframes",   [])
        self.test_dataframes  = config.get("test_dataframes",  [])  # list[(str, df)]

        # ── Legacy CSV path (external scripts / backward compat) ──────────────
        self.train_files = config.get("train_files", [])
        self.val_files   = config.get("val_files",   [])
        self.test_files  = config.get("test_files",  [])

        self.output_location = config.get("output_location", "workflow_data/models")
        self.window_size     = config.get("window_size", 40)
        self.rul_scale       = config.get("rul_scale", 30000.0)
        self.model_params    = config.get("model_params", {})
        self.state_location  = config.get("state_location")
        self.log_path        = config.get("log_path")

        # Inject rul_scale into model_params so the model knows how to unscale
        if "rul_scale" not in self.model_params:
            self.model_params["rul_scale"] = self.rul_scale

        self.logger = logging.getLogger(__name__)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_rolling_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add rolling mean/std/slope features matching the serving pipeline."""
        if df is None or len(df) < self.window_size:
            return None
        if TARGET not in df.columns:
            self.logger.warning(f"  No '{TARGET}' column — skipping DataFrame.")
            return None

        # Drop all non-numeric columns before rolling.
        # MongoDB documents carry extra string fields (dataset_id, version, etc.)
        # that must be removed before any rolling operation is attempted.
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        keep_cols    = list(dict.fromkeys(numeric_cols + [c for c in DROP_COLS if c in df.columns]))
        df = df[keep_cols].copy()

        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        if len(df) < self.window_size:
            return None

        # Only roll over numeric feature columns (exclude DROP_COLS)
        base_cols = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
        new_cols  = {}
        for col in base_cols:
            new_cols[f"{col}_mean"]  = df[col].rolling(self.window_size).mean()
            new_cols[f"{col}_std"]   = df[col].rolling(self.window_size).std()
            new_cols[f"{col}_slope"] = df[col].rolling(self.window_size).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
            )
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df.dropna().reset_index(drop=True)

    def _load_bearing(self, path: str) -> Optional[pd.DataFrame]:
        """Load a bearing features.csv from disk and add rolling features.
        Used only by the CSV fallback path."""
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

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def load_data(self) -> tuple:
        """
        Load and prepare training and validation data.

        Uses MongoDB DataFrames if available (orchestrator path), otherwise
        falls back to reading from CSV files (external scripts path).

        Returns
        -------
        X_train, y_train_scaled, X_val, y_val, scaler
        """
        # ── MongoDB DataFrame path (preferred) ────────────────────────────────
        if self.train_dataframes:
            self.logger.info(
                f"Loading data from {len(self.train_dataframes)} train DataFrame(s) "
                f"and {len(self.val_dataframes)} val DataFrame(s) [MongoDB path]."
            )
            train_dfs = []
            for df in self.train_dataframes:
                result = self._add_rolling_features(df.copy())
                if result is not None and len(result) > 0:
                    train_dfs.append(result)
                else:
                    self.logger.warning(
                        f"  A train DataFrame was skipped "
                        f"(too short or missing '{TARGET}' column)."
                    )

            val_dfs = []
            for df in self.val_dataframes:
                result = self._add_rolling_features(df.copy())
                if result is not None and len(result) > 0:
                    val_dfs.append(result)
                else:
                    self.logger.warning(
                        f"  A val DataFrame was skipped "
                        f"(too short or missing '{TARGET}' column)."
                    )

        # ── CSV fallback path (external scripts / backward compat) ────────────
        else:
            self.logger.info(
                f"Loading data from {len(self.train_files)} train CSV(s) "
                f"and {len(self.val_files)} val CSV(s) [CSV fallback path]."
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

        X_train, y_train = self._get_xy(train_combined)
        X_val,   y_val   = self._get_xy(val_combined)

        # Scale train target only
        y_train_scaled = y_train / self.rul_scale

        scaler       = StandardScaler()
        drop         = [c for c in DROP_COLS if c in train_combined.columns]
        feature_cols = [c for c in train_combined.columns if c not in drop]
        X_train      = scaler.fit_transform(pd.DataFrame(X_train, columns=feature_cols))
        X_val        = scaler.transform(pd.DataFrame(X_val, columns=feature_cols))

        self.logger.info(f"Training features ({len(feature_cols)}): {feature_cols}")
        self.logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")
        self.logger.info(
            f"RUL_s train range (scaled): "
            f"{y_train_scaled.min():.4f} -> {y_train_scaled.max():.4f}"
        )
        self.logger.info(
            f"RUL_s val range: {y_val.min():.0f} s -> {y_val.max():.0f} s"
        )

        return X_train, y_train_scaled, X_val, y_val, scaler

    def train(self, X_train, y_train, X_val, y_val) -> RULNetModel:
        """Train model and return trained RULNetModel instance."""
        self.logger.info(f"Training RULNetModel with params: {self.model_params}")
        model = RULNetModel(**self.model_params)
        model.train(X_train, y_train, X_val, y_val)
        return model

    def evaluate(self, model: RULNetModel, scaler: StandardScaler) -> Dict[str, Any]:
        """
        Evaluate on test bearings. Returns per-bearing results and mean metrics.

        Metrics computed per bearing (at final recorded burst):
            mae_s     — |predicted_RUL - gt_RUL| in seconds
            rmse_s    — sqrt(mean(errors^2)) across all test bearings
            mape      — mean absolute percentage error

        Uses test_dataframes if available (MongoDB path), falls back to
        test_files CSV paths, then falls back to val data if neither is set.
        """
        # ── Resolve evaluation items: list of (bearing_name, df) ─────────────
        if self.test_dataframes:
            # MongoDB path: list of (name, df) tuples
            self.logger.info(
                f"Evaluating on {len(self.test_dataframes)} test DataFrame(s) [MongoDB path]."
            )
            eval_items: List[Tuple[str, Optional[pd.DataFrame]]] = [
                (name, self._add_rolling_features(df.copy()))
                for name, df in self.test_dataframes
            ]
        elif self.test_files:
            # CSV fallback path
            self.logger.info(
                f"Evaluating on {len(self.test_files)} test CSV(s) [CSV fallback path]."
            )
            eval_items = [
                (self._parse_bearing_name(f), self._load_bearing(f))
                for f in self.test_files
            ]
        elif self.val_dataframes:
            # No test set — fall back to val DataFrames
            self.logger.info(
                "No test data configured — falling back to val DataFrames for evaluation."
            )
            eval_items = [
                (f"val_{i}", self._add_rolling_features(df.copy()))
                for i, df in enumerate(self.val_dataframes)
            ]
        else:
            # No test set — fall back to val CSVs
            self.logger.info(
                "No test files configured — falling back to val bearings for evaluation."
            )
            eval_items = [
                (self._parse_bearing_name(f), self._load_bearing(f))
                for f in self.val_files
            ]

        if not eval_items:
            self.logger.warning(
                "No evaluation data available — all metrics will be None. "
                "Check that test data is set in bearings config and features exist."
            )
            return {
                "mae_s":        None,
                "rmse_s":       None,
                "mape":         None,
                "mean_abs_pct": None,
                "summary_rows": [],
            }

        summary_rows = []
        for bearing_name, df_test in eval_items:
            if df_test is None or len(df_test) == 0:
                self.logger.warning(f"  [{bearing_name}] Could not load — skipping.")
                continue

            X_test, y_test = self._get_xy(df_test)
            X_test         = scaler.transform(
                pd.DataFrame(X_test, columns=scaler.feature_names_in_)
            )
            preds = model.predict(X_test)   # (n_windows, horizon), raw seconds

            pred_rul_s = float(np.clip(preds[-1, 0], 0, None))
            gt_rul_s   = ACTUAL_RUL_S.get(bearing_name)

            if gt_rul_s is None:
                # If not in ACTUAL_RUL_S, derive from last row of feature data
                gt_rul_s = float(df_test[TARGET].iloc[-1])
                self.logger.info(
                    f"  {bearing_name}: GT RUL derived from last feature row "
                    f"= {gt_rul_s:.0f} s"
                )

            error_s   = pred_rul_s - gt_rul_s
            abs_err_s = abs(error_s)
            abs_pct   = abs_err_s / gt_rul_s * 100 if gt_rul_s > 0 else 0.0

            summary_rows.append({
                "bearing":     bearing_name,
                "gt_rul_s":    gt_rul_s,
                "pred_rul_s":  pred_rul_s,
                "error_s":     error_s,
                "abs_err_s":   abs_err_s,
                "abs_pct_err": abs_pct,
                "timestamp":   datetime.now().isoformat(),
            })

        if not summary_rows:
            self.logger.warning(
                "No test bearings evaluated — all metrics will be None. "
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
            "mean_abs_pct": mape,   # backward compatibility alias
            "summary_rows": summary_rows,
        }

    def save_model(self, model: RULNetModel, scaler: StandardScaler) -> str:
        """
        Save model checkpoint (includes scaler) to the global model store.
        Returns saved path.
        """
        model_dir  = os.path.join("model_registry", "models")
        os.makedirs(model_dir, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"rul_model_{timestamp}.pt")
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
        model:      RULNetModel,
        model_path: str,
        metrics:    Dict,
    ) -> str:
        """
        Register trained model in ModelRegistry with all 4 metrics.
        Returns model_id.
        """
        registry = ModelRegistry()

        # Report the actual data source in training_data_info
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
        try:
            X_train, y_train, X_val, y_val, scaler = self.load_data()
            model       = self.train(X_train, y_train, X_val, y_val)
            metrics     = self.evaluate(model, scaler)
            model_path  = self.save_model(model, scaler)
            results_csv = self.save_results(metrics)
            model_id    = self.register_model(run_id, model, model_path, metrics)

            if self.state_location:
                os.makedirs(os.path.dirname(self.state_location), exist_ok=True)
                with open(self.state_location, "w") as f:
                    f.write("complete")

            return {
                "model_id":     model_id,
                "model_path":   model_path,
                "results_csv":  results_csv,
                "mae_s":        metrics["mae_s"],
                "rmse_s":       metrics["rmse_s"],
                "mape":         metrics["mape"],
                "mean_abs_pct": metrics["mean_abs_pct"],
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise