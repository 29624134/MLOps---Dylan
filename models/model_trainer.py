import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

from utils.model_registry import ModelRegistry
from models.rul_net_model import RULNetModel

logger = logging.getLogger(__name__)

# GT RULs from PHM 2012 Table 3 — scoring only, never used in training
ACTUAL_RUL_S = {
    "Bearing1_3": 5730,  "Bearing1_4":  339,  "Bearing1_5": 1610,
    "Bearing1_6": 1460,  "Bearing1_7": 7570,  "Bearing2_3": 7530,
    "Bearing2_4": 1390,  "Bearing2_5": 3090,  "Bearing2_6": 1290,
    "Bearing2_7":  580,  "Bearing3_3":  820,
}

TARGET    = "RUL_s"
DROP_COLS = ["file_id", "burst_idx", "time_s", TARGET]


class RULTrainerPHM:
    """
    Unified trainer for PHM 2012 RUL models.

    Mirrors the interface of ModelTrainer:
        load_data() -> train() -> evaluate() -> save_model() -> register_model() -> run(run_id)

    Config keys
    -----------
    train_files      : list[str]  — feature CSV paths for training bearings
    val_files        : list[str]  — feature CSV paths for validation bearings
    test_files       : list[str]  — feature CSV paths for test bearings
    output_location  : str        — folder to save model and results
    window_size      : int        — rolling feature window (default: 40)
    rul_scale        : float      — RUL normalisation factor (default: 30000.0)
    model_params     : dict       — hyperparameters forwarded to RULNetModel
    state_location   : str        — completion flag path (optional)
    log_path         : str        — log file path (optional)
    """

    def __init__(self, config: dict):
        self.train_files     = config.get("train_files", [])
        self.val_files       = config.get("val_files", [])
        self.test_files      = config.get("test_files", [])
        self.output_location = config.get("output_location")
        self.state_location  = config.get("state_location")
        self.window_size     = int(config.get("window_size", 40))
        self.rul_scale       = float(config.get("rul_scale", 30000.0))
        self.model_params    = config.get("model_params", {})
        self.config          = config

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        log_path = config.get("log_path")
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
        else:
            logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c not in DROP_COLS]
        new_cols = {}
        for col in feature_cols:
            new_cols[f"{col}_mean"]  = df[col].rolling(self.window_size).mean()
            new_cols[f"{col}_std"]   = df[col].rolling(self.window_size).std()
            new_cols[f"{col}_slope"] = df[col].rolling(self.window_size).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
            )
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df.dropna().reset_index(drop=True)

    def _load_bearing(self, csv_path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(csv_path):
            self.logger.warning(f"File not found, skipping: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        if TARGET not in df.columns:
            raise ValueError(f"'{TARGET}' column not found in {csv_path}")
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        if len(df) < self.window_size:
            self.logger.warning(f"Too few rows ({len(df)}) in {csv_path}, skipping.")
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

    # ------------------------------------------------------------------
    # Pipeline steps  (mirrors ModelTrainer)
    # ------------------------------------------------------------------

    def load_data(self) -> tuple:
        train_dfs = [df for f in self.train_files if (df := self._load_bearing(f)) is not None]
        val_dfs = [df for f in self.val_files if (df := self._load_bearing(f)) is not None]

        if not train_dfs:
            raise RuntimeError("No training data could be loaded.")
        if not val_dfs:
            raise RuntimeError("No validation data could be loaded.")

        train_combined = pd.concat(train_dfs, axis=0).reset_index(drop=True)
        val_combined = pd.concat(val_dfs, axis=0).reset_index(drop=True)

        X_train, y_train = self._get_xy(train_combined)
        X_val, y_val = self._get_xy(val_combined)

        # ONLY train target is scaled
        y_train_scaled = y_train / self.rul_scale

        scaler = StandardScaler()
        drop = [c for c in DROP_COLS if c in train_combined.columns]
        feature_cols = [c for c in train_combined.columns if c not in drop]
        X_train = scaler.fit_transform(pd.DataFrame(X_train, columns=feature_cols))
        X_val = scaler.transform(pd.DataFrame(X_val, columns=feature_cols))

        self.logger.info(f"Training features ({len(feature_cols)}): {feature_cols}")
        self.logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")
        self.logger.info(f"RUL_s train range (scaled): {y_train_scaled.min():.4f} -> {y_train_scaled.max():.4f}")
        self.logger.info(f"RUL_s val range: {y_val.min():.0f} s -> {y_val.max():.0f} s")

        return X_train, y_train_scaled, X_val, y_val, scaler

    def train(self, X_train, y_train, X_val, y_val) -> RULNetModel:
        """Train model and return trained RULNetModel instance."""
        self.logger.info(f"Training RULNetModel with params: {self.model_params}")
        model = RULNetModel(**self.model_params)
        model.train(X_train, y_train, X_val, y_val)
        return model

    def evaluate(self, model: RULNetModel, scaler: StandardScaler) -> Dict[str, float]:
        """Evaluate on test bearings. Returns per-bearing results and mean metrics."""
        self.logger.info(
            f"\n{'='*65}\n"
            f"  {'Bearing':<12} {'GT RUL':>9} {'Pred RUL':>10} "
            f"{'Error':>9} {'Abs%Err':>9}\n"
            f"  {'-'*12} {'-'*9} {'-'*10} {'-'*9} {'-'*9}"
        )

        summary_rows = []
        for test_path in self.test_files:
            bearing_name = self._parse_bearing_name(test_path)
            df_test      = self._load_bearing(test_path)
            if df_test is None:
                self.logger.warning(f"  [{bearing_name}] Could not load — skipping.")
                continue

            X_test, y_test = self._get_xy(df_test)
            X_test = scaler.transform(pd.DataFrame(X_test, columns=scaler.feature_names_in_))
            preds          = model.predict(X_test)   # (n_windows, horizon), raw seconds

            pred_rul_s = float(np.clip(preds[-1, 0], 0, None))
            gt_rul_s   = ACTUAL_RUL_S.get(bearing_name)
            if gt_rul_s is None:
                self.logger.warning(f"  {bearing_name}: no GT entry")
                continue

            error_s = pred_rul_s - gt_rul_s
            abs_pct = abs(error_s) / gt_rul_s * 100

            self.logger.info(
                f"  {bearing_name:<12} {gt_rul_s:>7.0f} s "
                f"{pred_rul_s:>8.0f} s {error_s:>+8.0f} s {abs_pct:>8.1f}%"
            )
            summary_rows.append({
                "bearing":     bearing_name,
                "gt_rul_s":    gt_rul_s,
                "pred_rul_s":  pred_rul_s,
                "error_s":     error_s,
                "abs_pct_err": abs_pct,
                "timestamp":   datetime.now().isoformat(),
            })

        mean_mae = float(np.mean([abs(r["error_s"])   for r in summary_rows])) if summary_rows else None
        mean_pct = float(np.mean([r["abs_pct_err"]    for r in summary_rows])) if summary_rows else None

        if mean_mae is not None:
            self.logger.info(
                f"  {'-'*65}\n"
                f"  Mean |error|: {mean_mae:.0f} s | Mean |%err|: {mean_pct:.1f}%"
            )

        return {
            "mae_s":          mean_mae,
            "mean_abs_pct":   mean_pct,
            "summary_rows":   summary_rows,
        }

    def save_model(self, model: RULNetModel, scaler: StandardScaler) -> str:
        """
        Save model checkpoint (includes scaler) to the global model store.
        Returns saved path.

        Models are saved to model_registry/models/ with a timestamp-based
        filename so each training run produces a uniquely named file,
        independent of run_id scoping.
        """
        model_dir = os.path.join("model_registry", "models")
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

    def register_model(self, run_id: str, model: RULNetModel,
                       model_path: str, metrics: Dict) -> str:
        """Register trained model in ModelRegistry. Returns model_id."""
        registry = ModelRegistry()

        training_data_info = {
            "num_train_files": len(self.train_files),
            "num_val_files":   len(self.val_files),
            "window_size":     self.window_size,
            "source":          self.train_files,
        }

        model_id = registry.register_model(
            model_path        = model_path,
            model_type        = model.get_model_name(),
            target_feature    = TARGET,
            metrics           = {"mae_s": metrics["mae_s"], "mean_abs_pct": metrics["mean_abs_pct"]},
            training_data_info= training_data_info,
            metadata          = {"run_id": run_id, "hyperparameters": self.model_params},
        )

        self.logger.info(f"Model registered with ID: {model_id}")
        return model_id

    # ------------------------------------------------------------------
    # Orchestrator entry point
    # ------------------------------------------------------------------

    def run(self, run_id: str) -> Dict[str, Any]:
        """
        Execute the full training pipeline.
        Returns dict with model_id, model_path, results_csv, and metrics.
        """
        try:
            X_train, y_train, X_val, y_val, scaler = self.load_data()
            model      = self.train(X_train, y_train, X_val, y_val)
            metrics    = self.evaluate(model, scaler)
            model_path = self.save_model(model, scaler)
            results_csv= self.save_results(metrics)
            model_id   = self.register_model(run_id, model, model_path, metrics)

            if self.state_location:
                os.makedirs(os.path.dirname(self.state_location), exist_ok=True)
                with open(self.state_location, "w") as f:
                    f.write("complete")

            return {
                "model_id":       model_id,
                "model_path":     model_path,
                "results_csv":    results_csv,
                "mae_s":          metrics["mae_s"],
                "mean_abs_pct":   metrics["mean_abs_pct"],
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise