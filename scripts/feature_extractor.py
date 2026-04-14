"""
scripts/feature_extractor.py
════════════════════════════════════════════════════════════════════════════════
Extracts burst-level statistical features and RUL labels directly from the
raw acc_*.csv files in a bearing folder — no parquet intermediate required.

Each acc_*.csv file is one burst (10-second recording). Files are read in
chronological order (sorted by filename), one at a time.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Burst-level statistical feature functions
# ──────────────────────────────────────────────────────────────────────────────

def _compute_skewness(x: np.ndarray) -> float:
    N, mu = len(x), np.mean(x)
    s3 = np.std(x, ddof=1) ** 3
    return (np.sum((x - mu) ** 3) / N) / s3 if s3 != 0 else 0.0


def _compute_kurtosis(x: np.ndarray) -> float:
    N = len(x)
    mu = np.mean(x)
    s4 = np.std(x, ddof=1) ** 4
    return (np.sum((x - mu) ** 4) / N) / s4 - 3 if s4 != 0 else 0.0


def _burst_features(sig: np.ndarray):
    mx    = float(np.max(sig))
    mn    = float(np.min(sig))
    mean  = float(np.mean(sig))
    sd    = float(np.std(sig, ddof=1))
    rms   = float(np.sqrt(np.mean(sig ** 2)))
    skew  = _compute_skewness(sig)
    kurt  = _compute_kurtosis(sig)
    crest = mx / rms   if rms  != 0 else 0.0
    form  = rms / mean if mean != 0 else 0.0
    return mx, mn, mean, sd, rms, skew, kurt, crest, form


# Column names for raw acc_*.csv files (PHM 2012 format)
_VIB_COLUMNS = ["Hour", "Minute", "Second", "Microsecond",
                 "Horizontal_Accel", "Vertical_Accel"]


def _read_acc_csv(file_path: Path) -> pd.DataFrame:
    """Read one acc_*.csv burst file. Returns empty DataFrame on failure."""
    try:
        df = pd.read_csv(file_path, header=None, names=_VIB_COLUMNS,
                         sep=None, engine="python")
        for col in _VIB_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


class FeatureExtractorPHM:
    """
    Extracts burst-level statistical features from raw acc_*.csv files.

    Input  : source folder containing acc_*.csv files
    Output : features.csv with 23 columns (file_id, burst_idx, time_s,
             18 features, RUL_s, RUL_norm)

    Config keys
    -----------
    input_location   : str   — folder containing acc_*.csv files
    output_location  : str   — path to write features.csv
    bearing_name     : str
    is_test          : bool  — if True, RUL is measured from last burst
    burst_period     : float — seconds per burst (default 10.0)
    failure_threshold: float — peak g threshold for failure detection (default 20.0)
    n_consecutive    : int   — bursts above threshold before declaring failure (default 1)
    """

    def __init__(self, config: dict):
        self.bearing_name      = config.get("bearing_name", "UnknownBearing")
        self.is_test           = bool(config.get("is_test", False))
        self.burst_period      = float(config.get("burst_period", 10.0))
        self.failure_threshold = float(config.get("failure_threshold", 20.0))
        self.n_consecutive     = int(config.get("n_consecutive", 1))
        log_path               = config.get("log_path")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
        else:
            logging.basicConfig(level=logging.INFO)

        self.input_path  = config.get("input_location")   # source folder
        self.output_path = config.get("output_location")  # features.csv path

    # ------------------------------------------------------------------
    # Failure detection
    # ------------------------------------------------------------------

    def _find_failure_time_s(self, df: pd.DataFrame) -> float:
        df   = df.sort_values("time_s").reset_index(drop=True)
        peak = df[["h_max", "v_max"]].abs().max(axis=1).values
        above = peak >= self.failure_threshold

        for i in range(len(above) - self.n_consecutive + 1):
            if above[i : i + self.n_consecutive].all():
                failure_s = float(df.loc[i, "time_s"])
                self.logger.info(
                    f"[{self.bearing_name}] Failure detected at {failure_s:.0f} s "
                    f"(sustained >{self.failure_threshold} g for "
                    f"{self.n_consecutive} consecutive bursts)."
                )
                return failure_s

        fallback = float(df["time_s"].max())
        self.logger.warning(
            f"[{self.bearing_name}] Threshold never sustained — "
            f"using last burst ({fallback:.0f} s) as failure point."
        )
        return fallback

    # ------------------------------------------------------------------
    # Main extraction  (reads directly from acc_*.csv — no parquet)
    # ------------------------------------------------------------------

    def extract_phm_features(self, source_folder: str, output_path: str) -> pd.DataFrame:
        """
        Read every acc_*.csv file in source_folder in chronological order,
        compute burst-level features, label RUL, and write features.csv.
        """
        acc_files = sorted(Path(source_folder).rglob("acc_*.csv"))
        if not acc_files:
            raise FileNotFoundError(
                f"No acc_*.csv files found in {source_folder}"
            )

        self.logger.info(
            f"[{self.bearing_name}] Extracting features from "
            f"{len(acc_files)} acc_*.csv files in {source_folder}"
        )

        rows = []
        for burst_idx, file_path in enumerate(acc_files):
            df_raw = _read_acc_csv(file_path)
            if df_raw.empty:
                self.logger.warning(f"  Skipping empty file: {file_path.name}")
                continue

            h = df_raw["Horizontal_Accel"].values.astype(np.float32)
            v = df_raw["Vertical_Accel"].values.astype(np.float32)

            h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form = _burst_features(h)
            v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form = _burst_features(v)

            time_s = burst_idx * self.burst_period

            rows.append([
                file_path.stem, burst_idx, time_s,
                h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form,
                v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form,
            ])

        cols = [
            "file_id", "burst_idx", "time_s",
            "h_max", "h_min", "h_mean", "h_sd", "h_rms", "h_skew", "h_kurt", "h_crest", "h_form",
            "v_max", "v_min", "v_mean", "v_sd", "v_rms", "v_skew", "v_kurt", "v_crest", "v_form",
        ]
        df = pd.DataFrame(rows, columns=cols).sort_values("time_s").reset_index(drop=True)

        self.logger.info(
            f"[{self.bearing_name}] Loaded {len(df)} bursts from raw acc_*.csv files"
        )

        # RUL labelling
        if not self.is_test:
            failure_s      = self._find_failure_time_s(df)
            df             = df[df["time_s"] <= failure_s].copy().reset_index(drop=True)
            df["RUL_s"]    = (failure_s - df["time_s"]).clip(lower=0.0)
            df["RUL_norm"] = df["RUL_s"] / failure_s
            self.logger.info(
                f"[{self.bearing_name}] Total life: {failure_s:.0f} s "
                f"({failure_s/3600:.2f} h) | {len(df)} bursts kept"
            )
        else:
            last_s         = float(df["time_s"].max())
            df["RUL_s"]    = (last_s - df["time_s"]).clip(lower=0.0)
            df["RUL_norm"] = df["RUL_s"] / last_s if last_s > 0 else 0.0
            self.logger.info(
                f"[{self.bearing_name}] Recording: {last_s:.0f} s "
                f"({last_s/3600:.2f} h) | {len(df)} bursts | RUL_s=0 at last burst"
            )

        self.logger.info(
            f"[{self.bearing_name}] Features: {df.shape[1]} columns | "
            f"RUL_s: {df['RUL_s'].max():.1f} s -> {df['RUL_s'].min():.1f} s"
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"[{self.bearing_name}] Features saved to {output_path}")
        return df

    # ------------------------------------------------------------------
    # Orchestrator entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Wrapper for orchestrator compatibility.
        Skips extraction if features.csv already exists.
        input_location is the bearing source folder (containing acc_*.csv).
        output_location is the full path to features.csv.
        """
        if not self.input_path:
            raise ValueError("No input_location provided for feature extraction.")
        if not self.output_path:
            raise ValueError("No output_location provided for feature extraction.")

        if os.path.exists(self.output_path):
            self.logger.info(
                f"[{self.bearing_name}] Features already exist at "
                f"{self.output_path} - skipping."
            )
            df = pd.read_csv(self.output_path)
            return {
                "features_csv": self.output_path,
                "bearing_name": self.bearing_name,
                "num_bursts":   len(df),
                "num_features": df.shape[1],
                "cached":       True,
            }

        df = self.extract_phm_features(self.input_path, self.output_path)
        return {
            "features_csv": self.output_path,
            "bearing_name": self.bearing_name,
            "num_bursts":   len(df),
            "num_features": df.shape[1],
            "cached":       False,
        }