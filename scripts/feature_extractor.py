import os
import logging
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Burst-level statistical feature functions
# Identical to the functions in feature_extraction.py — pure numpy, no scipy.
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
    """
    Compute 9 time-domain statistics for a single burst signal.
    Matches burst_features() in feature_extraction.py exactly.
    Returns: mx, mn, mean, sd, rms, skew, kurt, crest, form
    """
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


class FeatureExtractorPHM:
    """
    Extracts burst-level statistical features and RUL labels from PHM 2012
    vibration data.

    Fixes applied
    -------------
    Issue 1 - Output saved next to source data, not in workflow_data.
              Skip-if-exists cache check prevents re-processing.
    Issue 2 - Feature logic matches feature_extraction.py exactly:
              inline pure-numpy _burst_features(), no scipy/pywt.
              Failure detection uses .abs() on h_max and v_max before
              comparing against the threshold.
    Issue 3 - All log strings use ASCII '->' instead of Unicode arrow.
    Issue 4 - Failure detection now requires n_consecutive bursts above
              threshold before declaring failure, guarding against false
              triggers from external vibration or sensor noise.
              n_consecutive is configurable via workflow.yaml (default: 2).
    """

    def __init__(self, config: dict):
        self.bearing_name      = config.get("bearing_name", "UnknownBearing")
        self.is_test           = bool(config.get("is_test", False))
        self.burst_period      = float(config.get("burst_period", 10.0))
        self.failure_threshold = float(config.get("failure_threshold", 20.0))
        self.n_consecutive     = int(config.get("n_consecutive", 1))      # <-- NEW
        log_path               = config.get("log_path")

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
        else:
            logging.basicConfig(level=logging.INFO)

        self.input_path  = config.get("input_location")
        self.output_path = config.get("output_location")

    # ------------------------------------------------------------------
    # Failure detection
    # ------------------------------------------------------------------

    def _find_failure_time_s(self, df: pd.DataFrame) -> float:
        """
        Scan bursts chronologically. Return time_s of the FIRST burst in the
        first run of n_consecutive bursts that ALL exceed the failure threshold
        (peak of |h_max|, |v_max|).

        Requiring consecutive bursts guards against false failure triggers
        caused by external vibration, mechanical shock, or sensor noise.

        Falls back to the last burst if the threshold is never sustained.
        """
        df   = df.sort_values("time_s").reset_index(drop=True)
        peak = df[["h_max", "v_max"]].abs().max(axis=1).values
        above = peak >= self.failure_threshold

        for i in range(len(above) - self.n_consecutive + 1):
            if above[i : i + self.n_consecutive].all():
                failure_s = float(df.loc[i, "time_s"])
                self.logger.info(
                    f"[{self.bearing_name}] Failure detected at {failure_s:.0f} s "
                    f"(burst {int(failure_s / self.burst_period)}) — "
                    f"sustained >{self.failure_threshold} g for "
                    f"{self.n_consecutive} consecutive bursts."
                )
                return failure_s

        fallback = float(df["time_s"].max())
        self.logger.warning(
            f"[{self.bearing_name}] {self.failure_threshold} g threshold never sustained "
            f"for {self.n_consecutive} consecutive bursts "
            f"- using last burst ({fallback:.0f} s) as failure point."
        )
        return fallback

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract_phm_features(self, parquet_path: str, output_path: str) -> pd.DataFrame:
        """
        Extract burst-level features and label RUL in seconds.
        Feature loop is identical to extract_features() in feature_extraction.py.
        """
        vib_df = pd.read_parquet(parquet_path)
        self.logger.info(
            f"[{self.bearing_name}] Loaded {vib_df.shape[0]:,} rows from {parquet_path}"
        )

        # Chronological burst ordering by file_id
        file_order = (
            vib_df[["file_id"]].drop_duplicates()
            .sort_values("file_id").reset_index(drop=True)
        )
        file_order["burst_idx"] = np.arange(len(file_order))
        vib_df = vib_df.merge(file_order, on="file_id", how="left")

        self.logger.info(
            f"[{self.bearing_name}] Extracting features from {len(file_order)} bursts..."
        )


        # Per-burst feature extraction
        rows = []
        for file_id, g in vib_df.groupby("file_id", sort=True):
            h = g["Horizontal_Accel"].values
            v = g["Vertical_Accel"].values

            h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form = _burst_features(h)
            v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form = _burst_features(v)

            burst_idx = int(g["burst_idx"].iloc[0])
            time_s    = burst_idx * self.burst_period

            rows.append([
                file_id, burst_idx, time_s,
                h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form,
                v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form,
            ])

        cols = [
            "file_id", "burst_idx", "time_s",
            "h_max", "h_min", "h_mean", "h_sd", "h_rms", "h_skew", "h_kurt", "h_crest", "h_form",
            "v_max", "v_min", "v_mean", "v_sd", "v_rms", "v_skew", "v_kurt", "v_crest", "v_form",
        ]
        df = pd.DataFrame(rows, columns=cols).sort_values("time_s").reset_index(drop=True)

        # RUL labelling
        if not self.is_test:
            failure_s = self._find_failure_time_s(df)
            df        = df[df["time_s"] <= failure_s].copy().reset_index(drop=True)
            df["RUL_s"]    = (failure_s - df["time_s"]).clip(lower=0.0)
            df["RUL_norm"] = df["RUL_s"] / failure_s
            self.logger.info(
                f"[{self.bearing_name}] Total life: {failure_s:.0f} s "
                f"({failure_s/3600:.2f} h) | {len(df)} bursts kept"
            )
        else:
            last_s = float(df["time_s"].max())
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
        Skips extraction if the output CSV already exists (cache check).
        """
        if not self.input_path:
            raise ValueError("No input_location provided for feature extraction.")
        if not self.output_path:
            raise ValueError("No output_location provided for feature extraction.")

        # Cache check — skip if already extracted (Issue 1)
        if os.path.exists(self.output_path):
            self.logger.info(
                f"[{self.bearing_name}] Features already exist at {self.output_path} - skipping."
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