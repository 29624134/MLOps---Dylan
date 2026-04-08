"""
serving_pipeline/monitoring.py
═══════════════════════════════════════════════════════════════════════════════
Serving Pipeline — Stage 4: MLOps Monitoring

Per MLOps Project Instructions §3: full model-performance monitoring is out of
scope. This stage performs DATA-CENTRIC monitoring on incoming sensor data:

    1. Descriptive statistics   — mean, variance, min, max per feature
    2. Distribution drift       — compare current batch stats against a stored
                                  baseline (Population Stability Index style:
                                  z-score of mean shift vs baseline std)
    3. Anomaly flagging         — flag the batch if aggregate drift exceeds
                                  a configurable threshold

Output flows to the Dashboard (via Serving History / pipeline output).

Baseline management
───────────────────
The baseline is computed from the first N bursts seen (warm-up window) and
persisted to a JSON file so it survives process restarts.  If no baseline
exists yet the batch is not flagged as drifted.

Usage
─────
    from serving_pipeline.monitoring import MLOpsMonitor

    monitor = MLOpsMonitor(baseline_path="model_registry/monitoring_baseline.json")

    result = monitor.run(feature_engineering_output)
    # result["stats"]             — per-feature descriptive stats
    # result["drift_detected"]    — bool
    # result["drift_features"]    — list of feature names with significant drift
    # result["anomaly_flag"]      — bool (batch-level flag)
    # result["baseline_ready"]    — bool
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_BASELINE_PATH  = "model_registry/monitoring_baseline.json"
_BASELINE_WARMUP_BURSTS = 50      # collect this many bursts before computing baseline
_DRIFT_Z_THRESH         = 3.0    # |z| > 3 standard deviations from baseline mean
_ANOMALY_DRIFT_FRAC     = 0.20   # flag batch if >20% of features show drift


class MLOpsMonitor:
    """
    Stage 4 of the Serving Pipeline — data-centric monitoring.

    Parameters
    ──────────
    baseline_path : str
        Path to JSON file where the baseline statistics are persisted.
    drift_z_thresh : float
        Z-score threshold for flagging a feature as drifted.
    anomaly_drift_frac : float
        Fraction of drifted features that triggers a batch-level anomaly flag.
    """

    def __init__(
        self,
        baseline_path:      str   = _DEFAULT_BASELINE_PATH,
        drift_z_thresh:     float = _DRIFT_Z_THRESH,
        anomaly_drift_frac: float = _ANOMALY_DRIFT_FRAC,
    ):
        self._baseline_path      = baseline_path
        self._drift_z_thresh     = drift_z_thresh
        self._anomaly_drift_frac = anomaly_drift_frac

        self._warmup_buffer: List[np.ndarray] = []
        self._baseline: Optional[Dict[str, Any]] = None

        self._load_baseline()
        logger.info(
            f"[Monitor] MLOpsMonitor initialised — "
            f"baseline={'loaded' if self._baseline else 'not yet computed'}  "
            f"drift_z={drift_z_thresh}  anomaly_frac={anomaly_drift_frac}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, fe_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run data-centric monitoring on one burst's feature vector.

        Parameters
        ──────────
        fe_output : dict produced by ServingFeatureEngineer.process_burst()

        Returns
        ───────
        dict with keys:
            stats           : dict   — per-feature {mean, var, min, max}
            drift_detected  : bool
            drift_features  : list[str]
            anomaly_flag    : bool
            baseline_ready  : bool
            n_features      : int
            checked_at      : str    — ISO-8601 UTC
        """
        if not fe_output.get("ready", False):
            return self._empty_result(fe_output, reason="buffer_not_ready")

        feature_vector = fe_output["feature_vector"]
        feature_names  = fe_output["feature_names"]

        # 1. Compute descriptive stats for this burst
        stats = self._compute_stats(feature_vector, feature_names)

        # 2. Warm-up: collect vectors until baseline is established
        self._warmup_buffer.append(feature_vector)
        if self._baseline is None and len(self._warmup_buffer) >= _BASELINE_WARMUP_BURSTS:
            self._compute_and_save_baseline(feature_names)

        # 3. Drift detection (only once baseline is ready)
        drift_detected  = False
        drift_features: List[str] = []
        anomaly_flag    = False

        if self._baseline is not None:
            drift_features = self._detect_drift(feature_vector, feature_names)
            n_drifted      = len(drift_features)
            n_total        = len(feature_names)
            drift_detected = n_drifted > 0
            anomaly_flag   = (n_drifted / n_total) >= self._anomaly_drift_frac

            if anomaly_flag:
                logger.warning(
                    f"[Monitor] ANOMALY FLAG — {n_drifted}/{n_total} features drifted "
                    f"(threshold={self._anomaly_drift_frac:.0%})"
                )
            elif drift_detected:
                logger.info(
                    f"[Monitor] Drift detected in {n_drifted} feature(s): "
                    + ", ".join(drift_features[:5])
                    + ("..." if n_drifted > 5 else "")
                )

        return {
            "stats":          stats,
            "drift_detected": drift_detected,
            "drift_features": drift_features,
            "anomaly_flag":   anomaly_flag,
            "baseline_ready": self._baseline is not None,
            "n_features":     len(feature_names),
            "checked_at":     datetime.now(timezone.utc).isoformat(),
        }

    def reset_baseline(self) -> None:
        """
        Clear the stored baseline and warm-up buffer.
        Call this when a new bearing is started or model is redeployed.
        """
        self._baseline = None
        self._warmup_buffer.clear()
        if os.path.exists(self._baseline_path):
            os.remove(self._baseline_path)
        logger.info("[Monitor] Baseline reset.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_stats(
        self,
        feature_vector: np.ndarray,
        feature_names:  List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Descriptive statistics for one feature vector (single burst)."""
        stats: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(feature_names):
            val = float(feature_vector[i])
            stats[name] = {
                "value":   round(val, 6),
                "is_nan":  bool(np.isnan(val)),
                "is_inf":  bool(np.isinf(val)),
            }

        # Aggregate stats over the full vector
        valid = feature_vector[np.isfinite(feature_vector)]
        stats["_batch"] = {
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "var":  float(np.var(valid))  if len(valid) else float("nan"),
            "min":  float(np.min(valid))  if len(valid) else float("nan"),
            "max":  float(np.max(valid))  if len(valid) else float("nan"),
            "n_nan": int(np.sum(np.isnan(feature_vector))),
            "n_inf": int(np.sum(np.isinf(feature_vector))),
        }
        return stats

    def _detect_drift(
        self,
        feature_vector: np.ndarray,
        feature_names:  List[str],
    ) -> List[str]:
        """
        Return names of features whose current value deviates from the
        baseline mean by more than drift_z_thresh baseline standard deviations.
        """
        drifted: List[str] = []
        bl_means = self._baseline["means"]
        bl_stds  = self._baseline["stds"]

        for i, name in enumerate(feature_names):
            val = feature_vector[i]
            if np.isnan(val) or np.isinf(val):
                drifted.append(name)
                continue

            bl_std = bl_stds.get(name, 0.0)
            if bl_std < 1e-9:
                continue   # feature has no variance in baseline — skip

            z = abs(val - bl_means.get(name, val)) / bl_std
            if z > self._drift_z_thresh:
                drifted.append(name)

        return drifted

    def _compute_and_save_baseline(self, feature_names: List[str]) -> None:
        """Compute baseline stats from warm-up buffer and persist."""
        arr = np.array(self._warmup_buffer, dtype=np.float32)
        means = {name: float(np.nanmean(arr[:, i])) for i, name in enumerate(feature_names)}
        stds  = {name: float(np.nanstd(arr[:, i]))  for i, name in enumerate(feature_names)}

        self._baseline = {
            "computed_at":    datetime.now(timezone.utc).isoformat(),
            "n_bursts":       len(self._warmup_buffer),
            "feature_names":  feature_names,
            "means":          means,
            "stds":           stds,
        }
        self._save_baseline()
        logger.info(
            f"[Monitor] Baseline computed from {len(self._warmup_buffer)} bursts "
            f"and saved to {self._baseline_path}"
        )

    def _save_baseline(self) -> None:
        os.makedirs(os.path.dirname(self._baseline_path) or ".", exist_ok=True)
        with open(self._baseline_path, "w") as f:
            json.dump(self._baseline, f, indent=2)

    def _load_baseline(self) -> None:
        if os.path.exists(self._baseline_path):
            try:
                with open(self._baseline_path) as f:
                    self._baseline = json.load(f)
                logger.info(
                    f"[Monitor] Baseline loaded from {self._baseline_path} "
                    f"(computed {self._baseline.get('computed_at', 'unknown')})"
                )
            except Exception as exc:
                logger.warning(f"[Monitor] Could not load baseline: {exc}")
                self._baseline = None

    def _empty_result(self, fe_output: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "stats":          {},
            "drift_detected": False,
            "drift_features": [],
            "anomaly_flag":   False,
            "baseline_ready": self._baseline is not None,
            "n_features":     0,
            "checked_at":     datetime.now(timezone.utc).isoformat(),
            "skipped_reason": reason,
        }