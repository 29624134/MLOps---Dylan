"""
serving_pipeline/feature_engineering.py
═══════════════════════════════════════════════════════════════════════════════
Serving Pipeline — Stage 1: Feature Engineering

Responsibilities
────────────────
1. Accept one raw burst (h_signal, v_signal arrays) from the Feature Store
   / Scheduler.
2. Extract the 18 time-domain base features + rolling window stats → 76-dim
   vector (reuses LiveFeatureBuffer logic).
3. Attach MLOps Project Instruction §4 quality labels to every feature:
       outlier      : bool
       missing      : bool
       anomaly_type : "spike" | "dropout" | "null" | "none"
4. Return a structured dict consumable by the Inference stage.

The quality labels do NOT remove or impute values — they annotate them so
downstream stages (Inference, PM) can handle them gracefully.

Usage
─────
    from serving_pipeline.feature_engineering import ServingFeatureEngineer

    fe = ServingFeatureEngineer(window_size=40)

    result = fe.process_burst(
        burst_idx = 42,
        h_signal  = np.array([...]),
        v_signal  = np.array([...]),
    )
    # result["feature_vector"]   — np.ndarray (76,)
    # result["feature_names"]    — list[str]
    # result["quality_labels"]   — dict with outlier/missing/anomaly_type per feature
    # result["ready"]            — False until window_size bursts seen
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Outlier detection thresholds (z-score based) ──────────────────────────────
_OUTLIER_Z_THRESH   = 4.0   # |z| > 4  → outlier
_SPIKE_Z_THRESH     = 6.0   # |z| > 6  → spike anomaly (extreme outlier)
_DROPOUT_NEAR_ZERO  = 1e-6  # RMS < threshold → likely sensor dropout


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy feature extraction (mirrors LiveFeatureBuffer exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _skewness(x: np.ndarray) -> float:
    N, mu = len(x), np.mean(x)
    s3 = np.std(x, ddof=1) ** 3
    return float((np.sum((x - mu) ** 3) / N) / s3) if s3 != 0 else 0.0


def _kurtosis(x: np.ndarray) -> float:
    N, mu = len(x), np.mean(x)
    s4 = np.std(x, ddof=1) ** 4
    return float((np.sum((x - mu) ** 4) / N) / s4 - 3) if s4 != 0 else 0.0


def _burst_base_features(h: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    """Extract 18 time-domain base features from one burst."""
    out: Dict[str, float] = {}
    for prefix, sig in (("h", h), ("v", v)):
        mx   = float(np.max(sig))
        mn   = float(np.min(sig))
        mean = float(np.mean(sig))
        sd   = float(np.std(sig, ddof=1))
        rms  = float(np.sqrt(np.mean(sig ** 2)))
        skew = _skewness(sig)
        kurt = _kurtosis(sig)
        crest = mx / rms   if rms  != 0 else 0.0
        form  = rms / mean if mean != 0 else 0.0
        out.update({
            f"{prefix}_max": mx,  f"{prefix}_min": mn,
            f"{prefix}_mean": mean, f"{prefix}_sd": sd,
            f"{prefix}_rms": rms, f"{prefix}_skew": skew,
            f"{prefix}_kurt": kurt, f"{prefix}_crest": crest,
            f"{prefix}_form": form,
        })
    return out


_BASE_COLS = [
    "h_max","h_min","h_mean","h_sd","h_rms","h_skew","h_kurt","h_crest","h_form",
    "v_max","v_min","v_mean","v_sd","v_rms","v_skew","v_kurt","v_crest","v_form",
]
_ROLLING_COLS = _BASE_COLS + ["RUL_norm"]   # RUL_norm = 0.0 placeholder at serve time


def _rolling_feature_names() -> List[str]:
    names = list(_ROLLING_COLS)
    for col in _ROLLING_COLS:
        names += [f"{col}_mean", f"{col}_std", f"{col}_slope"]
    return names   # 76 total


# ─────────────────────────────────────────────────────────────────────────────
# Quality labelling
# ─────────────────────────────────────────────────────────────────────────────

def _label_quality(
    feature_vector: np.ndarray,
    feature_names: List[str],
    window_history: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Attach quality labels to the feature vector.

    Returns a dict with:
        outlier       : bool   — at least one feature is an outlier
        missing       : bool   — at least one feature is NaN / inf
        anomaly_type  : str    — "spike" | "dropout" | "null" | "none"
        per_feature   : dict   — per-feature breakdown {name: {outlier, missing, z_score}}
    """
    per_feature: Dict[str, Dict[str, Any]] = {}
    has_missing = False
    has_outlier = False
    anomaly_type = "none"

    # Compute z-scores against window history if available
    if window_history is not None and len(window_history) > 1:
        mu  = np.mean(window_history, axis=0)
        std = np.std(window_history, axis=0) + 1e-9
        z_scores = np.abs((feature_vector - mu) / std)
    else:
        z_scores = np.zeros_like(feature_vector)

    for i, name in enumerate(feature_names):
        val = feature_vector[i]
        z   = float(z_scores[i])

        is_missing = bool(np.isnan(val) or np.isinf(val))
        is_outlier = bool(z > _OUTLIER_Z_THRESH)

        per_feature[name] = {
            "missing": is_missing,
            "outlier": is_outlier,
            "z_score": round(z, 3),
        }

        if is_missing:
            has_missing = True
        if is_outlier:
            has_outlier = True

    # Determine anomaly type (in priority order)
    if has_missing:
        anomaly_type = "null"
    else:
        # Check for dropout: h_rms and v_rms near zero
        h_rms_idx = feature_names.index("h_rms") if "h_rms" in feature_names else -1
        v_rms_idx = feature_names.index("v_rms") if "v_rms" in feature_names else -1
        h_rms = feature_vector[h_rms_idx] if h_rms_idx >= 0 else 1.0
        v_rms = feature_vector[v_rms_idx] if v_rms_idx >= 0 else 1.0

        if h_rms < _DROPOUT_NEAR_ZERO or v_rms < _DROPOUT_NEAR_ZERO:
            anomaly_type = "dropout"
        elif has_outlier:
            # Check if any z-score is spike-level
            max_z = max(f["z_score"] for f in per_feature.values())
            anomaly_type = "spike" if max_z > _SPIKE_Z_THRESH else "none"
            if has_outlier and anomaly_type == "none":
                anomaly_type = "none"   # outlier but not spike-level

    return {
        "outlier":      has_outlier,
        "missing":      has_missing,
        "anomaly_type": anomaly_type,
        "per_feature":  per_feature,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ServingFeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

class ServingFeatureEngineer:
    """
    Stage 1 of the Serving Pipeline.

    Wraps LiveFeatureBuffer-compatible rolling-window logic and adds
    quality labelling per MLOps Project Instructions §4.

    Parameters
    ──────────
    window_size : int
        Number of bursts in the rolling window (must match training, default 40).
    """

    def __init__(self, window_size: int = 40):
        self._window_size = window_size
        self._window: List[np.ndarray] = []         # history of 19-dim base rows (for rolling vector building)
        self._vector_history: List[np.ndarray] = [] # history of 76-dim rolling vectors (for quality labelling)
        self._feature_names = _rolling_feature_names()
        self._burst_count = 0
        logger.info(f"[FE] ServingFeatureEngineer initialised (window={window_size})")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_burst(
        self,
        burst_idx: int,
        h_signal:  np.ndarray,
        v_signal:  np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process one burst and return a labelled feature dict.

        Returns dict with keys:
            ready          : bool   — False until window is full
            burst_idx      : int
            feature_vector : np.ndarray | None
            feature_names  : list[str]
            base_features  : dict   — raw 18-dim features for this burst
            quality_labels : dict   — outlier/missing/anomaly_type + per_feature
        """
        self._burst_count += 1

        # 1. Extract base features for this burst
        base = _burst_base_features(h_signal, v_signal)
        base_row = np.array([base[c] for c in _BASE_COLS] + [0.0], dtype=np.float32)
        # ^^ RUL_norm placeholder = 0.0 (not known at serve time)

        # 2. Append to rolling window
        self._window.append(base_row)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        if len(self._window) < self._window_size:
            logger.debug(
                f"[FE] Burst {burst_idx}: buffer filling "
                f"({len(self._window)}/{self._window_size})"
            )
            return {
                "ready":          False,
                "burst_idx":      burst_idx,
                "feature_vector": None,
                "feature_names":  self._feature_names,
                "base_features":  base,
                "quality_labels": None,
            }

        # 3. Build 76-dim rolling vector
        feature_vector = self._build_rolling_vector()

        # 4. Quality labelling — compare against previous 76-dim vectors
        history_arr = np.array(self._vector_history) if self._vector_history else None
        quality     = _label_quality(feature_vector, self._feature_names, history_arr)

        # 5. Store this vector for future quality comparisons
        self._vector_history.append(feature_vector)
        if len(self._vector_history) > self._window_size:
            self._vector_history.pop(0)

        if quality["anomaly_type"] != "none":
            logger.warning(
                f"[FE] Burst {burst_idx}: anomaly_type='{quality['anomaly_type']}'"
                f"  outlier={quality['outlier']}  missing={quality['missing']}"
            )

        return {
            "ready":          True,
            "burst_idx":      burst_idx,
            "feature_vector": feature_vector,
            "feature_names":  self._feature_names,
            "base_features":  base,
            "quality_labels": quality,
        }

    def reset(self) -> None:
        """Clear the rolling window (call between bearings)."""
        self._window.clear()
        self._vector_history.clear()
        self._burst_count = 0
        logger.info("[FE] Window reset.")

    @property
    def is_ready(self) -> bool:
        return len(self._window) >= self._window_size

    @property
    def bursts_seen(self) -> int:
        return self._burst_count

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_rolling_vector(self) -> np.ndarray:
        """Build the 76-element feature vector from the current window."""
        arr = np.array(self._window, dtype=np.float32)   # (window_size, 19)

        # Raw values from most-recent row (19)
        raw = arr[-1].tolist()

        # Rolling mean / std / slope per column (19 × 3 = 57)
        rolling = []
        for col_idx in range(arr.shape[1]):
            col_vals = arr[:, col_idx]
            mean     = float(np.mean(col_vals))
            std      = float(np.std(col_vals))
            # slope via simple linear regression over index
            x     = np.arange(len(col_vals), dtype=np.float32)
            x_bar = np.mean(x)
            denom = np.sum((x - x_bar) ** 2)
            slope = float(np.sum((x - x_bar) * (col_vals - np.mean(col_vals))) / denom) if denom != 0 else 0.0
            rolling += [mean, std, slope]

        vec = np.array(raw + rolling, dtype=np.float32)
        assert len(vec) == 76, f"Expected 76 features, got {len(vec)}"
        return vec