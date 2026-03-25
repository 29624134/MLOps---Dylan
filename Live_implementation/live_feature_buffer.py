import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pure-numpy burst feature functions — identical to feature_extractor.py
# ──────────────────────────────────────────────────────────────────────────────

def _compute_skewness(x: np.ndarray) -> float:
    N, mu = len(x), np.mean(x)
    s3 = np.std(x, ddof=1) ** 3
    return (np.sum((x - mu) ** 3) / N) / s3 if s3 != 0 else 0.0


def _compute_kurtosis(x: np.ndarray) -> float:
    N, mu = len(x), np.mean(x)
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


# ──────────────────────────────────────────────────────────────────────────────
# The 18 time-domain base feature names (matches feature_extractor.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
_BASE_FEATURES = [
    "h_max", "h_min", "h_mean", "h_sd", "h_rms",
    "h_skew", "h_kurt", "h_crest", "h_form",
    "v_max", "v_min", "v_mean", "v_sd", "v_rms",
    "v_skew", "v_kurt", "v_crest", "v_form",
]

# Rolling is computed over these 19 columns (base features + RUL_norm).
# RUL_norm is not available at live inference time so we set it to 0.0.
# The model saw RUL_norm during training but at live time we don't know the
# true RUL, so 0.0 is a neutral placeholder that keeps the feature present.
_ROLLING_COLS = _BASE_FEATURES + ["RUL_norm"]


class LiveFeatureBuffer:
    """
    Produces the exact 76-feature vector the deployed model expects:

        19 raw base values  (h_max … v_form, RUL_norm)
      + 19 × 3 rolling stats (mean / std / slope over window_size bursts)
      = 76 features

    RUL_norm is set to 0.0 at live inference time (true RUL is unknown).
    This matches the column order the StandardScaler was fitted on.

    Usage
    -----
        buffer = LiveFeatureBuffer(window_size=40)
        for burst in ingestor.stream_bursts(source_folder):
            vec = buffer.push_burst(burst["h_signal"], burst["v_signal"])
            if vec is not None:
                rul_s = predictor.predict(vec)
    """

    def __init__(self, window_size: int = 40):
        self.window_size  = window_size
        self._deque: deque = deque(maxlen=window_size)
        self._burst_count  = 0

    def push_burst(
        self,
        h_signal: np.ndarray,
        v_signal: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Accept one burst, extract base features, update rolling window.
        Returns the 76-element feature vector once window_size bursts have
        been seen, otherwise returns None.
        """
        row = self._extract_row(h_signal, v_signal)
        self._deque.append(row)
        self._burst_count += 1

        if len(self._deque) < self.window_size:
            logger.debug(
                f"Buffer filling: {len(self._deque)}/{self.window_size}"
            )
            return None

        return self._build_feature_vector()

    @property
    def bursts_seen(self) -> int:
        return self._burst_count

    @property
    def is_ready(self) -> bool:
        return len(self._deque) >= self.window_size

    def get_feature_names(self) -> list:
        """Return the 76 feature names in model-expected order."""
        names = list(_ROLLING_COLS)                          # 19 raw values
        for col in _ROLLING_COLS:                            # 19 × 3 rolling
            names += [f"{col}_mean", f"{col}_std", f"{col}_slope"]
        return names

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_row(self, h_signal: np.ndarray, v_signal: np.ndarray) -> dict:
        """Extract 18 time-domain base features + RUL_norm placeholder."""
        hf = _burst_features(h_signal)
        vf = _burst_features(v_signal)
        return {
            "h_max":   hf[0], "h_min":   hf[1], "h_mean":  hf[2],
            "h_sd":    hf[3], "h_rms":   hf[4], "h_skew":  hf[5],
            "h_kurt":  hf[6], "h_crest": hf[7], "h_form":  hf[8],
            "v_max":   vf[0], "v_min":   vf[1], "v_mean":  vf[2],
            "v_sd":    vf[3], "v_rms":   vf[4], "v_skew":  vf[5],
            "v_kurt":  vf[6], "v_crest": vf[7], "v_form":  vf[8],
            "RUL_norm": 0.0,   # unknown at live inference time
        }

    def _build_feature_vector(self) -> np.ndarray:
        """
        Build the 76-element vector in the exact order the scaler expects:
          [19 raw base values] + [19 × 3 rolling stats]
        """
        window_df = pd.DataFrame(list(self._deque))
        latest    = window_df.iloc[-1]   # most recent burst's raw values

        result = {}

        # 19 raw base values (from the most recent burst)
        for col in _ROLLING_COLS:
            result[col] = float(latest[col])

        # 19 × 3 rolling stats over the full window
        for col in _ROLLING_COLS:
            vals = window_df[col].values.astype(float)
            result[f"{col}_mean"]  = float(np.mean(vals))
            result[f"{col}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            result[f"{col}_slope"] = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])

        return np.array(list(result.values()), dtype=np.float32)