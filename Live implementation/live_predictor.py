import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from scripts.feature_extractors.time_features import extract_time_domain
from scripts.feature_extractors.wavelet_energy import extract_wavelet_energy
from scripts.feature_extractors.wavelet_entropy import extract_wavelet_entropy

logger = logging.getLogger(__name__)


class LiveFeatureBuffer:
    """
    Maintains a rolling window of burst-level features for live inference.

    The training pipeline processes an entire recording at once, so rolling
    statistics (mean, std, slope over the last 40 bursts) are computed in
    one vectorised pass. In live mode, bursts arrive one at a time every
    10 seconds, so this buffer replicates that behaviour by:

        1. Receiving one burst's raw signal via push_burst().
        2. Extracting that burst's base features (time-domain + wavelet).
        3. Appending the feature row to an internal deque of length window_size.
        4. Once the deque is full, computing rolling mean / std / slope over
           it to produce the exact same feature vector the model was trained on.

    The buffer returns None from get_feature_row() until window_size bursts
    have been seen — identical to how dropna() works during training.

    Usage
    -----
        buffer = LiveFeatureBuffer(window_size=40)
        for burst in ingestor.stream_bursts(source_folder):
            row = buffer.push_burst(burst["h_signal"], burst["v_signal"])
            if row is not None:
                rul = predictor.predict(row)
    """

    def __init__(
        self,
        window_size: int = 40,
        wavelet: str = "sym8",
        wavelet_maxlevel: int = 3,
    ):
        self.window_size     = window_size
        self.wavelet         = wavelet
        self.wavelet_maxlevel = wavelet_maxlevel

        # Deque stores one dict of base feature values per burst
        self._deque: deque = deque(maxlen=window_size)
        self._burst_count: int = 0
        self._base_feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push_burst(
        self,
        h_signal: np.ndarray,
        v_signal: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Accept one burst's raw signals, extract base features, update the
        rolling window, and return the full feature vector ready for the model.

        Parameters
        ----------
        h_signal : np.ndarray  — horizontal acceleration samples for this burst
        v_signal : np.ndarray  — vertical acceleration samples for this burst

        Returns
        -------
        np.ndarray of shape (n_features,) if the buffer has window_size bursts,
        otherwise None (not enough history yet for rolling stats).
        """
        base_row = self._extract_base_features(h_signal, v_signal)
        self._deque.append(base_row)
        self._burst_count += 1

        if len(self._deque) < self.window_size:
            remaining = self.window_size - len(self._deque)
            logger.debug(f"Buffer filling: {len(self._deque)}/{self.window_size} "
                         f"({remaining} more bursts needed)")
            return None

        return self._compute_rolling_features()

    @property
    def bursts_seen(self) -> int:
        """Total number of bursts pushed since creation."""
        return self._burst_count

    @property
    def is_ready(self) -> bool:
        """True once the buffer has accumulated window_size bursts."""
        return len(self._deque) >= self.window_size

    # ------------------------------------------------------------------
    # Base feature extraction  (one burst at a time)
    # ------------------------------------------------------------------

    def _extract_base_features(
        self,
        h_signal: np.ndarray,
        v_signal: np.ndarray,
    ) -> dict:
        """
        Extract per-burst statistical features from one burst's signals.
        Calls the same three shared feature extractors used during training,
        passing a (1, signal_length) array so the output is one row.
        """
        # Both extractors expect (n_samples, signal_length) — wrap as 1-row arrays
        H = h_signal.reshape(1, -1)
        V = v_signal.reshape(1, -1)

        # Time-domain
        df_h_time = extract_time_domain(H).add_prefix("h_")
        df_v_time = extract_time_domain(V).add_prefix("v_")

        # Wavelet energy
        df_h_energy = extract_wavelet_energy(
            H, wavelet=self.wavelet, maxlevel=self.wavelet_maxlevel
        ).add_prefix("h_")
        df_v_energy = extract_wavelet_energy(
            V, wavelet=self.wavelet, maxlevel=self.wavelet_maxlevel
        ).add_prefix("v_")

        # Wavelet entropy
        df_h_entropy = extract_wavelet_entropy(
            H, wavelet=self.wavelet, maxlevel=self.wavelet_maxlevel
        ).add_prefix("h_")
        df_v_entropy = extract_wavelet_entropy(
            V, wavelet=self.wavelet, maxlevel=self.wavelet_maxlevel
        ).add_prefix("v_")

        row = pd.concat(
            [df_h_time, df_v_time,
             df_h_energy, df_v_energy,
             df_h_entropy, df_v_entropy],
            axis=1
        ).iloc[0].to_dict()

        # Cache feature names on first call
        if self._base_feature_names is None:
            self._base_feature_names = list(row.keys())

        return row

    # ------------------------------------------------------------------
    # Rolling feature computation  (over the full deque)
    # ------------------------------------------------------------------

    def _compute_rolling_features(self) -> np.ndarray:
        """
        Compute rolling mean, std, and slope over the deque of base feature
        rows — replicating exactly what _add_rolling_features() does in
        RULTrainerPHM during training.

        Returns a 1D numpy array of shape (n_base_features * 3,) ready to
        be scaled and fed to the model.
        """
        # Build (window_size, n_base_features) matrix from the deque
        window_df = pd.DataFrame(list(self._deque))

        result = {}
        for col in self._base_feature_names:
            vals = window_df[col].values.astype(float)
            result[f"{col}_mean"]  = float(np.mean(vals))
            result[f"{col}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            result[f"{col}_slope"] = float(
                np.polyfit(np.arange(len(vals)), vals, 1)[0]
            )

        return np.array(list(result.values()), dtype=np.float32)

    def get_feature_names(self) -> Optional[list]:
        """
        Return the list of feature names in the order they appear in the
        vector returned by push_burst() / _compute_rolling_features().
        Returns None until the first burst has been pushed.
        """
        if self._base_feature_names is None:
            return None
        names = []
        for col in self._base_feature_names:
            names += [f"{col}_mean", f"{col}_std", f"{col}_slope"]
        return names