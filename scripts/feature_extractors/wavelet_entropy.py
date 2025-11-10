import pandas as pd
import numpy as np
import pywt
from pandas import DataFrame


def compute_shannon_entropy(signal: np.ndarray) -> float:
    """Compute non-normalized Shannon entropy for a 1D signal."""

    power = signal**2
    power = power[power>0]
    return -np.sum(power* np.log(power))

def extract_wavelet_entropy(data: np.ndarray, wavelet="sym8", maxlevel=3) -> DataFrame:
    """
    Extract wavelet packet Shannon entropy features.

    Parameters
    ----------
    data : np.ndarray
        Input data (samples x signal_length)
    wavelet : str
        Wavelet type
    maxlevel : int
        Decomposition level

    Returns
    -------
    features : np.ndarray
        Shape (n_samples, n_features)
    """
    n_samples = data.shape[0]
    n_features = 2 ** maxlevel
    features = np.zeros((n_samples, n_features))

    wp0 = pywt.WaveletPacket(data[0, :], wavelet=wavelet, maxlevel=maxlevel)
    packet_names = [node.path for node in wp0.get_level(maxlevel, "natural")]

    for i in range(n_samples):
        wp = pywt.WaveletPacket(data[i, :], wavelet=wavelet, maxlevel=maxlevel)
        for j, name in enumerate(packet_names):
            reconstructed = wp[name].reconstruct(update=False)
            features[i, j] = compute_shannon_entropy(reconstructed)

    cols = [f"wp_entropy_{j}" for j in range(n_features)]
    return pd.DataFrame(features, columns=cols)


