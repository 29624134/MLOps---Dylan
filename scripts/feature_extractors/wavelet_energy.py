import numpy as np
import pandas as pd
import pywt
from scipy import stats

def extract_wavelet_energy(data:np.ndarray, wavelet="sym8", maxlevel = 3) -> pd.DataFrame:
    """
    Extract wavelet packet energy features from a 2D array (samples x signal_length).

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (n_samples, signal_length)
    wavelet : str
        Wavelet type
    maxlevel : int
        Maximum decomposition level

    Returns
    -------
    features : np.ndarray
        2D array of shape (n_samples, n_features) containing wavelet packet energies
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
            features[i, j] = np.sum(reconstructed ** 2)   # squared L2 norm = sum of squares

    cols = [f"wp_energy_{j}" for j in range(n_features)]
    return pd.DataFrame(features, columns = cols)