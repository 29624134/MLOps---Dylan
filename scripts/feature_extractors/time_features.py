import numpy as np
import pandas as pd
from scipy import stats

def extract_time_domain(data: np.ndarray) -> pd.DataFrame:
    """
    Extract standard time-domain statistical features from vibration signals.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, signal_length)

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame of shape (n_samples, n_features) containing:
        max, min, mean, std, rms, skewness, kurtosis, crest_factor, form_factor
    """

    n_samples = data.shape[0]
    features = []

    for i in range(n_samples):
        x = data[i,:]
        f = {}
        f["max"] = np.max(x)
        f["min"] = np.min(x)
        f["mean"] = np.mean(x)
        f["std"] = np.std(x, ddof=1)
        f["rms"] = np.sqrt(np.mean(x ** 2))
        f["skewness"] = stats.skew(x)
        f["kurtosis"] = stats.kurtosis(x)
        f["crest_factor"] = f["max"] / f["rms"] if f["rms"] != 0 else 0
        f["form_factor"] = f["rms"] / (f["mean"] if f["mean"] != 0 else 1)
        features.append(f)

    features_df = pd.DataFrame(features)
    return features_df
