import os
import numpy as np
import pandas as pd
from scipy import stats
import logging

class FeatureExtractor:
    """Extracts only time-domain features from vibration data."""

    TIME_FEATURES = [
        "max", "min", "mean", "std", "rms",
        "skewness", "kurtosis", "crest_factor", "form_factor"
    ]

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 12000)
        self.version = config.get("version", "1.0")
        log_path = config.get("log_path")

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        else:
            logging.basicConfig(level=logging.INFO)

        # Store paths
        self.input_path = config.get("input_location")  # e.g., npz file
        self.output_path = config.get("output_location")  # e.g., CSV or feature store
        self.dataset_id = config.get("dataset_id", "default_dataset")
        self.feature_store_config = config.get("feature_store")  # optional

    # --------------------------
    # Time-domain feature extraction
    # --------------------------
    def _extract_time_domain(self, x):
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
        return f

    # --------------------------
    # Main processing
    # --------------------------
    def process_npz(self, npz_path, output_csv):
        """Extract time-domain features from NPZ and save CSV."""
        file = np.load(npz_path)
        data, labels = file["data"], file["labels"]

        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        self.logger.info(f"Extracting time-domain features from {data.shape[0]} samples...")

        features = []
        for sample in data:
            f = self._extract_time_domain(sample)
            features.append(f)

        df = pd.DataFrame(features)
        df["fault"] = labels

        # Ensure folder exists
        output_csv = os.path.abspath(output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

        self.logger.info(f"Saved time-domain features to {output_csv}")
        return df

    def run(self):
        if not self.input_path:
            raise ValueError("No input path provided for feature extraction.")

        # Extract features
        df = self.process_npz(self.input_path, self.output_path)



