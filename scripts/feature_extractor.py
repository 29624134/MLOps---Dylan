import os
import numpy as np
import pandas as pd
from scipy import stats
import logging
from scripts.feature_extractors.time_features import extract_time_domain
from scripts.feature_extractors.wavelet_energy import extract_wavelet_energy
from scripts.feature_extractors.wavelet_entropy import extract_wavelet_entropy

class FeatureExtractor:

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
    # Main processing
    # --------------------------
    def process_npz(self, npz_path, output_dir):
        file = np.load(npz_path)
        data, labels = file["data"], file["labels"]

        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        self.logger.info(f"Extracting features from {data.shape[0]} samples...")

        # ========== TIME DOMAIN ==========
        self.logger.info("Extracting time-domain features...")
        df_time = extract_time_domain(data)
        df_time["fault"] = labels
        os.makedirs(output_dir, exist_ok=True)
        time_csv = os.path.join(output_dir, "time_features.csv")
        df_time.to_csv(time_csv, index=False)
        self.logger.info(f"Saved time-domain features to {time_csv}")

        # ========== WAVELET ENERGY ==========
        self.logger.info("Extracting wavelet energy features...")
        df_wave_energy = extract_wavelet_energy(data)
        df_wave_energy["fault"] = labels
        wave_energy_csv = os.path.join(output_dir, "wavelet_energy.csv")
        df_wave_energy.to_csv(wave_energy_csv, index=False)
        self.logger.info(f"Saved wavelet energy features to {wave_energy_csv}")

        # ========== WAVELET ENTROPY ==========
        self.logger.info("Extracting wavelet entropy features...")
        df_wave_entropy = extract_wavelet_entropy(data)
        df_wave_entropy["fault"] = labels
        wave_entropy_csv = os.path.join(output_dir, "wavelet_entropy.csv")
        df_wave_entropy.to_csv(wave_entropy_csv, index=False)
        self.logger.info(f"Saved wavelet entropy features to {wave_entropy_csv}")

        return {
            "time": df_time,
            "wavelet_energy": df_wave_energy,
            "wavelet_entropy": df_wave_entropy
        }

    def run(self):
        if not self.input_path:
            raise ValueError("No input path provided for feature extraction.")

        # output_path should be a folder, not a single CSV
        os.makedirs(self.output_path, exist_ok=True)

        results = self.process_npz(self.input_path, self.output_path)
        return results



