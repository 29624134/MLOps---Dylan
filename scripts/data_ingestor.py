import os
import glob
import numpy as np
from scipy.io import loadmat
import logging

class DataIngestor:
    """Reads CWRU .mat vibration data and segments it into .npz format."""

    def __init__(self, input_folder, output_file, state_location=None,
                 segment_length=1024, segments_per_file=115, sample_rate=12000,
                 log_path=None):
        self.input_folder = input_folder
        self.output_file = output_file
        self.state_location = state_location
        self.segment_length = segment_length
        self.segments_per_file = segments_per_file
        self.sample_rate = sample_rate
        self.log_path = log_path

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            fh = logging.FileHandler(self.log_path)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        else:
            logging.basicConfig(level=logging.INFO)

    def _generate_label(self, filename: str) -> str:
        """Generate fault label based on CWRU naming convention."""
        if filename.startswith("Time_Normal"):
            return "Normal"
        elif filename.startswith("B"):
            return f"Ball_{filename[1:4]}"
        elif filename.startswith("IR"):
            return f"IR_{filename[2:5]}"
        elif filename.startswith("OR"):
            return f"OR_{filename[2:5]}"
        return "Unknown"

    def ingest(self):
        """Main ingestion method: reads files, segments data, and saves .npz."""
        files = np.sort(glob.glob(os.path.join(self.input_folder, "*")))
        self.logger.info(f"Found {len(files)} files in {self.input_folder}")

        total_segments = len(files) * self.segments_per_file
        segmented_data = np.empty((total_segments, self.segment_length), dtype=np.float64)
        labels = []

        segment_index = 0
        for file_path in files:
            filename = os.path.basename(file_path).split(".")[0]
            label = self._generate_label(filename)
            labels.extend([label] * self.segments_per_file)

            # Load MATLAB file
            mat_data = loadmat(file_path)
            # Determine the key dynamically for DE signal
            key_candidates = [k for k in mat_data.keys() if k.startswith("X") and "_DE_time" in k]
            if not key_candidates:
                self.logger.warning(f"No valid key found in {file_path}, skipping")
                continue
            key = key_candidates[0]
            drive_end_data = np.array(mat_data[key], dtype=np.float64).flatten()

            # Segment the data
            for i in range(self.segments_per_file):
                start = i * self.segment_length
                end = start + self.segment_length
                segment = drive_end_data[start:end]

                if len(segment) == self.segment_length and not np.isnan(segment).any():
                    segmented_data[segment_index, :] = segment
                    segment_index += 1
                else:
                    self.logger.warning(f"Skipped incomplete segment {i} in {filename}")

        # Trim unused rows
        segmented_data = segmented_data[:segment_index, :]
        labels = np.array(labels[:segment_index])

        # Ensure output folder exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        np.savez(self.output_file, data=segmented_data, labels=labels)
        self.logger.info(f"Saved {segment_index} segments to {self.output_file}")

        # Mark state flag if provided
        if self.state_location:
            os.makedirs(os.path.dirname(self.state_location), exist_ok=True)
            with open(self.state_location, "w") as f:
                f.write("complete")
            self.logger.info(f"Data ingestion state saved at {self.state_location}")

        return self.output_file
