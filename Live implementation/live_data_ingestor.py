import os
import time
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False


class DataIngestorPHM:
    """Reads PHM 2012 bearing sensor CSV files and consolidates them into parquet format."""

    def __init__(self, config=None, input_location=None, output_location=None,
                 state_location=None, save_format="parquet", log_path=None):
        """
        Accepts either direct arguments (manual run) or a config dict (for orchestrator).
        """
        if isinstance(config, dict):
            self.input_location  = config.get("input_location")
            self.output_location = config.get("output_location")
            self.state_location  = config.get("state_location")
            self.save_format     = config.get("save_format", "parquet")
            self.log_path        = config.get("log_path")
        else:
            self.input_location  = input_location
            self.output_location = output_location
            self.state_location  = state_location
            self.save_format     = save_format
            self.log_path        = log_path

        self.vibration_columns   = ["Hour", "Minute", "Second", "Microsecond",
                                    "Horizontal_Accel", "Vertical_Accel"]
        self.temperature_columns = ["Hour", "Minute", "Second", "Fraction_Second",
                                    "RTD_Sensor"]

        # Setup logger
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

    # ------------------------------------------------------------------
    # Internal helpers  (shared by both batch and streaming paths)
    # ------------------------------------------------------------------

    def _extract_7z_if_needed(self):
        """Extract any .7z archives found under input_location."""
        if not HAS_7Z:
            return
        for archive in Path(self.input_location).glob("*.7z"):
            dest = Path(self.input_location) / archive.stem
            if dest.exists():
                self.logger.info(f"Already extracted: {archive.name}")
                continue
            dest.mkdir(exist_ok=True)
            try:
                with py7zr.SevenZipFile(archive, mode="r") as arc:
                    arc.extractall(path=dest)
                self.logger.info(f"Extracted {archive.name} to {dest}")
            except Exception as e:
                self.logger.warning(f"Could not extract {archive.name}: {e}")

    def _discover_files(self) -> Tuple[List[Path], List[Path]]:
        """Return sorted lists of vibration and temperature CSV files."""
        base = Path(self.input_location)
        vib_files  = sorted(base.rglob("acc_*.csv"))
        temp_files = sorted(base.rglob("temp_*.csv"))
        self.logger.info(f"Found {len(vib_files)} vibration files, "
                         f"{len(temp_files)} temperature files in {base}")
        return vib_files, temp_files

    def _read_signal_file(self, file_path: Path, file_type: str) -> pd.DataFrame:
        """Read a single sensor CSV with automatic delimiter detection."""
        try:
            columns = (self.vibration_columns if file_type == "vibration"
                       else self.temperature_columns)
            df = pd.read_csv(file_path, header=None, names=columns,
                             sep=None, engine="python")

            numeric_cols = ["Hour", "Minute", "Second"]
            numeric_cols += (["Microsecond", "Horizontal_Accel", "Vertical_Accel"]
                             if file_type == "vibration" else
                             ["Fraction_Second", "RTD_Sensor"])

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=numeric_cols, inplace=True)

            df["file_id"]      = file_path.stem
            df["source_file"]  = file_path.name
            df["Time_Seconds"] = df["Hour"] * 3600 + df["Minute"] * 60 + df["Second"]

            if file_type == "vibration":
                df["Time_Seconds"] += df["Microsecond"] / 1e6
            else:
                df["Time_Seconds"] += df["Fraction_Second"]

            df["sample_index"] = range(len(df))
            return df

        except Exception as e:
            self.logger.warning(f"Skipped {file_path.name}: {e}")
            return pd.DataFrame()

    def _save_dataframe(self, df: pd.DataFrame, stem: str) -> str:
        """Save a dataframe in the configured format and return the file path."""
        ext_map = {"parquet": ".parquet", "csv": ".csv", "hdf5": ".h5"}
        if self.save_format not in ext_map:
            raise ValueError(f"Unknown save_format: {self.save_format!r}")
        os.makedirs(self.output_location, exist_ok=True)
        path = os.path.join(self.output_location, f"{stem}{ext_map[self.save_format]}")
        if self.save_format == "parquet":
            df.to_parquet(path, index=False, compression="snappy")
        elif self.save_format == "csv":
            df.to_csv(path, index=False)
        elif self.save_format == "hdf5":
            df.to_hdf(path, key="data", mode="w", complevel=9)
        self.logger.info(f"Saved {len(df):,} rows to {path}")
        return path

    # ------------------------------------------------------------------
    # BATCH ingestion  (training / offline mode)
    # ------------------------------------------------------------------

    def ingest(self) -> dict:
        """Main ingestion method: reads all sensor CSVs at once and saves parquet files."""
        if not self.input_location:
            raise ValueError("DataIngestorPHM requires 'input_location'.")
        if not self.output_location:
            raise ValueError("DataIngestorPHM requires 'output_location'.")

        self._extract_7z_if_needed()
        vib_files, temp_files = self._discover_files()
        outputs = {}

        # ── Vibration ─────────────────────────────────────────────────────────
        if vib_files:
            frames = []
            for idx, f in enumerate(vib_files, 1):
                if idx % 100 == 0 or idx == len(vib_files):
                    self.logger.info(f"Vibration progress: {idx}/{len(vib_files)}")
                df = self._read_signal_file(f, "vibration")
                if not df.empty:
                    frames.append(df)

            if frames:
                vib_df = pd.concat(frames, ignore_index=True)
                path = self._save_dataframe(vib_df, "vibration_consolidated")
                outputs["vibration_parquet"] = path
                self.logger.info(
                    f"Vibration summary: {len(vib_df):,} rows | "
                    f"{vib_df['file_id'].nunique()} files | "
                    f"H_accel mean={vib_df['Horizontal_Accel'].mean():.4f}"
                )

        # ── Temperature ───────────────────────────────────────────────────────
        if temp_files:
            frames = []
            for idx, f in enumerate(temp_files, 1):
                if idx % 100 == 0 or idx == len(temp_files):
                    self.logger.info(f"Temperature progress: {idx}/{len(temp_files)}")
                df = self._read_signal_file(f, "temperature")
                if not df.empty:
                    frames.append(df)

            if frames:
                temp_df = pd.concat(frames, ignore_index=True)
                path = self._save_dataframe(temp_df, "temperature_consolidated")
                outputs["temperature_parquet"] = path
                self.logger.info(
                    f"Temperature summary: {len(temp_df):,} rows | "
                    f"{temp_df['file_id'].nunique()} files | "
                    f"RTD mean={temp_df['RTD_Sensor'].mean():.4f}"
                )

        if not outputs:
            self.logger.warning("No sensor files were found or could be read.")

        if self.state_location:
            os.makedirs(os.path.dirname(self.state_location), exist_ok=True)
            with open(self.state_location, "w") as f:
                f.write("complete")
            self.logger.info(f"Data ingestion state saved at {self.state_location}")

        return outputs

    # ------------------------------------------------------------------
    # STREAMING ingestion  (live inference mode)
    # ------------------------------------------------------------------

    def stream_bursts(
        self,
        source_folder: str,
        burst_period: float = 10.0,
        realtime: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Simulate live sensor ingestion by yielding one burst at a time.

        In real deployment, new acc_*.csv files arrive every `burst_period`
        seconds. This generator replicates that by reading files in
        chronological order, pausing `burst_period` seconds between each
        (when realtime=True), or yielding immediately with no delay
        (when realtime=False, useful for fast replay in tests).

        Yields
        ------
        dict with keys:
            file_id     : str   — e.g. "acc_00001"
            burst_idx   : int   — 0-based chronological index
            time_s      : float — seconds since recording started
            h_signal    : np.ndarray — horizontal acceleration samples
            v_signal    : np.ndarray — vertical acceleration samples
            h_max       : float — peak horizontal acceleration (g)
            v_max       : float — peak vertical acceleration (g)

        Parameters
        ----------
        source_folder : str
            Path to the bearing folder containing acc_*.csv files.
        burst_period : float
            Simulated time between bursts in seconds (default 10.0).
        realtime : bool
            If True, sleep burst_period seconds between yields to
            simulate live data rate. If False, yield immediately
            (fast replay for testing).
        """
        base = Path(source_folder)
        vib_files = sorted(base.rglob("acc_*.csv"))

        if not vib_files:
            self.logger.warning(f"No acc_*.csv files found in {source_folder}")
            return

        self.logger.info(
            f"Streaming {len(vib_files)} bursts from {source_folder} "
            f"({'realtime' if realtime else 'fast replay'}, "
            f"burst_period={burst_period}s)"
        )

        for burst_idx, file_path in enumerate(vib_files):
            df = self._read_signal_file(file_path, "vibration")
            if df.empty:
                self.logger.warning(f"Skipping empty burst: {file_path.name}")
                continue

            h = df["Horizontal_Accel"].values
            v = df["Vertical_Accel"].values

            yield {
                "file_id":   file_path.stem,
                "burst_idx": burst_idx,
                "time_s":    burst_idx * burst_period,
                "h_signal":  h,
                "v_signal":  v,
                "h_max":     float(np.max(np.abs(h))),
                "v_max":     float(np.max(np.abs(v))),
            }

            if realtime:
                time.sleep(burst_period)

    # ------------------------------------------------------------------
    # Orchestrator entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Wrapper for orchestrator compatibility.
        Returns a dictionary with output paths.
        """
        outputs = self.ingest()
        return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Standalone / batch entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_PARENT_PATH = r"C:\Users\dylan\OneDrive - Stellenbosch University\MADRG - Opperman, Dylan\Data\ieee-phm-2012-data-challenge-dataset-master\Full_Test_Set"

    bearings_to_process = [
        "Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
        "Bearing3_1", "Bearing3_2",
        "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
        "Bearing3_3",
    ]

    for bearing_name in bearings_to_process:
        print(f"\n{'='*60}\nProcessing {bearing_name}\n{'='*60}")
        config = {
            "input_location":  str(Path(BASE_PARENT_PATH) / bearing_name),
            "output_location": str(Path(BASE_PARENT_PATH) / bearing_name / "consolidated"),
            "save_format":     "parquet",
        }
        ingestor = DataIngestorPHM(config)
        result = ingestor.run()
        print(f"Outputs: {result}")