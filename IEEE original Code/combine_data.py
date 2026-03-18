import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Optional: for 7z support
try:
    import py7zr
    HAS_7Z = True
except ImportError:
    HAS_7Z = False
    print("Warning: py7zr not installed. 7z extraction disabled.")


class SensorDataConsolidator:
    """
    Consolidates all sensor data into single files for easy analysis
    """

    def __init__(self, base_path: str, output_dir: str = "./consolidated_data"):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.vibration_columns = [
            'Hour', 'Minute', 'Second', 'Microsecond',
            'Horizontal_Accel', 'Vertical_Accel'
        ]

        self.temperature_columns = [
            'Hour', 'Minute', 'Second', 'Fraction_Second',
            'RTD_Sensor'
        ]

        print(f"Initialized with base path: {self.base_path.absolute()}")
        print(f"Consolidated data will be saved to: {self.output_dir.absolute()}")

    def discover_files(self) -> Tuple[List[Path], List[Path]]:
        print("\nDiscovering files...")
        print(f"Searching in: {self.base_path.absolute()}")

        vibration_files = sorted(list(self.base_path.rglob("acc_*.csv")))
        temperature_files = sorted(list(self.base_path.rglob("temp_*.csv")))

        print(f"\n✓ Found {len(vibration_files)} vibration files (acc_*.csv)")
        print(f"✓ Found {len(temperature_files)} temperature files (temp_*.csv)")

        if vibration_files:
            print(f"\nSample vibration files:")
            for f in vibration_files[:3]:
                print(f"  - {f.name}")

        if temperature_files:
            print(f"\nSample temperature files:")
            for f in temperature_files[:3]:
                print(f"  - {f.name}")

        return vibration_files, temperature_files

    def extract_7z_if_needed(self) -> List[Path]:
        if not HAS_7Z:
            return []

        archives = list(self.base_path.glob("*.7z"))
        if not archives:
            print("\nNo 7z archives found")
            return []

        print(f"\nFound {len(archives)} 7z archives")
        extracted = []

        for archive_file in archives:
            extract_folder = self.base_path / archive_file.stem
            if extract_folder.exists():
                print(f"  ⊙ Already extracted: {archive_file.name}")
                extracted.append(extract_folder)
                continue

            print(f"  Extracting: {archive_file.name}")
            extract_folder.mkdir(exist_ok=True)
            try:
                with py7zr.SevenZipFile(archive_file, mode='r') as archive:
                    archive.extractall(path=extract_folder)
                extracted.append(extract_folder)
                print(f"    ✓ Extracted to: {extract_folder}")
            except Exception as e:
                print(f"    ✗ Error: {e}")

        return extracted

    def read_signal_file(self, file_path: Path, file_type: str) -> pd.DataFrame:
        """
        Read a single signal file with automatic delimiter detection
        """
        try:
            columns = self.vibration_columns if file_type == 'vibration' else self.temperature_columns

            # Automatic delimiter detection for commas, tabs, or spaces
            df = pd.read_csv(file_path, header=None, names=columns, sep=None, engine='python')

            # Convert numeric columns to float/int
            numeric_cols = ['Hour', 'Minute', 'Second']
            if file_type == 'vibration':
                numeric_cols += ['Microsecond', 'Horizontal_Accel', 'Vertical_Accel']
            else:
                numeric_cols += ['Fraction_Second', 'RTD_Sensor']

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=numeric_cols, inplace=True)

            # Add metadata
            df['file_id'] = file_path.stem
            df['source_file'] = file_path.name

            # Time in seconds
            df['Time_Seconds'] = (
                df['Hour'] * 3600 +
                df['Minute'] * 60 +
                df['Second']
            )

            if file_type == 'vibration':
                df['Time_Seconds'] += df['Microsecond'] / 1e6
            else:
                df['Time_Seconds'] += df['Fraction_Second']

            df['sample_index'] = range(len(df))

            return df

        except Exception as e:
            print(f"  ✗ Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def consolidate_all_data(self, save_format: str = 'parquet'):
        self.extract_7z_if_needed()
        vib_files, temp_files = self.discover_files()

        if not vib_files and not temp_files:
            print("\n⚠ No files found! Please check your base_path.")
            return

        if vib_files:
            print(f"\n{'=' * 60}")
            print(f"Consolidating {len(vib_files)} vibration files...")
            print(f"{'=' * 60}")
            all_vibration_data = []

            for idx, vib_file in enumerate(vib_files, 1):
                if idx % 100 == 0 or idx == len(vib_files):
                    print(f"  Progress: {idx}/{len(vib_files)} files...")

                df = self.read_signal_file(vib_file, 'vibration')
                if not df.empty:
                    all_vibration_data.append(df)

            print("  Combining all vibration data...")
            vibration_df = pd.concat(all_vibration_data, ignore_index=True)
            self._save_dataframe(vibration_df, 'vibration_consolidated', save_format)

            print(f"\n  ✓ Vibration Data Summary:")
            print(f"    Total records: {len(vibration_df):,}")
            print(f"    Total files: {vibration_df['file_id'].nunique()}")
            print(
                f"    Date range: {vibration_df['Time_Seconds'].min():.2f}s to {vibration_df['Time_Seconds'].max():.2f}s")
            print(
                f"    Horizontal Accel - Mean: {vibration_df['Horizontal_Accel'].mean():.6f}, Std: {vibration_df['Horizontal_Accel'].std():.6f}")
            print(
                f"    Vertical Accel   - Mean: {vibration_df['Vertical_Accel'].mean():.6f}, Std: {vibration_df['Vertical_Accel'].std():.6f}")

        if temp_files:
            print(f"\n{'=' * 60}")
            print(f"Consolidating {len(temp_files)} temperature files...")
            print(f"{'=' * 60}")
            all_temperature_data = []

            for idx, temp_file in enumerate(temp_files, 1):
                if idx % 100 == 0 or idx == len(temp_files):
                    print(f"  Progress: {idx}/{len(temp_files)} files...")

                df = self.read_signal_file(temp_file, 'temperature')
                if not df.empty:
                    all_temperature_data.append(df)

            if all_temperature_data:
                print("  Combining all temperature data...")
                temperature_df = pd.concat(all_temperature_data, ignore_index=True)
                self._save_dataframe(temperature_df, 'temperature_consolidated', save_format)

                print(f"\n  ✓ Temperature Data Summary:")
                print(f"    Total records: {len(temperature_df):,}")
                print(f"    Total files: {temperature_df['file_id'].nunique()}")
                print(
                    f"    Date range: {temperature_df['Time_Seconds'].min():.2f}s to {temperature_df['Time_Seconds'].max():.2f}s")
                print(
                    f"    RTD Sensor - Mean: {temperature_df['RTD_Sensor'].mean():.6f}, Std: {temperature_df['RTD_Sensor'].std():.6f}")
            else:
                print("⚠ No temperature data could be read! Check delimiter and file formatting.")

        print(f"\n{'=' * 60}")
        print(f"✓ CONSOLIDATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Files saved to: {self.output_dir.absolute()}")

        self._create_usage_guide(save_format)

    def _save_dataframe(self, df: pd.DataFrame, filename: str, save_format: str):
        if save_format == 'csv':
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif save_format == 'parquet':
            filepath = self.output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False, compression='snappy')
        elif save_format == 'hdf5':
            filepath = self.output_dir / f"{filename}.h5"
            df.to_hdf(filepath, key='data', mode='w', complevel=9)
        else:
            raise ValueError(f"Unknown format: {save_format}")

    def _create_usage_guide(self, save_format: str):
        guide_path = self.output_dir / "how_to_use_consolidated_data.py"
        if save_format == 'csv':
            load_code = """
vibration_df = pd.read_csv('vibration_consolidated.csv')
temperature_df = pd.read_csv('temperature_consolidated.csv')
"""
        elif save_format == 'parquet':
            load_code = """
vibration_df = pd.read_parquet('vibration_consolidated.parquet')
temperature_df = pd.read_parquet('temperature_consolidated.parquet')
"""
        else:
            load_code = """
vibration_df = pd.read_hdf('vibration_consolidated.h5', key='data')
temperature_df = pd.read_hdf('temperature_consolidated.h5', key='data')
"""

        guide_content = f"""\"\"\"Quick Guide: How to Use Consolidated Sensor Data\"\"\"

import pandas as pd
import matplotlib.pyplot as plt

{load_code}
"""
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        print(f"\n📖 Usage guide created: {guide_path.name}")


if __name__ == "__main__":
    BASE_PARENT_PATH = r"C:\Users\29624134\Downloads\Original Data\Full_Test_Set"

    bearings_to_process = [
        #"Bearing1_1",
        #"Bearing1_2",
        "Bearing1_3",
        "Bearing1_4",
        "Bearing1_5",
        "Bearing1_6",
        "Bearing1_7",
        #"Bearing2_1",
        #"Bearing2_2",
        "Bearing2_3",
        "Bearing2_4",
        "Bearing2_5",
        "Bearing2_6",
        "Bearing2_7",
        #"Bearing3_1",
        #"Bearing3_2",
        "Bearing3_3",
    ]

    for bearing_name in bearings_to_process:
        base_path = Path(BASE_PARENT_PATH) / bearing_name
        output_dir = base_path / "Big File"

        print(f"\n{'='*80}")
        print(f"Processing {bearing_name}")
        print(f"{'='*80}\n")

        consolidator = SensorDataConsolidator(base_path, output_dir)
        consolidator.consolidate_all_data(save_format='parquet')