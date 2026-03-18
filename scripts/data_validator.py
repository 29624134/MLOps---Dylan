import os
import json
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataSchema:
    name: str
    dtype: str
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unique_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None


class DataValidatorPHM:
    """Validator for PHM 2012 consolidated parquet files and feature CSVs."""

    def __init__(self, schema_path: Optional[str] = None, log_path: Optional[str] = None):
        # Logger first
        self.logger = logging.getLogger("DataValidatorPHM")
        self.logger.setLevel(logging.INFO)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
        else:
            logging.basicConfig(level=logging.INFO)

        # Schema
        self.schema: Dict[str, DataSchema] = {}
        if schema_path:
            self.load_schema(schema_path)

    def load_schema(self, schema_path: str):
        """Load schema JSON for validation."""
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema not found at {schema_path}")
        with open(schema_path, "r") as f:
            schema_json = json.load(f)
        # data_schema.json has two top-level sections (raw_consolidated, features).
        # Each section has a nested "features" dict. We load the feature-level
        # section so validate_features() has the right column definitions.
        section = schema_json.get("features", schema_json)
        feature_defs = section.get("features", section)
        for feature, rules in feature_defs.items():
            if not isinstance(rules, dict):
                continue
            self.schema[feature] = DataSchema(
                name=feature,
                dtype=rules.get("dtype", "float64"),
                nullable=rules.get("nullable", False),
                min_value=rules.get("min"),
                max_value=rules.get("max"),
                unique_values=rules.get("allowed_values"),
                regex_pattern=rules.get("regex")
            )
        self.logger.info(f"Loaded schema with {len(self.schema)} features.")

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate a single dataframe against the loaded schema."""
        results = {
            "missing_values": {},
            "type_violations": {},
            "range_violations": {},
            "unique_violations": {},
            "pattern_violations": {},
            "missing_columns": []
        }

        for feature, schema in self.schema.items():
            if feature not in df.columns:
                results["missing_columns"].append(feature)
                continue

            # Type casting
            try:
                df[feature] = df[feature].astype(schema.dtype)
            except Exception as e:
                results["type_violations"][feature] = str(e)

            # Null check
            if not schema.nullable:
                null_indices = df[df[feature].isnull()].index.tolist()
                if null_indices:
                    results["missing_values"][feature] = null_indices

            # Range check
            violations = []
            if schema.min_value is not None:
                violations.extend(df[df[feature] < schema.min_value].index.tolist())
            if schema.max_value is not None:
                violations.extend(df[df[feature] > schema.max_value].index.tolist())
            if violations:
                results["range_violations"][feature] = violations

            # Unique values check
            if schema.unique_values is not None:
                invalid = df[~df[feature].isin(schema.unique_values)]
                if not invalid.empty:
                    results["unique_violations"][feature] = invalid.index.tolist()

        return df, results

    def save_results(self, results: Dict, output_path: str):
        """Save validation results to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Validation results saved to {output_path}")

    def _results_have_errors(self, results: Dict) -> bool:
        """Return True if any validation violations were found."""
        return bool(
            results.get("missing_columns") or
            results.get("missing_values") or
            results.get("type_violations") or
            results.get("range_violations") or
            results.get("unique_violations")
        )

    def validate_raw(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate a consolidated vibration parquet file.
        Checks for expected sensor columns, non-null accelerations,
        and physically plausible acceleration ranges.
        """
        results = {
            "missing_values": {},
            "type_violations": {},
            "range_violations": {},
            "unique_violations": {},
            "pattern_violations": {},
            "missing_columns": [],
            "row_count": len(df),
            "file_count": df["file_id"].nunique() if "file_id" in df.columns else None
        }

        required_cols = ["Hour", "Minute", "Second", "Microsecond",
                         "Horizontal_Accel", "Vertical_Accel",
                         "file_id", "Time_Seconds"]

        for col in required_cols:
            if col not in df.columns:
                results["missing_columns"].append(col)

        if results["missing_columns"]:
            self.logger.warning(f"Missing columns: {results['missing_columns']}")
            return df, results

        # Null checks on critical columns
        for col in ["Horizontal_Accel", "Vertical_Accel", "Time_Seconds"]:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                results["missing_values"][col] = int(null_count)
                self.logger.warning(f"{col} has {null_count} null values.")

        # Physical range: accelerations should be within [-50g, +50g] for PHM 2012
        for col in ["Horizontal_Accel", "Vertical_Accel"]:
            out_of_range = df[(df[col].abs() > 50)].index.tolist()
            if out_of_range:
                results["range_violations"][col] = len(out_of_range)
                self.logger.warning(f"{col} has {len(out_of_range)} values outside [-50, 50] g.")

        # Time must be monotonically non-decreasing within each file
        non_monotone_files = []
        for file_id, group in df.groupby("file_id"):
            if not group["Time_Seconds"].is_monotonic_increasing:
                non_monotone_files.append(file_id)
        if non_monotone_files:
            results["unique_violations"]["Time_Seconds_monotone"] = non_monotone_files
            self.logger.warning(f"Non-monotone timestamps in {len(non_monotone_files)} files.")

        is_valid = not self._results_have_errors(results)
        self.logger.info(
            f"Raw data validation {'PASSED' if is_valid else 'FAILED'} — "
            f"{results['row_count']:,} rows across {results['file_count']} files."
        )
        return df, results

    def validate_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate a feature CSV produced by FeatureExtractorPHM.
        Checks that all expected burst-level feature columns are present,
        non-null, and that RUL values are non-negative.
        """
        results = {
            "missing_values": {},
            "type_violations": {},
            "range_violations": {},
            "unique_violations": {},
            "pattern_violations": {},
            "missing_columns": [],
            "row_count": len(df)
        }

        # All columns produced by FeatureExtractorPHM
        required_cols = [
            "file_id", "burst_idx", "time_s",
            "h_max", "h_min", "h_mean", "h_sd", "h_rms",
            "h_skew", "h_kurt", "h_crest", "h_form",
            "v_max", "v_min", "v_mean", "v_sd", "v_rms",
            "v_skew", "v_kurt", "v_crest", "v_form",
            "RUL_s", "RUL_norm"
        ]

        for col in required_cols:
            if col not in df.columns:
                results["missing_columns"].append(col)

        if results["missing_columns"]:
            self.logger.warning(f"Missing feature columns: {results['missing_columns']}")
            return df, results

        # Null checks on numeric feature columns
        numeric_cols = [c for c in required_cols if c != "file_id"]
        for col in numeric_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                results["missing_values"][col] = int(null_count)
                self.logger.warning(f"{col} has {null_count} null values.")

        # RUL must be >= 0
        for col in ["RUL_s", "RUL_norm"]:
            neg = df[df[col] < 0].index.tolist()
            if neg:
                results["range_violations"][col] = len(neg)
                self.logger.warning(f"{col} has {len(neg)} negative values.")

        # RUL_norm must be in [0, 1]
        out_of_norm = df[(df["RUL_norm"] < 0) | (df["RUL_norm"] > 1)].index.tolist()
        if out_of_norm:
            results["range_violations"]["RUL_norm_bounds"] = len(out_of_norm)
            self.logger.warning(f"RUL_norm has {len(out_of_norm)} values outside [0, 1].")

        # time_s must be non-negative and non-decreasing
        if (df["time_s"] < 0).any():
            results["range_violations"]["time_s_negative"] = int((df["time_s"] < 0).sum())
        if not df["time_s"].is_monotonic_increasing:
            results["unique_violations"]["time_s_monotone"] = True
            self.logger.warning("time_s is not monotonically increasing.")

        # RMS must be positive
        for col in ["h_rms", "v_rms"]:
            non_pos = df[df[col] <= 0].index.tolist()
            if non_pos:
                results["range_violations"][f"{col}_positive"] = len(non_pos)
                self.logger.warning(f"{col} has {len(non_pos)} non-positive values.")

        is_valid = not self._results_have_errors(results)
        self.logger.info(
            f"Feature validation {'PASSED' if is_valid else 'FAILED'} — "
            f"{results['row_count']:,} bursts."
        )
        return df, results


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    BASE  = r"C:\Users\29624134\Downloads\Original Data\Full_Test_Set"
    EXTRA = "Original"

    bearings = [
        "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
        "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
        "Bearing3_3",
    ]

    validator = DataValidatorPHM()

    for bearing_name in bearings:
        print(f"\n{'='*60}\nValidating {bearing_name}\n{'='*60}")
        big_file = Path(BASE) / bearing_name / "Big File"

        # Validate raw consolidated data
        raw_path = big_file / "vibration_consolidated.parquet"
        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
            _, raw_results = validator.validate_raw(raw_df)
            validator.save_results(raw_results, str(big_file / "validation_raw.json"))

        # Validate extracted features
        feat_path = big_file / f"phm2012_vibration_features_{EXTRA}.csv"
        if feat_path.exists():
            feat_df = pd.read_csv(feat_path)
            _, feat_results = validator.validate_features(feat_df)
            validator.save_results(feat_results, str(big_file / "validation_features.json"))