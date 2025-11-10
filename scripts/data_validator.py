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

class DataValidator:
    """Validator for feature CSVs produced by FeatureExtractor."""

    def __init__(self, schema_path: Optional[str] = None, log_path: Optional[str] = None):
        # Logger first
        self.logger = logging.getLogger("DataValidator")
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
        for feature, rules in schema_json.get("features", {}).items():
            self.schema[feature] = DataSchema(
                name=feature,
                dtype=rules.get("dtype", "object"),
                nullable=rules.get("nullable", True),
                min_value=rules.get("min"),
                max_value=rules.get("max"),
                unique_values=rules.get("allowed_values"),
                regex_pattern=rules.get("regex")
            )
        self.logger.info(f"Loaded schema with {len(self.schema)} features.")

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
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

            # Unique values
            if schema.unique_values is not None:
                invalid = df[~df[feature].isin(schema.unique_values)]
                if not invalid.empty:
                    results["unique_violations"][feature] = invalid.index.tolist()

        return df, results

    def save_results(self, results: Dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Validation results saved to {output_path}")
