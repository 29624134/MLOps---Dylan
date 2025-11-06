import os
import json
import joblib
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

class DatasetBuilder:
    """
    Preprocess extracted vibration feature data for model training.
    - Numeric feature scaling
    - Categorical feature encoding
    - Label columns detection
    - Save scalers/encoders for reuse
    """

    def __init__(self, config: dict, label_columns=None):
        """
        Args:
            config (dict): Orchestrator step config
            label_columns (list): Columns to exclude from preprocessing
        """
        self.config = config
        self.label_columns = label_columns or ["fault", "label", "fault_type"]
        self.scalers = {}
        self.encoders = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Step-specific config
        self.input_csv = config.get("input_csv") or config.get("input_location")
        self.output_file = config.get("output_file") or config.get("output_location")
        self.feature_metadata_path = config.get("feature_metadata_path", "metadata/feature_metadata.json")
        self.one_hot = config.get("one_hot_encoding", True)
        self.scaling_method = config.get("scaling_method", "minmax")
        self.drop_na = config.get("drop_na", False)
        self.save_scalers = config.get("save_scalers", True)
        self.save_encoders = config.get("save_encoders", True)

    def run(self):
        """Run preprocessing as orchestrator step."""
        if not self.input_csv or not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")

        self.logger.info(f"Loading data from {self.input_csv}")
        df = pd.read_csv(self.input_csv)

        # Separate labels
        label_cols_present = [col for col in self.label_columns if col in df.columns]
        labels_df = df[label_cols_present].copy() if label_cols_present else pd.DataFrame()
        df_features = df.drop(columns=label_cols_present, errors="ignore")

        # Identify numeric and categorical columns
        numeric_cols = df_features.select_dtypes(include="number").columns.tolist()
        cat_cols = df_features.select_dtypes(exclude="number").columns.tolist()
        self.logger.info(f"Numeric columns: {numeric_cols}")
        self.logger.info(f"Categorical columns: {cat_cols}")
        processed_df = pd.DataFrame()

        # Scale numeric features
        scaler_cls = MinMaxScaler if self.scaling_method.lower() == "minmax" else StandardScaler
        for col in numeric_cols:
            scaler = scaler_cls()
            df_features[col] = scaler.fit_transform(df_features[[col]].astype(float))
            self.scalers[col] = scaler
        if numeric_cols:
            processed_df[numeric_cols] = df_features[numeric_cols]

        # Encode categorical features
        for col in cat_cols:
            if self.one_hot:
                ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoded = ohe.fit_transform(df_features[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                processed_df[ohe_cols] = pd.DataFrame(encoded, index=df_features.index)
                self.encoders[col] = ohe
            else:
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
                processed_df[col] = df_features[col]
                self.encoders[col] = le

        # Drop NaN if requested
        if self.drop_na:
            before = len(processed_df)
            processed_df = processed_df.dropna()
            after = len(processed_df)
            self.logger.info(f"Dropped {before - after} rows containing NaN")

        # Reattach labels
        if not labels_df.empty:
            processed_df = pd.concat([processed_df.reset_index(drop=True),
                                      labels_df.reset_index(drop=True)], axis=1)

        # Ensure output folder exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        processed_df.to_parquet(self.output_file, index=False)
        self.logger.info(f"Processed dataset saved to {self.output_file}")

        # Save metadata
        os.makedirs(os.path.dirname(self.feature_metadata_path), exist_ok=True)
        metadata = {
            "numeric_features": numeric_cols,
            "categorical_features": cat_cols,
            "label_features": label_cols_present
        }
        with open(self.feature_metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        self.logger.info(f"Feature metadata saved to {self.feature_metadata_path}")

        # Save scalers/encoders
        if self.save_scalers:
            scaler_dir = os.path.join(os.path.dirname(self.output_file), "scalers")
            os.makedirs(scaler_dir, exist_ok=True)
            for col, scaler in self.scalers.items():
                joblib.dump(scaler, os.path.join(scaler_dir, f"{col}_scaler.pkl"))
            self.logger.info(f"Saved {len(self.scalers)} scalers to {scaler_dir}")
        if self.save_encoders:
            encoder_dir = os.path.join(os.path.dirname(self.output_file), "encoders")
            os.makedirs(encoder_dir, exist_ok=True)
            for col, enc in self.encoders.items():
                joblib.dump(enc, os.path.join(encoder_dir, f"{col}_encoder.pkl"))
            self.logger.info(f"Saved {len(self.encoders)} encoders to {encoder_dir}")

        return {
            "processed_dataset": self.output_file,
            "metadata": self.feature_metadata_path
        }
