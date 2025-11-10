from pymongo import MongoClient
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FeatureStore:
    """MongoDB Feature Store for versioned feature storage, storing each feature type in its own collection."""

    def __init__(self, config: dict):
        self.mongo_uri = config.get("mongo_uri")
        self.db_name = config.get("db_name")
        self.base_collection_name = config.get("collection_name")
        self.dataset_id = config.get("dataset_id")
        self.version = config.get("version")
        self.df_path = config.get("df_path")  # can now be a folder
        self.metadata = config.get("metadata", {})

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.meta_collection = self.db["metadata"]

        logger.info(f"Connected to MongoDB Feature Store → {self.db_name}")

    def run(self):
        """Orchestrator-compatible run method."""
        if not self.df_path:
            raise ValueError("No df_path provided for feature storage.")

        # If df_path is a folder, store each CSV separately
        if os.path.isdir(self.df_path):
            csv_files = [f for f in os.listdir(self.df_path) if f.endswith(".csv")]
            for csv_file in csv_files:
                file_path = os.path.join(self.df_path, csv_file)
                df = pd.read_csv(file_path)
                feature_type = os.path.splitext(csv_file)[0]  # e.g., time_features
                self.save_features(self.dataset_id, self.version, df, self.metadata, feature_type)
        else:
            # Existing behavior for single CSV or Parquet
            ext = os.path.splitext(self.df_path)[-1].lower()
            if ext == ".csv":
                df = pd.read_csv(self.df_path)
            elif ext == ".parquet":
                df = pd.read_parquet(self.df_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            self.save_features(self.dataset_id, self.version, df, self.metadata)

        return {"dataset_id": self.dataset_id, "version": self.version}

    def save_features(self, dataset_id: str, version: str, df: pd.DataFrame, metadata=None, feature_type=None):
        """Save features (with version, feature_type, and metadata) to MongoDB."""
        if feature_type:
            collection_name = f"{self.base_collection_name}{feature_type}"  # each feature type in its own collection
        else:
            collection_name = self.base_collection_name

        collection = self.db[collection_name]

        records = df.to_dict("records")
        for record in records:
            record["_dataset_id"] = dataset_id
            record["_version"] = version
            if feature_type:
                record["_feature_type"] = feature_type

        result = collection.insert_many(records)
        logger.info(f"Stored {len(result.inserted_ids)} records for {dataset_id} (v{version}) - type: {feature_type}")

        meta = {
            "dataset_id": dataset_id,
            "version": version,
            "feature_type": feature_type,
            "row_count": len(df),
            "columns": list(df.columns),
            "created_at": pd.Timestamp.now().isoformat(),
            **(metadata or {})
        }
        self.meta_collection.insert_one(meta)
        logger.info(f"Metadata saved for {dataset_id} (v{version}) - type: {feature_type}")

    def get_features(self, dataset_id: str, version: str = "latest", feature_type: str = None) -> pd.DataFrame:
        """Retrieve features by dataset_id, version, and optionally feature_type."""
        if not feature_type:
            raise ValueError("Must provide feature_type to select the correct collection in this setup.")

        collection_name = f"{self.base_collection_name}{feature_type}"
        collection = self.db[collection_name]

        if version == "latest":
            query = {"dataset_id": dataset_id}
            meta = self.meta_collection.find_one({**query, "feature_type": feature_type}, sort=[("created_at", -1)])
            if not meta:
                raise ValueError(f"No metadata found for dataset '{dataset_id}' and feature_type '{feature_type}'")
            version = meta["version"]

        query = {"_dataset_id": dataset_id, "_version": version}
        records = list(collection.find(query))
        if not records:
            raise ValueError(f"No records found for dataset '{dataset_id}' (v{version}) - type: {feature_type}")

        df = pd.DataFrame(records)
        df.drop(columns=["_id", "_dataset_id", "_version", "_feature_type"], inplace=True, errors="ignore")
        logger.info(f"Retrieved {len(df)} records for {dataset_id} (v{version}) - type: {feature_type}")
        return df
