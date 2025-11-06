import os
import logging
import yaml
import pandas as pd
from datetime import datetime

# --- Import your workflow classes ---
from utils.database import FeatureStore
from scripts.data_ingestor import DataIngestor
from scripts.feature_extractor_TDF import FeatureExtractor
from scripts.dataset_builder import DatasetBuilder
from utils.config import load_config

# --- Logger setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def get_run_id():
    """Generate a unique run ID based on current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------
def main():
    run_id = get_run_id()
    workflow = load_config("../config/workflow.yaml")

    steps = workflow["workflows"]["predictive_maintenance"]["steps"]

    # -------------------------------
    # Step 1: Data Ingestion
    # -------------------------------
    ingestion_cfg = next(s for s in steps if s["id"] == "data_ingestion")["config"]
    ingestion_cfg = {
        **ingestion_cfg,
        "output_file": ingestion_cfg["output_location"].format(run_id=run_id),
        "state_location": ingestion_cfg["state_location"].format(run_id=run_id),
        "log_path": ingestion_cfg["log_path"].format(run_id=run_id)
    }

    os.makedirs(os.path.dirname(ingestion_cfg["output_file"]), exist_ok=True)
    ingestor = DataIngestor(
        input_folder=ingestion_cfg["input_location"],
        output_file=ingestion_cfg["output_file"],
        state_location=ingestion_cfg["state_location"],
        segment_length=ingestion_cfg["segment_length"],
        segments_per_file=ingestion_cfg["segments_per_file"],
        log_path=ingestion_cfg["log_path"]
    )

    npz_path = ingestor.ingest()
    logger.info(f"✅ Data ingestion complete. NPZ saved at: {npz_path}")

    # -------------------------------
    # Step 2: Feature Extraction
    # -------------------------------
    feature_cfg = next(s for s in steps if s["id"] == "feature_extraction")["config"]
    feature_cfg = {
        **feature_cfg,
        "input_npz": feature_cfg["input_location"].format(run_id=run_id),
        "output_csv": feature_cfg["output_location"].format(run_id=run_id),
        "log_path": feature_cfg["log_path"].format(run_id=run_id)
    }

    os.makedirs(os.path.dirname(feature_cfg["output_csv"]), exist_ok=True)
    extractor = FeatureExtractor(
        sample_rate=feature_cfg["sample_rate"],
        version=feature_cfg["extractor_version"],
        log_path=feature_cfg["log_path"]
    )

    df_features = extractor.process_npz(
        npz_path=feature_cfg["input_npz"],
        output_csv=feature_cfg["output_csv"]
    )
    logger.info(f"✅ Feature extraction complete. CSV saved at: {feature_cfg['output_csv']}")

    # -------------------------------
    # Step 3: Dataset Builder (Preprocessing)
    # -------------------------------
    builder_cfg = next(s for s in steps if s["id"] == "dataset_builder")["config"]
    builder_cfg = {
        **builder_cfg,
        "input_csv": builder_cfg["input_location"].format(run_id=run_id),
        "output_file": builder_cfg["output_location"].format(run_id=run_id),
        "metadata_path": builder_cfg["feature_metadata_path"].format(run_id=run_id),
        "state_location": builder_cfg["state_location"].format(run_id=run_id),
        "log_path": builder_cfg["log_path"].format(run_id=run_id)
    }

    os.makedirs(os.path.dirname(builder_cfg["output_file"]), exist_ok=True)
    logger.info("🚀 Starting dataset preprocessing step...")

    df = pd.read_csv(builder_cfg["input_csv"])
    dataset_builder = DatasetBuilder(
        one_hot=builder_cfg.get("one_hot_encoding", True),
        scaling_method=builder_cfg.get("scaling_method", "minmax"),
        label_columns=["fault"]  # ✅ auto-preserve your PdM label column
    )

    processed_df, num_cols, cat_cols, label_cols = dataset_builder.preprocess_data(
        df, drop_na=builder_cfg.get("drop_na", False)
    )

    # Save processed data and artifacts
    processed_df.to_parquet(builder_cfg["output_file"], index=False)
    dataset_builder.save_metadata(num_cols, cat_cols, label_cols, builder_cfg["metadata_path"])
    dataset_builder.save_transformers("saved_models")

    # Create state flag
    os.makedirs(os.path.dirname(builder_cfg["state_location"]), exist_ok=True)
    with open(builder_cfg["state_location"], "w") as f:
        f.write("dataset_builder_complete")

    logger.info(f"✅ DatasetBuilder complete. Processed data saved at: {builder_cfg['output_file']}")

    # -------------------------------
    # Step 4: Feature Storage
    # -------------------------------
    storage_cfg = next(s for s in steps if s["id"] == "feature_storage")["config"]
    storage_cfg = {
        **storage_cfg,
        "input_csv_path": storage_cfg["input_csv_path"].format(run_id=run_id)
    }

    feature_store = FeatureStore(
        mongo_uri=storage_cfg["uri"],
        db_name=storage_cfg["database"],
        collection_name=storage_cfg["collection"]
    )

    # Load processed dataset (instead of df_features)
    df_to_store = pd.read_parquet(builder_cfg["output_file"])

    feature_store.save_features(
        dataset_id=storage_cfg["dataset_id"],
        version=storage_cfg["version"],
        df=df_to_store,
        metadata=storage_cfg.get("metadata", {})
    )

    logger.info(f"✅ Features successfully uploaded to MongoDB ({storage_cfg['database']}.{storage_cfg['collection']})")
    logger.info("🎉 Pipeline completed successfully!")

# ----------------------------------------------------------------------
# Run the pipeline
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd  # needed for reading CSV/parquet
    main()
