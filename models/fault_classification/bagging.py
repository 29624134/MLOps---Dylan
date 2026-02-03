import uuid

import pandas as pd
import logging
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
from utils.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaggingTrainer:

    def __init__(self, config: dict):
        self.input_location = config.get("input_location")
        self.output_location = config.get("output_location", "models/bagging_model.pkl")
        self.test_size = config.get("test_size", 0.3)
        self.random_state = config.get("random_state", 42)
        self.config = config

    def run(self, run_id: str):
        # Load precomputed features
        df = pd.read_csv(self.input_location)
        X = df.drop(columns=["fault"]).values
        y = df["fault"].values

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        # Train Bagging
        clf = BaggingClassifier(n_estimators=10, n_jobs=-1, random_state=self.random_state)
        clf.fit(X_train, y_train)

        # Metrics
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        logger.info(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        # Save model
        os.makedirs(os.path.dirname(self.output_location), exist_ok=True)
        joblib.dump(clf, self.output_location)
        logger.info(f"Model saved to {self.output_location}")

        # --- REGISTER MODEL ---
        registry_path = os.path.abspath(f"workflow_data/{run_id}/models/model_registry/registry.json")
        logger.info(f"Registering model to registry at: {registry_path}")
        registry = ModelRegistry(run_id=run_id)
        # Use the actual path we saved to, not a hardcoded path
        model_path = self.output_location

        training_data_info = {
            "num_samples": len(df),
            "features": list(df.columns.drop("fault")),
            "source": self.input_location
        }

        model_id = registry.register_model(
            model_path=model_path,
            model_type="BaggingClassifier",
            target_feature="fault",
            metrics={"train_acc": train_acc, "test_acc": test_acc},
            training_data_info=training_data_info,
            metadata={"run_id": run_id}
        )
        logger.info(f"Model registered with ID: {model_id} (path: {model_path})")

        return {
            "model_id": model_id,
            "model_path": self.output_location,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        }