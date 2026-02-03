import os
import uuid

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from utils.model_registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BoostingTrainer:
    def __init__(self, config: dict):
        self.input_location = config.get("input_location")  # CSV or npz path
        self.output_location = config.get("output_location", "models/boosting_model.pkl")
        self.test_size = config.get("test_size", 0.3)
        self.random_state = config.get("random_state", 42)
        self.config = config

    def run(self, run_id: str):
        # Load NPZ file
        file = np.load(self.input_location, allow_pickle=True)
        data = file["data"]
        labels = file["labels"]

        # Reshape (as in your current code)
        resized_data = np.reshape(data, (1150, 1024))

        # Compute wavelet packet energy features
        import pywt
        wp = pywt.WaveletPacket(resized_data[0, :], wavelet="sym8", maxlevel=3)
        packet_names = [node.path for node in wp.get_level(3, "natural")]
        feature_matrix = np.empty((resized_data.shape[0], len(packet_names)))
        for i in range(resized_data.shape[0]):
            wp = pywt.WaveletPacket(resized_data[i, :], wavelet="sym8", maxlevel=3)
            for j, name in enumerate(packet_names):
                new_wp = pywt.WaveletPacket(data=None, wavelet="sym8", maxlevel=3)
                new_wp[name] = wp[name].data
                feature_matrix[i, j] = np.linalg.norm(new_wp.reconstruct(update=False)) ** 2

        # Re-encode labels (your categories)
        categories = ["Ball_007","Ball_014","Ball_021","IR_007","IR_014","IR_021",
                      "OR_007","OR_014","OR_021","Normal"]
        labels = pd.Categorical(np.repeat(categories, repeats=115), categories=categories)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, labels, test_size=self.test_size,
            stratify=labels, random_state=self.random_state
        )

        # Train Gradient Boosting
        clf = GradientBoostingClassifier(
            n_estimators=self.config.get("n_estimators", 50),
            max_depth=self.config.get("max_depth", 2),
            subsample=self.config.get("subsample", 0.5),
            random_state=self.random_state
        )
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
            "num_samples": feature_matrix.shape[0],
            "num_features": feature_matrix.shape[1],
            "source": self.input_location
        }

        model_id = registry.register_model(
            model_path=model_path,
            model_type="BoostingClassifier",
            target_feature="fault",  # or whatever your predicted column/label is called
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