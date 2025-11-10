import pandas as pd
import logging
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaggingTrainer:

    def __init__(self, config: dict):
        self.input_location = config.get("input_location")
        self.output_location = config.get("output_location", "models/bagging_model.pkl")
        self.test_size = config.get("test_size", 0.3)
        self.random_state = config.get("random_state", 42)

    def run(self):
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

        return {
            "model_path": self.output_location,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        }
