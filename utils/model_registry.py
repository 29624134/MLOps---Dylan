import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle states"""
    PENDING = "pending"  # Just trained, awaiting review
    APPROVED = "approved"  # Approved for production
    REJECTED = "rejected"  # Rejected by domain expert
    DEPLOYED = "deployed"  # Currently in production
    ARCHIVED = "archived"  # Retired from production
    EXPERIMENTAL = "experimental"  # From external research


class ModelRegistry:
    """
    Central registry for all trained models.

    Key responsibilities:
    1. Track model versions and metadata
    2. Manage model approval workflow
    3. Retrieve best performing models
    4. Maintain model lineage
    """

    def __init__(self, run_id: str, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = f"workflow_data/{run_id}/models/model_registry/registry.json"
        # Convert to absolute path to avoid working directory issues
        if not os.path.isabs(registry_path):
            registry_path = os.path.abspath(registry_path)

        self.registry_path = registry_path
        self.registry_dir = os.path.dirname(registry_path)
        os.makedirs(self.registry_dir, exist_ok=True)

        logger.info(f"ModelRegistry initialized with path: {self.registry_path}")
        logger.info(f"  Current working directory: {os.getcwd()}")
        logger.info(f"  Registry file exists: {os.path.exists(self.registry_path)}")

        # Initialize registry if doesn't exist
        if not os.path.exists(registry_path):
            logger.info(f"  Creating new registry file at: {registry_path}")
            self._initialize_registry()
        else:
            # Log existing registry contents
            try:
                registry = self._load_registry()
                num_models = len(registry.get("models", []))
                logger.info(f"  Loaded existing registry with {num_models} models")
            except Exception as e:
                logger.error(f"  Failed to load existing registry: {e}")

    def _initialize_registry(self):
        """Create empty registry structure"""
        initial_registry = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "models": []
        }
        self._save_registry(initial_registry)

    def _load_registry(self) -> Dict:
        """Load current registry"""
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save_registry(self, registry: Dict):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def _generate_model_id(self, model_path: str, timestamp: str) -> str:
        """Generate unique model ID"""
        unique_string = f"{model_path}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    def register_model(
            self,
            model_path: str,
            model_type: str,
            target_feature: str,
            metrics: Dict[str, float],
            training_data_info: Dict[str, Any],
            metadata: Optional[Dict] = None
    ) -> str:
        """
        Register a newly trained model.

        Args:
            model_path: Path to saved model file
            model_type: Type of model (e.g., 'BaggingClassifier', 'RandomForest')
            target_feature: Feature being predicted
            metrics: Performance metrics (accuracy, f1, precision, etc.)
            training_data_info: Info about training data (size, version, source)
            metadata: Additional metadata

        Returns:
            model_id: Unique identifier for registered model
        """
        registry = self._load_registry()
        timestamp = datetime.now().isoformat()
        model_id = self._generate_model_id(model_path, timestamp)

        model_entry = {
            "model_id": model_id,
            "model_path": model_path,
            "model_type": model_type,
            "target_feature": target_feature,
            "metrics": metrics,
            "training_data": training_data_info,
            "status": ModelStatus.PENDING.value,
            "created_at": timestamp,
            "approved_at": None,
            "deployed_at": None,
            "approved_by": None,
            "metadata": metadata or {}
        }

        registry["models"].append(model_entry)
        self._save_registry(registry)

        logger.info(f"Registered model {model_id} ({model_type}) for {target_feature}")
        logger.info(f"  Registry now has {len(registry['models'])} models")
        logger.info(f"  Saved to: {self.registry_path}")

        return model_id

    def approve_model(self, model_id: str, approved_by: str) -> bool:
        """
        Approve model for production deployment.

        This would typically be called from the Dashboard after domain expert review.
        """
        registry = self._load_registry()

        for model in registry["models"]:
            if model["model_id"] == model_id:
                if model["status"] != ModelStatus.PENDING.value:
                    logger.warning(f"Model {model_id} is not in PENDING status")
                    return False

                model["status"] = ModelStatus.APPROVED.value
                model["approved_at"] = datetime.now().isoformat()
                model["approved_by"] = approved_by

                self._save_registry(registry)
                logger.info(f"Model {model_id} approved by {approved_by}")
                return True

        logger.error(f"Model {model_id} not found in registry")
        return False

    def reject_model(self, model_id: str, reason: str) -> bool:
        """Reject model with reason"""
        registry = self._load_registry()

        for model in registry["models"]:
            if model["model_id"] == model_id:
                model["status"] = ModelStatus.REJECTED.value
                model["rejection_reason"] = reason
                model["rejected_at"] = datetime.now().isoformat()

                self._save_registry(registry)
                logger.info(f"Model {model_id} rejected: {reason}")
                return True

        return False

    def deploy_model(self, model_id: str) -> bool:
        """Mark model as deployed in production"""
        registry = self._load_registry()

        for model in registry["models"]:
            if model["model_id"] == model_id:
                if model["status"] != ModelStatus.APPROVED.value:
                    logger.error(f"Cannot deploy model {model_id}: not approved")
                    return False

                # Archive any currently deployed models for same target
                self._archive_deployed_models(model["target_feature"])

                model["status"] = ModelStatus.DEPLOYED.value
                model["deployed_at"] = datetime.now().isoformat()

                self._save_registry(registry)
                logger.info(f"Model {model_id} deployed to production")
                return True

        return False

    def _archive_deployed_models(self, target_feature: str):
        """Archive currently deployed models for a target feature"""
        registry = self._load_registry()

        for model in registry["models"]:
            if (model["target_feature"] == target_feature and
                    model["status"] == ModelStatus.DEPLOYED.value):
                model["status"] = ModelStatus.ARCHIVED.value
                model["archived_at"] = datetime.now().isoformat()

        self._save_registry(registry)

    def get_deployed_model(self, target_feature: str) -> Optional[Dict]:
        """Get currently deployed model for a target feature"""
        registry = self._load_registry()

        for model in registry["models"]:
            if (model["target_feature"] == target_feature and
                    model["status"] == ModelStatus.DEPLOYED.value):
                return model

        return None

    def get_best_model(
            self,
            model_type: Optional[str] = None,
            target_feature: Optional[str] = None,
            metric: str = "accuracy",
            status_filter: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Get best performing model based on metric.

        Args:
            model_type: Filter by model type (optional)
            target_feature: Filter by target feature (optional)
            metric: Metric to optimize (default: accuracy)
            status_filter: List of acceptable statuses (default: APPROVED only)

        Returns:
            Best model entry or None
        """
        registry = self._load_registry()

        if status_filter is None:
            status_filter = [ModelStatus.APPROVED.value]

        # Filter models
        candidates = []
        for model in registry["models"]:
            if model["status"] not in status_filter:
                continue
            if model_type and model["model_type"] != model_type:
                continue
            if target_feature and model["target_feature"] != target_feature:
                continue
            if metric not in model["metrics"]:
                continue

            candidates.append(model)

        if not candidates:
            return None

        # Sort by metric (descending)
        best_model = max(candidates, key=lambda m: m["metrics"][metric])
        return best_model

    def list_models(
            self,
            status: Optional[str] = None,
            target_feature: Optional[str] = None
    ) -> List[Dict]:
        """List all models with optional filters"""
        registry = self._load_registry()
        models = registry["models"]

        logger.debug(f"list_models called from {self.registry_path}")
        logger.debug(f"  Total models in registry: {len(models)}")
        logger.debug(f"  Filtering by status={status}, target_feature={target_feature}")

        if status:
            models = [m for m in models if m["status"] == status]
            logger.debug(f"  After status filter: {len(models)} models")

        if target_feature:
            models = [m for m in models if m["target_feature"] == target_feature]
            logger.debug(f"  After target_feature filter: {len(models)} models")

        return models

    def get_model_lineage(self, model_id: str) -> Dict:
        """
        Get lineage information for a model.

        This should track:
        - Training data version
        - Parent models (if ensemble)
        - Experiment that produced it
        - Feature engineering pipeline used
        """
        registry = self._load_registry()

        for model in registry["models"]:
            if model["model_id"] == model_id:
                lineage = {
                    "model_id": model_id,
                    "training_data": model["training_data"],
                    "created_at": model["created_at"],
                    "parent_experiment": model["metadata"].get("experiment_id"),
                    "feature_pipeline_version": model["metadata"].get("pipeline_version"),
                    "hyperparameters": model["metadata"].get("hyperparameters")
                }
                return lineage

        return None

    def get_model_performance_history(self, target_feature: str) -> List[Dict]:
        """
        Get performance history for all models predicting a target feature.

        Useful for tracking model performance over time.
        """
        registry = self._load_registry()

        history = []
        for model in registry["models"]:
            if model["target_feature"] == target_feature:
                history.append({
                    "model_id": model["model_id"],
                    "model_type": model["model_type"],
                    "created_at": model["created_at"],
                    "metrics": model["metrics"],
                    "status": model["status"]
                })

        # Sort by creation time
        history.sort(key=lambda x: x["created_at"])
        return history