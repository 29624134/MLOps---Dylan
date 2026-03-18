from typing import Dict, Type, List
from models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)



class ModelTypeRegistry:
    """
    Registry for available model types/classes.

    This is separate from the workflow ModelRegistry which tracks
    trained model instances and their lifecycle.

    This registry answers: "What types of models can I train?"
    The workflow ModelRegistry answers: "What models have I trained?"
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]):
        """
        Register a model class.

        Args:
            model_type: String identifier for the model (e.g., "bagging", "knn")
            model_class: The model class (must inherit from BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"{model_class} must inherit from BaseModel")

        cls._models[model_type.lower()] = model_class
        logger.debug(f"Registered model type: {model_type} -> {model_class.__name__}")

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """
        Get a registered model class.

        Args:
            model_type: String identifier for the model

        Returns:
            Model class

        Raises:
            ValueError: If model_type is not registered
        """
        model_type = model_type.lower()

        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )

        return cls._models[model_type]

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create an instance of a registered model.

        Args:
            model_type: String identifier for the model
            **kwargs: Hyperparameters for the model

        Returns:
            Model instance
        """
        model_class = cls.get_model_class(model_type)
        return model_class(**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered model types.

        Returns:
            List of model type strings
        """
        return list(cls._models.keys())

    @classmethod
    def get_model_info(cls, model_type: str) -> Dict:
        """
        Get information about a model type.

        Args:
            model_type: String identifier for the model

        Returns:
            Dictionary with model information
        """
        model_class = cls.get_model_class(model_type)
        instance = model_class()

        return {
            "model_type": model_type,
            "class_name": model_class.__name__,
            "model_name": model_class.get_model_name(),
            "default_params": instance.get_default_params()
        }


# Auto-register all models
def register_all_models():
    """Register all available models."""
    from models.fault_classification.bagging_model import BaggingModel
    from models.fault_classification.boosting_model import BoostingModel
    from models.fault_classification.decisiontree_model import DecisionTreeModel
    from models.fault_classification.randomforest_model import RandomForestModel
    from models.fault_classification.knn_model import KNNModel
    from models.fault_classification.lda_model import LDAModel
    from models.fault_classification.qda_model import QDAModel

    ModelTypeRegistry.register("bagging", BaggingModel)
    ModelTypeRegistry.register("boosting", BoostingModel)
    ModelTypeRegistry.register("decisiontree", DecisionTreeModel)
    ModelTypeRegistry.register("randomforest", RandomForestModel)
    ModelTypeRegistry.register("knn", KNNModel)
    ModelTypeRegistry.register("lda", LDAModel)
    ModelTypeRegistry.register("qda", QDAModel)

    logger.info(f"Registered {len(ModelTypeRegistry.list_models())} model types")


# Register models on import
register_all_models()