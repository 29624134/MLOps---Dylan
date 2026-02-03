# orchestrator.py
import os
import json
import yaml
import time
import logging
from datetime import datetime
from typing import Dict, Any
from scripts.data_ingestor import DataIngestor
from scripts.feature_extractor import FeatureExtractor
from utils.MongoDB import FeatureStore
from utils.config import load_config
from scripts.dataset_builder import DatasetBuilder
from models.fault_classification.bagging import BaggingTrainer
from models.fault_classification.boosting import BoostingTrainer
from scripts.data_validator import DataValidator
import pandas as pd

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# WORKFLOW STATE MANAGER
# -----------------------------
class WorkflowStateManager:
    """Manages the progress and state of each workflow run"""

    def __init__(self, base_dir="workflow_data", run_id=None):
        self.base_dir = base_dir
        self.run_id = run_id

    def create_run_state(self, run_id: str, workflow_def: dict) -> Dict[str, Any]:
        state = {
            "run_id": run_id,
            "status": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "steps": {
                step["id"]: {
                    "status": "PENDING",
                    "start_time": None,
                    "end_time": None,
                    "outputs": {},
                    "error": None,
                }
                for step in workflow_def["steps"]
            },
        }
        self.save_state(run_id, state)
        return state

    def get_state_path(self, run_id=None):
        run = run_id or self.run_id
        return os.path.join(self.base_dir, run, "state", f"{run}.json")

    def save_state(self, run_id: str, state: Dict[str, Any]):
        state_path = self.get_state_path(run_id)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, run_id: str) -> Dict[str, Any]:
        """Read from file"""
        with open(self.get_state_path(run_id), "r") as f:
            return json.load(f)

    def update_step_status(self, run_id: str, step_id: str, status: str, error: str = None):
        """Update step status"""
        state = self.load_state(run_id)
        step = state["steps"][step_id]
        step["status"] = status

        if status == "RUNNING":
            step["start_time"] = datetime.now().isoformat()
        elif status in ["COMPLETE", "FAILED"]:
            step["end_time"] = datetime.now().isoformat()

        if error:
            step["error"] = str(error)

        self.save_state(run_id, state)


# -----------------------------
# DATA CONTRACT MANAGER
# -----------------------------
class DataContractManager:
    """Resolves and tracks input/output/state paths"""

    def __init__(self, base_dir="workflow_data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def resolve(self, path_template: str, run_id: str) -> str:
        if not path_template:
            return ""
        return path_template.format(run_id=run_id)


# -----------------------------
# MAIN WORKFLOW EXECUTOR
# -----------------------------
class WorkflowExecutor:
    def __init__(self, yaml_path="config/workflow.yaml"):
        with open(yaml_path, "r") as f:
            workflows = yaml.safe_load(f)["workflows"]
        self.workflow_def = workflows["predictive_maintenance"]
        self.state_manager = WorkflowStateManager()
        self.contract_manager = DataContractManager()

    def start_workflow(self, run_id: str):
        """Initialize and run workflow"""
        logger.info(f"Starting workflow with run_id: {run_id}")

        state = self.state_manager.create_run_state(run_id, self.workflow_def)
        steps = self.workflow_def["steps"]

        for step in steps:
            step_id = step["id"]
            depends_on = step.get("depends_on", [])

            # Wait for dependencies
            if depends_on:
                for dep in depends_on:
                    dep_status = state["steps"][dep]["status"]
                    if dep_status != "COMPLETE":
                        logger.warning(f"Step {step_id} waiting for dependency {dep}")
                        continue

            self.execute_step(run_id, step)

        logger.info(f"Workflow {run_id} complete!")

    def execute_step(self, run_id: str, step: Dict[str, Any]):
        """Executes a single step based on its type"""
        step_id = step["id"]
        step_type = step["type"]
        config = step["config"]

        # Resolve all {run_id} placeholders
        config = {
            key: self.contract_manager.resolve(str(val), run_id) if isinstance(val, str) else val
            for key, val in config.items()
        }

        logger.info(f"[{run_id}] Starting step '{step_id}' ({step_type})")
        self.state_manager.update_step_status(run_id, step_id, "RUNNING")

        try:
            start_time = time.time()
            if step_type == "ingestion":
                ingestor = DataIngestor(config)
                outputs = ingestor.run()

            elif step_type == "feature_engineering":
                extractor = FeatureExtractor(config)
                outputs = extractor.run()

            elif step_type == "validation":
                validator = DataValidator(schema_path=config.get("schema_path"), log_path=config.get("log_path"))
                # multiple csv files therefor need to iterate over them.
                input_location = config.get("input_location")
                results_list = []

                for file in os.listdir(input_location):
                    if file.endswith(".csv"):
                        df = pd.read_csv(os.path.join(input_location, file))
                        df, result = validator.validate_dataframe(df)
                        results_list.append({file: result})
                validator.save_results(results_list, config.get("output_location"))
                outputs = {"validation_results": config.get("output_location")}

            elif step_type == "preprocessing":
                extractor = DatasetBuilder(config)
                outputs = extractor.run()

            elif step_type == "storage":
                storage = FeatureStore(config)
                outputs = storage.run()


            elif step_type == "training":
                trainer_type = config.get("trainer_type", "bagging").lower()

                if trainer_type == "bagging":
                    trainer = BaggingTrainer(config)
                elif trainer_type == "boosting":
                    trainer = BoostingTrainer(config)
                else:
                    raise ValueError(f"Unknown trainer type: {trainer_type}")
                outputs = trainer.run(run_id=run_id)

            elif step_type == "model_selection":
                best_model = self.select_best_model(run_id, metric=config.get("metric", "test_accuracy"))
                outputs = {"best_model_id": best_model["model_id"]}
            else:
                raise ValueError(f"Unknown step type: {step_type}")

            duration = time.time() - start_time
            logger.info(f"[{run_id}] Step '{step_id}' complete in {duration:.2f}s")

            # Update state
            state = self.state_manager.load_state(run_id)
            state["steps"][step_id]["outputs"] = outputs
            self.state_manager.update_step_status(run_id, step_id, "COMPLETE")

            # Write flag file
            if "state_location" in config:
                flag_path = config["state_location"]
                os.makedirs(os.path.dirname(flag_path), exist_ok=True)
                with open(flag_path, "w") as f:
                    f.write(datetime.now().isoformat())

        except Exception as e:
            logger.error(f"[{run_id}] Step '{step_id}' failed: {e}", exc_info=True)
            self.state_manager.update_step_status(run_id, step_id, "FAILED", str(e))

    def select_best_model(self, run_id: str, metric: str = "test_accuracy"):
        from utils.model_registry import ModelRegistry
        import time

        # Small delay to ensure file system has flushed registry writes
        time.sleep(5)

        # Use absolute path to ensure we're reading the same file
        registry_path = os.path.abspath(f"workflow_data/{run_id}/models/model_registry/registry.json")
        logger.info(f"[{run_id}] Using registry path: {registry_path}")

        registry = ModelRegistry(run_id=run_id)

        # Debug: Show registry file location
        logger.info(f"[{run_id}] Reading registry from: {registry.registry_path}")
        logger.info(f"[{run_id}] Registry file exists: {os.path.exists(registry.registry_path)}")

        # Check file size
        if os.path.exists(registry.registry_path):
            file_size = os.path.getsize(registry.registry_path)
            logger.info(f"[{run_id}] Registry file size: {file_size} bytes")

        models = registry.list_models(status="pending") + registry.list_models(status="approved")

        logger.info(f"[{run_id}] Found {len(models)} models in registry with PENDING/APPROVED status")

        # Debug: Log all models and their metadata
        for m in models:
            m_run_id = m.get('metadata', {}).get('run_id')
            match = "✓ MATCH" if m_run_id == run_id else "✗ no match"
            logger.info(f"  Model {m['model_id']}: run_id={m_run_id} {match}")

        # Filter only models produced in this run
        models_this_run = [m for m in models if m.get("metadata", {}).get("run_id") == run_id]

        logger.info(f"[{run_id}] Filtered to {len(models_this_run)} models for this run")

        if not models_this_run:
            logger.error(f"[{run_id}] FILTERING FAILED!")
            logger.error(f"  Looking for run_id: '{run_id}'")
            logger.error(f"  run_id type: {type(run_id)}")
            if models:
                sample_run_id = models[0].get('metadata', {}).get('run_id')
                logger.error(f"  Sample model run_id: '{sample_run_id}'")
                logger.error(f"  Sample run_id type: {type(sample_run_id)}")
                logger.error(f"  Are they equal? {sample_run_id == run_id}")
                logger.error(f"  Byte comparison: {repr(run_id)} vs {repr(sample_run_id)}")
            raise ValueError(
                f"No models were registered for run_id='{run_id}'. Total models in registry: {len(models)}")

        # Map metric names
        metric_key_map = {"test_accuracy": "test_acc", "train_accuracy": "train_acc"}
        metric_key = metric_key_map.get(metric, metric)

        # Pick best by metric
        best_model = max(models_this_run, key=lambda m: m["metrics"].get(metric_key, 0))

        registry.approve_model(best_model["model_id"], approved_by="orchestrator")
        registry.deploy_model(best_model["model_id"])

        logger.info(
            f"[{run_id}] Best model selected: {best_model['model_id']} ({best_model['model_type']}) "
            f"with {metric_key}={best_model['metrics'].get(metric_key, 0):.4f}"
        )
        return best_model


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    executor = WorkflowExecutor("config/workflow.yaml")
    executor.start_workflow()