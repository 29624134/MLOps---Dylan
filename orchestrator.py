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

    def start_workflow(self):
        """Initialize and run workflow"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                validator = DataValidator(schema_path=config.get("schema_path"),log_path=config.get("log_path"))
                #multiple csv files therefor need to iterate over them.
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
                trainer = BaggingTrainer(config)
                outputs = trainer.run()
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


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    executor = WorkflowExecutor("config/workflow.yaml")
    executor.start_workflow()
