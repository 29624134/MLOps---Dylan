"""
orchestrator.py
═══════════════════════════════════════════════════════════════════════════════
Workflow Orchestrator — bearing-by-bearing edition.

start_workflow() behaviour
──────────────────────────
FIRST RUN  (no deployed model in model registry):
    1. Ingest   current live bearing
    2. Extract  current live bearing
    2b. Backfill MongoDB for train/val bearings
    3. Validate val bearings
    4. Train    on train bearings
    5. Select & deploy best model
    6. Serving Pipeline — current live bearing

SUBSEQUENT RUNS  (deployed model already exists, triggered via /bearing/continue):
    1. Ingest   current live bearing
    2. Extract  current live bearing
    3. Serving Pipeline — current live bearing
    (Training and model selection are skipped.)

Fault confirmation flow (called from API after tech confirms):
    confirm_fault_and_push_to_store() → re-labels features → pushes to MongoDB
    advance_live_bearing()            → increments queue index in bearings.json
"""

import os
import json
import yaml
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

from scripts.data_ingestor import DataIngestorPHM
from scripts.feature_extractor import FeatureExtractorPHM
from scripts.data_validator import DataValidatorPHM
from models.model_trainer import RULTrainerPHM
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BEARING REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class BearingRegistry:
    """
    Loads config/bearings.json and tracks bearing state.

    roles
    ─────
    train  — run-to-failure bearings used for model training
    val    — validation bearings used during training evaluation
    test   — held-out test bearings (not used in current pipeline)
    live   — bearings queued for live inference, processed one at a time

    status values
    ─────────────
    available  — data on disk, ready to ingest
    missing    — data not yet on disk
    ingested   — CSVs consolidated into parquet
    extracted  — features.csv written
    confirmed  — tech confirmed fault; features pushed to Feature Store
    denied     — tech denied fault; features NOT pushed
    error      — a pipeline step failed
    """

    VALID_STATUSES = {
        "available", "missing", "ingested", "extracted",
        "confirmed", "denied", "error",
    }

    def __init__(self, config_path: str = "config/bearings.json"):
        self.config_path = config_path
        self._load()

    def _load(self):
        with open(self.config_path, "r") as f:
            data = json.load(f)
        self.base_path          = data["base_path"]
        self.bearings           = data["bearings"]
        self.live_queue         = data.get("live_bearing_queue", [])
        self.current_live_index = data.get("current_live_index", 0)

    def _save(self):
        with open(self.config_path, "r") as f:
            data = json.load(f)
        data["bearings"]           = self.bearings
        data["current_live_index"] = self.current_live_index
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Queries ───────────────────────────────────────────────────────────────

    def all_bearings(self) -> List[Dict]:
        return self.bearings

    def train_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "train"]

    def val_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "val"]

    def live_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "live"]

    def current_live_bearing(self) -> Optional[Dict]:
        """Return the single live bearing currently being processed from the queue."""
        if not self.live_queue:
            return None
        if self.current_live_index >= len(self.live_queue):
            logger.warning("Live bearing queue exhausted.")
            return None
        name = self.live_queue[self.current_live_index]
        return self.get_bearing(name)

    def advance_live_bearing(self) -> Optional[Dict]:
        """Increment the queue pointer and persist it to disk."""
        self.current_live_index += 1
        self._save()
        next_b = self.current_live_bearing()
        if next_b:
            logger.info(f"Live queue advanced → now serving: {next_b['name']}")
        else:
            logger.info("Live bearing queue exhausted — no more bearings.")
        return next_b

    def is_test(self, bearing: Dict) -> bool:
        return bearing["role"] != "train"

    def source_path(self, bearing: Dict) -> str:
        return os.path.join(self.base_path, bearing["name"])

    def get_bearing(self, name: str) -> Optional[Dict]:
        for b in self.bearings:
            if b["name"] == name:
                return b
        return None

    def set_status(self, bearing_name: str, status: str):
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'.")
        for b in self.bearings:
            if b["name"] == bearing_name:
                b["status"] = status
                self._save()
                return
        raise KeyError(f"Bearing '{bearing_name}' not in registry.")

    def print_status(self):
        from collections import Counter
        counts = Counter(b.get("status", "unknown") for b in self.bearings)
        logger.info("── Bearing registry status ──────────────────────")
        for b in self.bearings:
            exists = "exists" if os.path.isdir(self.source_path(b)) else "NO FOLDER"
            logger.info(
                f"  {b['name']:<14} role={b['role']:<6} "
                f"status={b.get('status','?'):<10} disk={exists}"
            )
        logger.info(f"  Summary: {dict(counts)}")
        logger.info("─────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowStateManager:
    STATE_DIR = "workflow_data"

    def _state_path(self, run_id: str) -> str:
        return os.path.join(self.STATE_DIR, run_id, "workflow_state.json")

    def init_state(self, run_id: str, workflow_name: str) -> Dict:
        state = {
            "run_id":        run_id,
            "workflow_name": workflow_name,
            "status":        "RUNNING",
            "start_time":    datetime.now().isoformat(),
            "end_time":      None,
            "steps":         {},
        }
        os.makedirs(os.path.dirname(self._state_path(run_id)), exist_ok=True)
        self._write(run_id, state)
        return state

    def load_state(self, run_id: str) -> Dict:
        path = self._state_path(run_id)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def update_step_status(self, run_id: str, step_id: str,
                           status: str, error: str = None):
        state = self.load_state(run_id) or {"steps": {}}
        state.setdefault("steps", {}).setdefault(step_id, {})
        state["steps"][step_id]["status"] = status
        if error:
            state["steps"][step_id]["error"] = error
        if status == "RUNNING":
            state["steps"][step_id]["start_time"] = datetime.now().isoformat()
        if status in ("COMPLETE", "FAILED", "SKIPPED"):
            state["steps"][step_id]["end_time"] = datetime.now().isoformat()
        self._write(run_id, state)

    def mark_step_outputs(self, run_id: str, step_id: str, outputs: Dict):
        state = self.load_state(run_id) or {"steps": {}}
        state.setdefault("steps", {}).setdefault(step_id, {})["outputs"] = outputs
        self._write(run_id, state)

    def mark_workflow_complete(self, run_id: str):
        state = self.load_state(run_id)
        state["status"]   = "COMPLETE"
        state["end_time"] = datetime.now().isoformat()
        self._write(run_id, state)

    def mark_workflow_failed(self, run_id: str, error: str):
        state = self.load_state(run_id)
        if not state:
            state = {"run_id": run_id, "steps": {}}
        state["status"]   = "FAILED"
        state["error"]    = error
        state["end_time"] = datetime.now().isoformat()
        self._write(run_id, state)

    def _write(self, run_id: str, state: Dict):
        path = self._state_path(run_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ContractManager:
    def resolve_config(self, config: Dict, run_id: str) -> Dict:
        resolved = {}
        for k, v in config.items():
            if isinstance(v, str):
                resolved[k] = v.replace("{run_id}", run_id)
            elif isinstance(v, dict):
                resolved[k] = self.resolve_config(v, run_id)
            else:
                resolved[k] = v
        return resolved


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowExecutor:

    def __init__(self, yaml_path: str = "config/workflow.yaml",
                 workflow_name: str = "rul_prediction"):
        self.state_manager    = WorkflowStateManager()
        self.contract_manager = ContractManager()

        # Try workflow registry first, fall back to YAML
        try:
            from utils.workflow_registry import WorkflowRegistry
            reg    = WorkflowRegistry()
            active = reg.get_active_workflow(workflow_name)
            if active:
                self.workflow_def = active["definition"]
            else:
                raise LookupError("no active workflow")
        except Exception:
            logger.warning("WorkflowExecutor: falling back to local YAML.")
            with open(yaml_path, "r") as f:
                self.workflow_def = yaml.safe_load(f)["workflows"][workflow_name]

        bearing_config_path = self.workflow_def.get("bearing_config", "config/bearings.json")
        self.registry = BearingRegistry(bearing_config_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _mongo_config(self) -> Dict:
        mongo = self.workflow_def.get("mongodb", {})
        return {
            "enabled": bool(mongo.get("enabled", False)),
            "uri":     mongo.get("uri",     "mongodb://localhost:27017"),
            "db_name": mongo.get("db_name", "phm_mlops"),
        }

    def _step(self, step_id: str) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == step_id)

    def _ingestion_template(self) -> Dict:
        return self._step("ingestion")["config"]

    def _extraction_template(self) -> Dict:
        return self._step("feature_engineering")["config"]

    def _validation_step(self) -> Dict:
        return self._step("validation")

    def _training_step(self) -> Dict:
        return self._step("training")

    def _model_selection_step(self) -> Dict:
        return self._step("model_selection")

    def _serving_pipeline_step(self) -> Dict:
        return self._step("serving_pipeline")

    def _deployed_model_exists(self) -> bool:
        """Check whether any deployed RUL model already exists in the registry."""
        from utils.model_registry import ModelRegistry
        entry = ModelRegistry().get_deployed_model("RUL_s")
        return entry is not None

    def _execute(self, run_id: str, step_id: str, step_type: str,
                 config: Dict, fn):
        self.state_manager.update_step_status(run_id, step_id, "RUNNING")
        logger.info(f"  [{run_id}] Step '{step_id}' ({step_type}) — RUNNING")
        try:
            outputs = fn(config)
            self.state_manager.update_step_status(run_id, step_id, "COMPLETE")
            if outputs:
                self.state_manager.mark_step_outputs(run_id, step_id, outputs)
            logger.info(f"  [{run_id}] Step '{step_id}' — COMPLETE")
        except Exception as e:
            self.state_manager.update_step_status(run_id, step_id, "FAILED", str(e))
            logger.error(f"  [{run_id}] Step '{step_id}' — FAILED: {e}", exc_info=True)
            raise

    # ── Main workflow entry point ─────────────────────────────────────────────

    def start_workflow(self, workflow_name: str = "rul_prediction",
                       config_overrides: Dict = None) -> str:
        """
        Bearing-by-bearing workflow.

        If no deployed model exists (first run):
            1. Ingest   current live bearing
            2. Extract  current live bearing
            2b. MongoDB backfill (train/val)
            3. Validate val bearings
            4. Train    on train bearings
            5. Select & deploy best model
            6. Serving Pipeline — current live bearing

        If a deployed model already exists (subsequent runs):
            1. Ingest   current live bearing
            2. Extract  current live bearing
            6. Serving Pipeline — current live bearing
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state_manager.init_state(run_id, workflow_name)
        self.registry.print_status()

        live_bearing    = self.registry.current_live_bearing()
        model_deployed  = self._deployed_model_exists()

        if live_bearing:
            logger.info(
                f"[{run_id}] Current live bearing: {live_bearing['name']} | "
                f"Deployed model: {'YES' if model_deployed else 'NO — will train first'}"
            )
        else:
            logger.warning(f"[{run_id}] No live bearing in queue.")

        try:
            # ── 1. Extract live bearing (reads directly from acc_*.csv) ──────
            if live_bearing:
                logger.info(f"[{run_id}] Phase 1: Feature extraction — {live_bearing['name']}")
                self._run_extraction(run_id, live_bearing)
            else:
                logger.info(f"[{run_id}] Phase 1: Feature extraction — skipped (no live bearing)")

            if not model_deployed:
                # ── First run: ensure train/val bearings are extracted ────────
                # Train and val bearings are not processed via the live queue,
                # so we ingest + extract any that are missing features.csv here.
                logger.info(f"[{run_id}] Phase 1a: Extract train/val bearings (from acc_*.csv)")
                for b in self.registry.train_bearings() + self.registry.val_bearings():
                    source_folder = self.registry.source_path(b)
                    features_path = os.path.join(source_folder, "features.csv")
                    if os.path.exists(features_path):
                        logger.info(f"  [{b['name']}] features.csv exists — skipping.")
                        continue
                    logger.info(f"  [{b['name']}] Missing features.csv — extracting from raw CSVs.")
                    self._run_extraction(run_id, b)

                # 2b. MongoDB backfill
                logger.info(f"[{run_id}] Phase 2b: MongoDB backfill (train/val)")
                self._run_mongo_backfill(run_id)

                # 3. Validation
                logger.info(f"[{run_id}] Phase 3: Validation")
                self._run_validation(run_id)

                # 4. Training
                logger.info(f"[{run_id}] Phase 4: Training")
                self._run_training(run_id)

                # 5. Model selection
                logger.info(f"[{run_id}] Phase 5: Model selection")
                self._run_model_selection(run_id)

            else:
                logger.info(
                    f"[{run_id}] Phases 2b–5: Skipped "
                    f"(deployed model already exists)"
                )

            # ── 6. Serving Pipeline ───────────────────────────────────────────
            if live_bearing:
                logger.info(f"[{run_id}] Phase 6: Serving Pipeline — {live_bearing['name']}")
                self._run_serving_pipeline(run_id)
            else:
                logger.info(f"[{run_id}] Phase 6: Serving Pipeline — skipped")

            self.state_manager.mark_workflow_complete(run_id)
            logger.info(f"[{run_id}] Workflow complete!")

        except Exception as e:
            self.state_manager.mark_workflow_failed(run_id, traceback.format_exc())
            logger.error(f"[{run_id}] Workflow FAILED: {e}", exc_info=True)
            raise

        return run_id

    # ── Per-phase helpers ─────────────────────────────────────────────────────

    def _run_ingestion(self, run_id: str, bearing: Dict):
        step_id       = f"ingest_{bearing['name'].lower()}"
        template      = self.contract_manager.resolve_config(self._ingestion_template(), run_id)
        source_folder = self.registry.source_path(bearing)

        # Skip if already done
        if bearing.get("status") in ("ingested", "extracted", "confirmed"):
            logger.info(f"  [{bearing['name']}] Already ingested — skipping.")
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        config = {
            "input_location":  source_folder,
            "output_location": source_folder,
            "state_location":  os.path.join(template["state_base"], f"{step_id}.flag"),
            "save_format":     template.get("save_format", "parquet"),
            "log_path":        os.path.join(template["log_base"], f"{step_id}.log"),
        }

        self._execute(run_id, step_id, "ingestion", config,
                      lambda cfg: DataIngestorPHM(cfg).run())

        state = self.state_manager.load_state(run_id)
        new_status = ("ingested"
                      if state["steps"][step_id]["status"] == "COMPLETE"
                      else "error")
        self.registry.set_status(bearing["name"], new_status)

    def _run_extraction(self, run_id: str, bearing: Dict):
        step_id       = f"extract_{bearing['name'].lower()}"
        template      = self.contract_manager.resolve_config(self._extraction_template(), run_id)
        source_folder = self.registry.source_path(bearing)
        features_path = os.path.join(source_folder, "features.csv")

        # Skip if already extracted
        if bearing.get("status") == "extracted" and os.path.exists(features_path):
            logger.info(f"  [{bearing['name']}] Features already extracted — skipping.")
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            # Still push to Mongo if not already there
            self._push_features_to_mongo(bearing, features_path, run_id)
            return

        config = {
            "input_location":    source_folder,
            "output_location":   features_path,
            "state_location":    os.path.join(template["state_base"], f"{step_id}.flag"),
            "log_path":          os.path.join(template["log_base"], f"{step_id}.log"),
            "bearing_name":      bearing["name"],
            "is_test":           self.registry.is_test(bearing),
            "burst_period":      template.get("burst_period", 10.0),
            "failure_threshold": template.get("failure_threshold", 20.0),
            "n_consecutive":     template.get("n_consecutive", 5),
        }

        self._execute(run_id, step_id, "feature_engineering", config,
                      lambda cfg: FeatureExtractorPHM(cfg).run())

        state = self.state_manager.load_state(run_id)
        if state["steps"][step_id]["status"] == "COMPLETE":
            self.registry.set_status(bearing["name"], "extracted")
            self._push_features_to_mongo(bearing, features_path, run_id)
        else:
            self.registry.set_status(bearing["name"], "error")

    def _push_features_to_mongo(self, bearing: Dict, features_path: str, run_id: str):
        """Push a features.csv to MongoDB Feature Store."""
        mongo_cfg = self._mongo_config()
        if not mongo_cfg.get("enabled") or not os.path.exists(features_path):
            return
        from utils.MongoDB import FeatureStore
        store = FeatureStore({
            "mongo_uri":       mongo_cfg["uri"],
            "db_name":         mongo_cfg["db_name"],
            "collection_name": "features",
            "dataset_id":      bearing["name"],
            "version":         run_id,
            "df_path":         features_path,
            "metadata": {
                "bearing_name": bearing["name"],
                "role":         bearing["role"],
                "run_id":       run_id,
            },
        })
        store.run()
        logger.info(f"  [{bearing['name']}] Features pushed to MongoDB Feature Store")

    def _run_mongo_backfill(self, run_id: str):
        """Push already-extracted train/val features to MongoDB (skips live bearings)."""
        mongo_cfg = self._mongo_config()
        if not mongo_cfg.get("enabled"):
            logger.info("MongoDB not enabled — skipping backfill.")
            return
        from utils.MongoDB import FeatureStore
        logger.info(f"[{run_id}] MongoDB backfill: train/val bearings...")
        for bearing in self.registry.all_bearings():
            if bearing["role"] == "live":
                continue
            source_folder = self.registry.source_path(bearing)
            features_path = os.path.join(source_folder, "features.csv")
            if not os.path.exists(features_path):
                logger.warning(f"  [{bearing['name']}] No features.csv — skipping.")
                continue
            store = FeatureStore({
                "mongo_uri":       mongo_cfg["uri"],
                "db_name":         mongo_cfg["db_name"],
                "collection_name": "features",
                "dataset_id":      bearing["name"],
                "version":         run_id,
                "df_path":         features_path,
                "metadata": {
                    "bearing_name": bearing["name"],
                    "role":         bearing["role"],
                    "run_id":       run_id,
                    "source":       "backfill",
                },
            })
            store.run()
            logger.info(f"  [{bearing['name']}] ✓ ingested into MongoDB")

    def _run_validation(self, run_id: str):
        step    = self._validation_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "validation"

        def _validate(cfg):
            validator    = DataValidatorPHM(schema_path=cfg.get("schema_path"),
                                            log_path=cfg.get("log_path"))
            results_list = []
            for bearing in self.registry.val_bearings():
                csv_path = os.path.join(self.registry.source_path(bearing), "features.csv")
                if not os.path.exists(csv_path):
                    logger.warning(f"  No features.csv for {bearing['name']} — skipping.")
                    continue
                df = pd.read_csv(csv_path)
                df, result = validator.validate_features(df)
                results_list.append({bearing["name"]: result})
                logger.info(f"  Validated {bearing['name']}")
            validator.save_results(results_list, cfg["output_location"])
            return {"validation_results": cfg["output_location"]}

        self._execute(run_id, step_id, "validation", config, _validate)

    def _confirmed_fault_files(self) -> List[str]:
        """
        Return paths to features_confirmed.csv for every live bearing that has
        been confirmed by the maintenance tech.  These are included as extra
        training data alongside the normal train-role bearings.
        """
        confirmed = []
        for b in self.registry.live_bearings():
            if b.get("status") == "confirmed":
                path = os.path.join(self.registry.source_path(b), "features_confirmed.csv")
                if os.path.exists(path):
                    confirmed.append(path)
                    logger.info(f"  [confirmed fault] Including {path} in training data")
                else:
                    logger.warning(
                        f"  [{b['name']}] Status is 'confirmed' but "
                        f"features_confirmed.csv not found — skipping."
                    )
        return confirmed

    def _run_training(self, run_id: str):
        step    = self._training_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "training"

        # Base training files from train-role bearings
        train_files = [
            os.path.join(self.registry.source_path(b), "features.csv")
            for b in self.registry.train_bearings()
            if os.path.exists(os.path.join(self.registry.source_path(b), "features.csv"))
        ]

        # Add confirmed fault data from live bearings that the tech has validated.
        # This is the mechanism by which the system learns from real-world faults.
        confirmed_files = self._confirmed_fault_files()
        train_files.extend(confirmed_files)

        val_files = [
            os.path.join(self.registry.source_path(b), "features.csv")
            for b in self.registry.val_bearings()
            if os.path.exists(os.path.join(self.registry.source_path(b), "features.csv"))
        ]

        # Test files are used only for evaluation (mae_s metric) — never for training.
        # Without these, mae_s will be None and model comparison cannot work.
        test_files = [
            os.path.join(self.registry.source_path(b), "features.csv")
            for b in self.registry.all_bearings()
            if b["role"] == "test"
            and os.path.exists(os.path.join(self.registry.source_path(b), "features.csv"))
        ]

        if not train_files:
            logger.warning(f"[{run_id}] No training files found — skipping training.")
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        logger.info(
            f"[{run_id}] Training on {len(train_files)} train file(s) "
            f"({len(confirmed_files)} confirmed fault file(s) included), "
            f"{len(val_files)} val file(s), "
            f"{len(test_files)} test file(s) for evaluation."
        )
        config["train_files"] = train_files
        config["val_files"]   = val_files
        config["test_files"]  = test_files

        self._execute(run_id, step_id, "training", config,
                      lambda cfg: RULTrainerPHM(cfg).run(run_id=run_id))

    def _run_model_selection(self, run_id: str):
        """
        Select and deploy the best model trained in this run.
        If the training step was skipped (no new models for this run_id),
        this step is also skipped gracefully.
        """
        step    = self._model_selection_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "model_selection"

        # Check if training was skipped
        state = self.state_manager.load_state(run_id)
        training_status = state.get("steps", {}).get("training", {}).get("status")
        if training_status == "SKIPPED":
            logger.info(
                f"[{run_id}] Model selection skipped — training was skipped."
            )
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        from utils.model_registry import ModelRegistry
        models_this_run = ModelRegistry().list_models(run_id=run_id, status="pending")
        if not models_this_run:
            # No new models — check if any approved/deployed model already exists
            deployed = ModelRegistry().get_deployed_model("RUL_s")
            if deployed:
                logger.info(
                    f"[{run_id}] Model selection: no new models for this run, "
                    f"existing deployed model '{deployed['model_id']}' retained."
                )
                self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
                return
            else:
                # Genuinely nothing to work with
                logger.error(
                    f"[{run_id}] Model selection: no trained models and no deployed model."
                )
                self.state_manager.update_step_status(
                    run_id, step_id, "FAILED",
                    "No trained models found and no deployed model exists."
                )
                raise RuntimeError(
                    "No trained models found for this run and no deployed model exists. "
                    "Ensure training completed successfully."
                )

        self._execute(
            run_id, step_id, "model_selection", config,
            lambda cfg: {
                "best_model_id": self.select_best_model(
                    run_id, cfg.get("metric", "mae_s")
                )["model_id"]
            },
        )

    def select_best_model(self, run_id: str, metric: str = "mae_s") -> Dict:
        """
        Compare the best new model from this run against the currently deployed
        model on the chosen metric (lower is better for MAE).

        - No deployed model        → auto-approve and deploy (first run).
        - New model is better      → auto-approve and deploy, archive old.
        - New model is not better  → leave as pending for manual review.
        - Metric is None/missing   → treat as infinity so the model is never
                                     auto-deployed (forces manual review).
        """
        from utils.model_registry import ModelRegistry
        registry = ModelRegistry()

        # Candidates: pending models from this run
        models = registry.list_models(run_id=run_id, status="pending")
        if not models:
            raise RuntimeError(f"No pending models found for run_id={run_id}")

        best = min(models, key=lambda m: m.get("metrics", {}).get(metric, float("inf")))
        new_score = best.get("metrics", {}).get(metric)

        currently_deployed = registry.get_deployed_model("RUL_s")

        if currently_deployed is None:
            # No deployed model at all — always deploy regardless of metrics.
            # This covers first run AND cases where registry was wiped/reset.
            registry.approve_model(best["model_id"], approved_by="orchestrator_auto")
            registry.deploy_model(best["model_id"])
            logger.info(
                f"[{run_id}] No deployed model — auto-approved & deployed: "
                f"{best['model_id']} ({metric}={new_score})"
            )

        elif new_score is None:
            # New model has no metric — can't compare, leave as pending.
            # This should not block serving since a deployed model already exists.
            logger.warning(
                f"[{run_id}] New model '{best['model_id']}' has no {metric} metric — "
                f"left as PENDING for manual review. Existing deployed model retained."
            )

        else:
            old_score = currently_deployed.get("metrics", {}).get(metric)

            if old_score is None or new_score < old_score:
                # New model is better (or old deployed model had no metric) — deploy it
                registry.approve_model(best["model_id"], approved_by="orchestrator_auto")
                registry.deploy_model(best["model_id"])
                logger.info(
                    f"[{run_id}] New model is BETTER — auto-deployed: "
                    f"{best['model_id']} "
                    f"({metric}: {old_score} → {new_score})"
                )
            else:
                # New model is not better — leave as pending, keep current deployment
                logger.info(
                    f"[{run_id}] New model is NOT better — left as PENDING: "
                    f"{best['model_id']} "
                    f"({metric}: new={new_score} vs deployed={old_score})"
                )

        return best

    def _run_serving_pipeline(self, run_id: str):
        """
        Run the 4-stage Serving Pipeline burst-by-burst for the current live bearing.

        Each burst is ingested from disk, run through feature engineering → inference
        → predictive maintenance → monitoring, and written to MongoDB Serving History
        immediately. The dashboard polls Serving History and updates in real time.

        Stops early if:
          - PM status reaches 'critical' (RUL below critical threshold)
          - All acc_*.csv files have been read (bearing exhausted)
        """
        live_bearing = self.registry.current_live_bearing()
        if not live_bearing:
            logger.info(f"[{run_id}] Serving Pipeline skipped — no live bearing.")
            return

        step    = self._serving_pipeline_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "serving_pipeline"

        self.state_manager.update_step_status(run_id, step_id, "RUNNING")
        logger.info(f"  [{run_id}] Step '{step_id}' (serving_pipeline) — RUNNING")

        try:
            from serving_pipeline.serving_pipeline import ServingPipeline
            from scripts.data_ingestor import DataIngestorPHM
            from utils.model_registry import ModelRegistry

            model_entry = ModelRegistry().get_deployed_model("RUL_s")
            if not model_entry:
                raise RuntimeError(
                    "No deployed model found. Ensure training + model selection completed."
                )

            mongo_cfg = self._mongo_config()
            pipeline  = ServingPipeline(config={
                "window_size":            int(config.get("window_size", 40)),
                "critical_threshold_s":   int(config.get("critical_threshold_s", 3600)),
                "warning_threshold_s":    int(config.get("warning_threshold_s", 14400)),
                "baseline_path":          config.get("baseline_path",
                                                     "model_registry/monitoring_baseline.json"),
                "mongo_uri":              mongo_cfg.get("uri"),
                "db_name":                mongo_cfg.get("db_name"),
                "enable_serving_history": True,
            })
            pipeline.reset_bearing()

            source_folder = self.registry.source_path(live_bearing)
            ingestor      = DataIngestorPHM(config={
                "input_location":  source_folder,
                "output_location": source_folder,
            })

            burst_period = float(config.get("burst_period", 10.0))
            realtime     = bool(config.get("realtime", False))
            max_bursts   = config.get("max_bursts")

            n_total  = 0
            n_ready  = 0
            n_alerts = 0
            stop_reason = "exhausted"

            for burst in ingestor.stream_bursts(
                source_folder,
                burst_period=burst_period,
                realtime=realtime,
            ):
                if max_bursts is not None and burst["burst_idx"] >= max_bursts:
                    stop_reason = "max_bursts"
                    break

                # Run this single burst through all 4 pipeline stages.
                # run_burst() writes the result to MongoDB immediately.
                result = pipeline.run_burst(
                    run_id       = run_id,
                    bearing_name = live_bearing["name"],
                    burst_idx    = burst["burst_idx"],
                    h_signal     = burst["h_signal"],
                    v_signal     = burst["v_signal"],
                )

                n_total += 1
                if result.get("ready"):
                    n_ready  += 1
                    pm        = result.get("pm") or {}
                    pm_status = pm.get("status", "unknown")
                    rul_s     = pm.get("rul_s")
                    rul_min   = pm.get("rul_min")

                    if pm.get("alert"):
                        n_alerts += 1

                    logger.info(
                        f"  [{live_bearing['name']}] Burst {burst['burst_idx']:>5} "
                        f"| t={burst['time_s']:>8.0f}s "
                        f"| RUL={rul_s:>8.0f}s ({rul_min:.1f} min) "
                        f"| status={pm_status}"
                    )

                    # Stop early if critical — fault has occurred or is imminent
                    if pm_status == "critical":
                        stop_reason = "critical_alert"
                        logger.warning(
                            f"  [{live_bearing['name']}] CRITICAL threshold reached "
                            f"at burst {burst['burst_idx']} — stopping live stream."
                        )
                        break
                else:
                    logger.debug(
                        f"  [{live_bearing['name']}] Burst {burst['burst_idx']:>5} "
                        f"— warming up feature buffer"
                    )

            logger.info(
                f"  [{live_bearing['name']}] Stream ended | reason={stop_reason} "
                f"| bursts={n_total} ready={n_ready} alerts={n_alerts}"
            )

            outputs = {
                "bearing":      live_bearing["name"],
                "n_bursts":     n_total,
                "n_ready":      n_ready,
                "n_alerts":     n_alerts,
                "stop_reason":  stop_reason,
            }
            self.state_manager.update_step_status(run_id, step_id, "COMPLETE")
            self.state_manager.mark_step_outputs(run_id, step_id, outputs)
            logger.info(f"  [{run_id}] Step '{step_id}' — COMPLETE")

        except Exception as e:
            self.state_manager.update_step_status(run_id, step_id, "FAILED", str(e))
            logger.error(f"  [{run_id}] Step '{step_id}' — FAILED: {e}", exc_info=True)
            raise

    # ── Fault confirmation ────────────────────────────────────────────────────

    def confirm_fault_and_push_to_store(
        self,
        bearing_name: str,
        run_id: str,
        rul_at_failure: float,
    ) -> Dict:
        """
        Called by the API after the maintenance tech confirms a fault.

        1. Reads the bearing's features.csv
        2. Re-labels RUL from the confirmed failure point
        3. Pushes to MongoDB Feature Store under 'confirmed_faults' collection
        4. Sets bearing status to 'confirmed'
        """
        bearing = self.registry.get_bearing(bearing_name)
        if not bearing:
            raise ValueError(f"Bearing '{bearing_name}' not in registry.")

        source_folder  = self.registry.source_path(bearing)
        features_path  = os.path.join(source_folder, "features.csv")
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"No features.csv for {bearing_name} at {features_path}"
            )

        df = pd.read_csv(features_path)
        failure_s      = float(df["time_s"].max())
        df["RUL_s"]    = (failure_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / failure_s if failure_s > 0 else 0.0
        # Do NOT add extra columns to the CSV — it must have the exact same
        # schema as features.csv so the trainer's scaler doesn't get a column
        # count mismatch. The 'confirmed' flag lives only in MongoDB metadata.

        confirmed_path = os.path.join(source_folder, "features_confirmed.csv")
        df.to_csv(confirmed_path, index=False)

        mongo_cfg = self._mongo_config()
        if mongo_cfg.get("enabled"):
            from utils.MongoDB import FeatureStore
            store = FeatureStore({
                "mongo_uri":       mongo_cfg["uri"],
                "db_name":         mongo_cfg["db_name"],
                "collection_name": "confirmed_faults",
                "dataset_id":      bearing_name,
                "version":         run_id,
                "df_path":         confirmed_path,
                "metadata": {
                    "bearing_name":   bearing_name,
                    "role":           bearing["role"],
                    "run_id":         run_id,
                    "confirmed":      True,
                    "rul_at_failure": rul_at_failure,
                    "confirmed_at":   datetime.now().isoformat(),
                },
            })
            result = store.run()
            logger.info(
                f"[{bearing_name}] Confirmed fault features pushed to "
                f"MongoDB 'confirmed_faults'."
            )
        else:
            result = {"skipped": "MongoDB not enabled"}

        self.registry.set_status(bearing_name, "confirmed")
        return result