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
    6. Serving Pipeline — current live bearing (legacy path only)

SUBSEQUENT RUNS  (deployed model already exists):
    Phases 2b–5 are skipped. Serving is now handled by run_serving.py.

run_training_only() — called by run_preprod.py after fault confirmation:
    Retrains on confirmed_faults (FS Mirrored) + train-role bearings.
    Does NOT touch the live Feature Store or Serving Pipeline.
    Registers new model as PENDING — run_preprod.py handles promotion.

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
                all_workflows = yaml.safe_load(f)
            self.workflow_def = all_workflows["workflows"][workflow_name]

        self.registry = BearingRegistry(
            self.workflow_def.get("bearing_config", "config/bearings.json")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Full workflow (first run / legacy)
    # ─────────────────────────────────────────────────────────────────────────

    def start_workflow(
        self,
        workflow_name:    str = "rul_prediction",
        config_overrides: Dict = None,
    ) -> str:
        """
        Run the full workflow end-to-end.

        FIRST RUN (no deployed model):
            ingest → extract → backfill → validate → train → select → serve

        SUBSEQUENT RUNS (deployed model exists):
            ingest → extract only
            (Serving is handled externally by run_serving.py)
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state_manager.init_state(run_id, workflow_name)

        try:
            from utils.model_registry import ModelRegistry
            currently_deployed = ModelRegistry().get_deployed_model("RUL_s")
            live_bearing       = self.registry.current_live_bearing()

            # ── 1. Ingest live bearing ────────────────────────────────────────
            if live_bearing:
                logger.info(f"[{run_id}] Phase 1: Ingestion — {live_bearing['name']}")
                self._run_ingestion(run_id, live_bearing)

            # ── 2. Extract live bearing ───────────────────────────────────────
            if live_bearing:
                logger.info(f"[{run_id}] Phase 2: Feature extraction — {live_bearing['name']}")
                self._run_extraction(run_id, live_bearing)

            if not currently_deployed:
                # First run — train a model from scratch
                logger.info(f"[{run_id}] No deployed model found — running full pipeline.")

                # Also extract all train/val bearings
                for b in self.registry.train_bearings() + self.registry.val_bearings():
                    if b["name"] != (live_bearing["name"] if live_bearing else ""):
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
                    f"(deployed model already exists — serving handled by run_serving.py)"
                )

            # ── 6. Serving Pipeline ───────────────────────────────────────────────
            # Serving is handled by run_serving.py which is started automatically
            # by the API's ProcessManager after this workflow completes.
            # The legacy _run_serving_pipeline() is no longer called here.
            logger.info(
                f"[{run_id}] Phase 6: Serving Pipeline — "
                f"handled by run_serving.py (started by ProcessManager after workflow)"
            )

        except Exception as e:
            self.state_manager.mark_workflow_failed(run_id, traceback.format_exc())
            logger.error(f"[{run_id}] Workflow FAILED: {e}", exc_info=True)
            raise

        return run_id

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Pre-Production retraining (called by run_preprod.py)
    # ─────────────────────────────────────────────────────────────────────────

    def run_training_only(
        self,
        run_id:    str,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name:   str = "phm_mlops",
    ) -> str:
        """
        Pre-Production retraining path — called by run_preprod.py.

        Retrains ONLY using:
          - All train-role bearings (features.csv on disk)
          - Confirmed fault data from MongoDB 'confirmed_faults' collection
            (Feature Store Mirrored), restored to disk if not already present

        The live Feature Store and Serving Pipeline are NOT touched.
        The Serving Pipeline continues making predictions concurrently.

        Steps
        ─────
        1. Restore confirmed fault CSVs from MongoDB if not on disk
        2. Validate train + val feature files
        3. Train model
        4. Register new model as PENDING in ModelRegistry
           → run_preprod.py calls registry.compare_and_promote() to decide
             whether to write champion.json and hot-swap in run_serving.py

        Parameters
        ----------
        run_id    : str — unique identifier for this training run
        mongo_uri : str — MongoDB connection string
        db_name   : str — MongoDB database name

        Returns
        -------
        run_id : str
        """
        logger.info(f"[{run_id}] Pre-Production retraining started.")
        logger.info(f"[{run_id}] Data source: FS Mirrored (confirmed_faults) + train bearings.")
        logger.info(f"[{run_id}] Live Feature Store and Serving Pipeline are UNAFFECTED.")

        self.state_manager.init_state(run_id, "preprod_training")

        try:
            # 1. Restore confirmed fault CSVs from FS Mirrored (MongoDB)
            logger.info(f"[{run_id}] Phase 1: Restoring confirmed faults from FS Mirrored...")
            self._restore_confirmed_faults_from_mongo(run_id, mongo_uri, db_name)

            # 2. Validate feature files
            logger.info(f"[{run_id}] Phase 2: Validation")
            self._run_validation(run_id)

            # 3. Train
            logger.info(f"[{run_id}] Phase 3: Training")
            self._run_training(run_id)

            self.state_manager.mark_workflow_complete(run_id)
            logger.info(
                f"[{run_id}] Pre-Production retraining complete. "
                f"New model registered as PENDING — run_preprod.py will "
                f"compare and promote via registry.compare_and_promote()."
            )

        except Exception as e:
            self.state_manager.mark_workflow_failed(run_id, traceback.format_exc())
            logger.error(f"[{run_id}] Pre-Production retraining FAILED: {e}", exc_info=True)
            raise

        return run_id

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Fault confirmation
    # ─────────────────────────────────────────────────────────────────────────

    def confirm_fault_and_push_to_store(
        self,
        bearing_name:   str,
        run_id:         str,
        rul_at_failure: float = 0.0,
    ) -> Dict:
        """
        Called from the API after a maintenance worker confirms a fault.

        1. Reads the bearing's features.csv
        2. Re-labels RUL from the confirmed failure point
        3. Pushes labelled data to MongoDB 'confirmed_faults' (Feature Store Mirrored)
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
        # Do NOT add extra columns — must match features.csv schema exactly
        # so the trainer's scaler doesn't get a column count mismatch.
        # The 'confirmed' flag lives only in MongoDB metadata.

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
                f"MongoDB 'confirmed_faults' (Feature Store Mirrored)."
            )
        else:
            result = {"skipped": "MongoDB not enabled"}

        self.registry.set_status(bearing_name, "confirmed")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Per-phase helpers
    # ─────────────────────────────────────────────────────────────────────────

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
            validator    = DataValidatorPHM(
                schema_path=cfg.get("schema_path"),
                log_path=cfg.get("log_path"),
            )
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
        been confirmed by the maintenance tech. These are included as extra
        training data alongside the normal train-role bearings.
        """
        confirmed = []
        for b in self.registry.live_bearings():
            if b.get("status") == "confirmed":
                path = os.path.join(
                    self.registry.source_path(b), "features_confirmed.csv"
                )
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

        # Add confirmed fault data from live bearings validated by the tech
        confirmed_files = self._confirmed_fault_files()
        train_files.extend(confirmed_files)

        val_files = [
            os.path.join(self.registry.source_path(b), "features.csv")
            for b in self.registry.val_bearings()
            if os.path.exists(os.path.join(self.registry.source_path(b), "features.csv"))
        ]

        # Test files used only for evaluation (mae_s metric) — never for training
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
        If the training step was skipped, this step is also skipped gracefully.
        """
        step    = self._model_selection_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "model_selection"

        # Check if training was skipped
        state           = self.state_manager.load_state(run_id)
        training_status = state.get("steps", {}).get("training", {}).get("status")
        if training_status == "SKIPPED":
            logger.info(f"[{run_id}] Model selection skipped — training was skipped.")
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        from utils.model_registry import ModelRegistry
        models_this_run = ModelRegistry().list_models(run_id=run_id, status="pending")
        if not models_this_run:
            deployed = ModelRegistry().get_deployed_model("RUL_s")
            if deployed:
                logger.info(
                    f"[{run_id}] Model selection: no new models for this run, "
                    f"existing deployed model '{deployed['model_id']}' retained."
                )
                self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
                return
            else:
                logger.error(
                    f"[{run_id}] Model selection: no trained models and no deployed model."
                )
                self.state_manager.update_step_status(
                    run_id, step_id, "FAILED",
                    "No trained models found and no deployed model exists."
                )
                raise RuntimeError(
                    "No trained models found for this run and no deployed model exists."
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

        Uses ModelRegistry.compare_and_promote() which also writes champion.json
        so run_serving.py can hot-swap between bursts.
        """
        from utils.model_registry import ModelRegistry
        registry = ModelRegistry()
        result   = registry.compare_and_promote(run_id=run_id, metric=metric)

        if not result["model_id"]:
            raise RuntimeError(f"No pending models found for run_id={run_id}")

        logger.info(
            f"[{run_id}] Model selection: {result['reason']}"
        )

        # Return a dict with model_id so the orchestrator state can record it
        return {"model_id": result["model_id"], "promoted": result["promoted"]}

    def _run_serving_pipeline(self, run_id: str):
        """
        Legacy serving pipeline path — used on the first run and when
        triggered via the old /workflow/trigger endpoint.

        In the new architecture, serving is handled by run_serving.py which
        polls live_features independently. This method is kept for backward
        compatibility only.
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

            pipeline = ServingPipeline(config={
                "mongo_uri":              self._mongo_config().get("uri"),
                "db_name":                self._mongo_config().get("db_name"),
                "window_size":            int(config.get("window_size", 40)),
                "critical_threshold_s":   int(config.get("critical_threshold_s", 3600)),
                "warning_threshold_s":    int(config.get("warning_threshold_s", 14400)),
                "baseline_path":          config.get("baseline_path",
                    "model_registry/monitoring_baseline.json"),
                "enable_serving_history": True,
            })

            source_folder = self.registry.source_path(live_bearing)
            results = pipeline.run_bearing(
                run_id        = run_id,
                bearing_name  = live_bearing["name"],
                source_folder = source_folder,
                burst_period  = float(config.get("burst_period", 10.0)),
                realtime      = bool(config.get("realtime", False)),
                max_bursts    = config.get("max_bursts"),
            )

            self.state_manager.update_step_status(run_id, step_id, "COMPLETE")
            self.state_manager.mark_step_outputs(run_id, step_id, {
                "bearing":       live_bearing["name"],
                "n_predictions": len(results),
            })
            logger.info(
                f"  [{run_id}] Serving pipeline complete — "
                f"{len(results)} bursts processed."
            )

        except Exception as e:
            self.state_manager.update_step_status(run_id, step_id, "FAILED", str(e))
            logger.error(f"  [{run_id}] Serving pipeline FAILED: {e}", exc_info=True)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Restore confirmed faults from MongoDB to disk
    # ─────────────────────────────────────────────────────────────────────────

    def _restore_confirmed_faults_from_mongo(
        self,
        run_id:    str,
        mongo_uri: str,
        db_name:   str,
    ) -> None:
        """
        Pull confirmed fault records from MongoDB 'confirmed_faults' collection
        and write them to disk as features_confirmed.csv so the trainer can
        include them in the training set.

        This is the read path from the Feature Store Mirrored (FS Mirrored).
        The FS Mirrored is populated by confirm_fault_and_push_to_store() when
        a maintenance worker confirms a fault on the dashboard.

        If features_confirmed.csv already exists on disk for a bearing, the
        restore is skipped (idempotent).
        """
        try:
            from pymongo import MongoClient

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            db     = client[db_name]
            col    = db["confirmed_faults"]

            bearings_with_faults = col.distinct("dataset_id")
            if not bearings_with_faults:
                logger.info(
                    f"[{run_id}] No confirmed faults in MongoDB — "
                    f"training on base train set only."
                )
                return

            for bearing_name in bearings_with_faults:
                bearing = self.registry.get_bearing(bearing_name)
                if not bearing:
                    logger.warning(
                        f"[{run_id}] Confirmed fault bearing '{bearing_name}' "
                        f"not found in registry — skipping."
                    )
                    continue

                source_folder  = self.registry.source_path(bearing)
                confirmed_path = os.path.join(source_folder, "features_confirmed.csv")

                if os.path.exists(confirmed_path):
                    logger.info(
                        f"[{run_id}] [{bearing_name}] features_confirmed.csv "
                        f"already on disk — skipping restore."
                    )
                    continue

                # Pull all records for this bearing from MongoDB
                records = list(col.find({"dataset_id": bearing_name}))
                if not records:
                    continue

                # Strip MongoDB internal fields before writing to CSV
                for r in records:
                    r.pop("_id",      None)
                    r.pop("dataset_id", None)
                    r.pop("version",  None)
                    r.pop("metadata", None)

                df = pd.DataFrame(records)
                os.makedirs(source_folder, exist_ok=True)
                df.to_csv(confirmed_path, index=False)
                logger.info(
                    f"[{run_id}] [{bearing_name}] Restored {len(df)} confirmed "
                    f"fault rows from FS Mirrored → {confirmed_path}"
                )

        except Exception as e:
            logger.warning(
                f"[{run_id}] Could not restore confirmed faults from MongoDB: {e}. "
                f"Training will proceed with base train set only."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Workflow step templates (read from workflow.yaml)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_step(self, step_type: str) -> Dict:
        for step in self.workflow_def.get("steps", []):
            if step["type"] == step_type:
                return step
        raise KeyError(f"No step of type '{step_type}' in workflow definition.")

    def _ingestion_template(self) -> Dict:
        return self._get_step("ingestion")["config"]

    def _extraction_template(self) -> Dict:
        return self._get_step("feature_engineering")["config"]

    def _validation_step(self) -> Dict:
        return self._get_step("validation")

    def _training_step(self) -> Dict:
        return self._get_step("training")

    def _model_selection_step(self) -> Dict:
        return self._get_step("model_selection")

    def _serving_pipeline_step(self) -> Dict:
        return self._get_step("serving_pipeline")

    def _mongo_config(self) -> Dict:
        return self.workflow_def.get("mongodb", {"enabled": False})

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Generic step executor
    # ─────────────────────────────────────────────────────────────────────────

    def _execute(
        self,
        run_id:    str,
        step_id:   str,
        step_type: str,
        config:    Dict,
        fn,
    ):
        """
        Generic step executor with state tracking.
        Calls fn(config), catches exceptions, updates WorkflowStateManager.
        """
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
            logger.error(
                f"  [{run_id}] Step '{step_id}' FAILED: {e}", exc_info=True
            )
            raise