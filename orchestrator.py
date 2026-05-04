"""
orchestrator.py
═══════════════════════════════════════════════════════════════════════════════
Workflow Orchestrator — group-per-condition edition.

Bearing groups
──────────────
Bearings are split into 3 groups by the "group" field in bearings.json:
    group "1" → all Bearing1_x  (condition 1 operating environment)
    group "2" → all Bearing2_x  (condition 2 operating environment)
    group "3" → all Bearing3_x  (condition 3 operating environment)

Each group trains its own model and writes its own champion file:
    model_registry/champion_bearing1.json
    model_registry/champion_bearing2.json
    model_registry/champion_bearing3.json

start_workflow() behaviour
──────────────────────────
FIRST RUN (no champion files exist for any group):
    Trains all 3 group models IN PARALLEL (one thread per group).
    Each group: validate → train (group data only) → select & write champion.

SUBSEQUENT RUNS (champion files already exist):
    All training skipped — serving continues via run_serving.py.

run_training_only(group) — called by run_preprod.py after fault confirmation:
    Retrains ONLY the specified group's model.
    Reads group-specific train bearings + confirmed faults for that group.
    Registers new model as PENDING — run_preprod.py handles champion promotion.
    Other groups' models and serving pipelines are completely unaffected.

Fault confirmation flow:
    confirm_fault_and_push_to_store() → re-labels features → pushes to MongoDB
    The group is derived automatically from the bearing's "group" field.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import yaml
import logging
import traceback
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from scripts.data_validator import DataValidatorPHM
from models.model_trainer import RULTrainerPHM
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY_DIR = "model_registry"


# ─────────────────────────────────────────────────────────────────────────────
# BEARING REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class BearingRegistry:
    """
    Loads config/bearings.json and tracks bearing state.

    Group structure
    ───────────────
    Each bearing has a "group" field ("1", "2", or "3") matching its
    condition number. The config also has a top-level "groups" dict with
    one entry per group containing:
        live_bearing_queue  : ordered list of live bearing names
        current_live_index  : current position in that queue
        champion_file       : path to this group's champion JSON file

    roles
    ─────
    train  — run-to-failure bearings used for model training (one per group)
    val    — validation bearings used during training evaluation (one per group)
    live   — bearings queued for live inference, processed one at a time per group

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
        self.base_path = data["base_path"]
        self.bearings  = data["bearings"]
        self.groups    = data.get("groups", {})

        # Backward compat: if old flat queue exists alongside new groups, ignore it
        # The new structure keeps everything inside self.groups

    def _save(self):
        with open(self.config_path, "r") as f:
            data = json.load(f)
        data["bearings"] = self.bearings
        data["groups"]   = self.groups
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Group helpers ─────────────────────────────────────────────────────────

    def all_groups(self) -> List[str]:
        """Return sorted list of group IDs e.g. ["1", "2", "3"]."""
        return sorted(self.groups.keys())

    def group_of(self, bearing_name: str) -> Optional[str]:
        """Return the group ID for a bearing name, or None if not found."""
        b = self.get_bearing(bearing_name)
        return b.get("group") if b else None

    def champion_path(self, group: str) -> str:
        """Return the champion file path for a group."""
        return self.groups[group]["champion_file"]

    def bearings_in_group(self, group: str) -> List[Dict]:
        """Return all bearings belonging to the given group."""
        return [b for b in self.bearings if b.get("group") == group]

    def train_bearings_in_group(self, group: str) -> List[Dict]:
        return [b for b in self.bearings
                if b.get("group") == group and b["role"] == "train"]

    def val_bearings_in_group(self, group: str) -> List[Dict]:
        return [b for b in self.bearings
                if b.get("group") == group and b["role"] == "val"]

    def live_bearings_in_group(self, group: str) -> List[Dict]:
        return [b for b in self.bearings
                if b.get("group") == group and b["role"] == "live"]

    # ── Queue helpers (per group) ─────────────────────────────────────────────

    def current_live_bearing(self, group: str) -> Optional[Dict]:
        """Return the bearing currently at the head of a group's live queue."""
        grp  = self.groups.get(group, {})
        q    = grp.get("live_bearing_queue", [])
        idx  = grp.get("current_live_index", 0)
        if not q or idx >= len(q):
            return None
        return self.get_bearing(q[idx])

    def advance_live_bearing(self, group: str) -> Optional[Dict]:
        """Increment a group's queue pointer and persist to disk."""
        self.groups[group]["current_live_index"] += 1
        self._save()
        next_b = self.current_live_bearing(group)
        if next_b:
            logger.info(f"[Group {group}] Queue advanced → now serving: {next_b['name']}")
        else:
            logger.info(f"[Group {group}] Live queue exhausted.")
        return next_b

    def current_live_bearings_all_groups(self) -> Dict[str, Optional[Dict]]:
        """Return {group: current_live_bearing} for all groups."""
        return {g: self.current_live_bearing(g) for g in self.all_groups()}

    # ── Generic bearing queries ───────────────────────────────────────────────

    def all_bearings(self) -> List[Dict]:
        return self.bearings

    def train_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "train"]

    def val_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "val"]

    def live_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "live"]

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

    def source_path(self, bearing: Dict) -> str:
        return os.path.join(self.base_path, bearing["name"])

    def is_test(self, bearing: Dict) -> bool:
        return bearing["role"] != "train"

    def print_status(self):
        from collections import Counter
        counts = Counter(b.get("status", "unknown") for b in self.bearings)
        logger.info("── Bearing registry status ──────────────────────")
        for b in self.bearings:
            exists = "exists" if os.path.isdir(self.source_path(b)) else "NO FOLDER"
            logger.info(
                f"  {b['name']:<14} group={b.get('group','?')} role={b['role']:<6} "
                f"status={b.get('status','?'):<12} [{exists}]"
            )
        logger.info(f"  Status counts: {dict(counts)}")
        logger.info("────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowStateManager:
    STATE_DIR = "workflow_state"

    def _state_path(self, run_id: str) -> str:
        return os.path.join(self.STATE_DIR, f"{run_id}.json")

    def init_state(self, run_id: str, workflow_name: str):
        state = {
            "run_id":     run_id,
            "workflow":   workflow_name,
            "status":     "RUNNING",
            "start_time": datetime.now().isoformat(),
            "end_time":   None,
            "steps":      {},
        }
        self._write(run_id, state)

    def load_state(self, run_id: str) -> Optional[Dict]:
        path = self._state_path(run_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def update_step_status(
        self, run_id: str, step_id: str, status: str, error: str = None
    ):
        state = self.load_state(run_id) or {"steps": {}}
        state.setdefault("steps", {}).setdefault(step_id, {})["status"] = status
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
    # PUBLIC: Full workflow — trains all 3 groups in parallel on first run
    # ─────────────────────────────────────────────────────────────────────────

    def start_workflow(
        self,
        workflow_name:    str = "rul_prediction",
        config_overrides: Dict = None,
    ) -> str:
        """
        Run the training workflow.

        FIRST RUN (no champion files exist for any group):
            Trains all 3 groups IN PARALLEL. Each group runs independently:
                validate → train (group-specific data) → select & write champion

        SUBSEQUENT RUNS (all champion files already exist):
            Training skipped entirely. Serving continues via run_serving.py.

        If SOME groups have champions and others don't, only the missing ones
        are trained — allowing recovery from a partial first run.
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state_manager.init_state(run_id, workflow_name)

        try:
            groups_needing_training = []
            for group in self.registry.all_groups():
                champion_path = self.registry.champion_path(group)
                if not os.path.exists(champion_path):
                    groups_needing_training.append(group)
                else:
                    logger.info(
                        f"[{run_id}] Group {group}: champion already exists "
                        f"({champion_path}) — skipping training."
                    )

            if not groups_needing_training:
                logger.info(
                    f"[{run_id}] All group champions exist — skipping training. "
                    f"Serving is handled by run_serving.py."
                )
                self.state_manager.mark_workflow_complete(run_id)
                return run_id

            logger.info(
                f"[{run_id}] Training groups in parallel: {groups_needing_training}"
            )

            # Train all groups that need it concurrently
            errors: Dict[str, str] = {}
            threads = []
            for group in groups_needing_training:
                t = threading.Thread(
                    target=self._train_group_thread,
                    args=(run_id, group, errors),
                    name=f"train-group-{group}",
                    daemon=True,
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            if errors:
                error_summary = "; ".join(
                    f"Group {g}: {e}" for g, e in errors.items()
                )
                self.state_manager.mark_workflow_failed(run_id, error_summary)
                raise RuntimeError(f"Training failed for groups: {error_summary}")

            self.state_manager.mark_workflow_complete(run_id)
            logger.info(
                f"[{run_id}] All groups trained successfully. "
                f"Serving Pipeline is handled by run_serving.py."
            )

        except Exception as e:
            self.state_manager.mark_workflow_failed(run_id, traceback.format_exc())
            logger.error(f"[{run_id}] Workflow FAILED: {e}", exc_info=True)
            raise

        return run_id

    def _train_group_thread(
        self, run_id: str, group: str, errors: Dict[str, str]
    ):
        """Thread target — train one group's model. Writes errors dict on failure."""
        group_run_id = f"{run_id}_g{group}"
        try:
            logger.info(f"[Group {group}] Starting training (run_id={group_run_id})")
            self._run_group_training(group_run_id, group)
            logger.info(f"[Group {group}] Training complete.")
        except Exception as e:
            errors[group] = str(e)
            logger.error(f"[Group {group}] Training FAILED: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Pre-Production retraining for a single group
    # ─────────────────────────────────────────────────────────────────────────

    def run_training_only(
        self,
        run_id:    str,
        group:     str,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name:   str = "phm_mlops",
    ) -> str:
        """
        Pre-Production retraining for a specific bearing group.

        Called by run_preprod.py after a fault is confirmed on a bearing
        belonging to this group. Only this group's model is retrained —
        other groups' models and serving pipelines are completely unaffected.

        Retrains using:
          - train-role bearings in this group (from MongoDB factory_features)
          - confirmed fault data for this group (from feature_store_mirrored)

        Registers the new model as PENDING — run_preprod.py calls
        registry.compare_and_promote() to decide whether to write the
        group champion file.

        Parameters
        ----------
        run_id    : str — unique identifier for this training run
        group     : str — group ID to retrain ("1", "2", or "3")
        mongo_uri : str — MongoDB connection string
        db_name   : str — MongoDB database name
        """
        logger.info(f"[{run_id}] Pre-Production retraining — Group {group}.")
        logger.info(f"[{run_id}] Data: factory_features + feature_store_mirrored (Group {group} only).")
        logger.info(f"[{run_id}] Other groups are UNAFFECTED — their serving continues.")

        self.state_manager.init_state(run_id, f"preprod_group{group}")

        try:
            logger.info(f"[{run_id}] Phase 1: Validation (Group {group})")
            self._run_validation(run_id, group=group)

            logger.info(f"[{run_id}] Phase 2: Training (Group {group})")
            self._run_training(run_id, group=group)

            self.state_manager.mark_workflow_complete(run_id)
            logger.info(
                f"[{run_id}] Pre-Production retraining complete for Group {group}. "
                f"New model registered as PENDING — run_preprod.py will "
                f"compare and promote via registry.compare_and_promote()."
            )

        except Exception as e:
            self.state_manager.mark_workflow_failed(run_id, traceback.format_exc())
            logger.error(
                f"[{run_id}] Pre-Production FAILED (Group {group}): {e}", exc_info=True
            )
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

        1. Determines which group this bearing belongs to
        2. Reads the bearing's features.csv
        3. Re-labels RUL from the confirmed failure point
        4. Pushes labelled data to MongoDB feature_store_mirrored
           (tagged with the bearing's group for scoped retraining)
        5. Sets bearing status to 'confirmed'

        The group tag in MongoDB means run_training_only(group) will only
        pick up confirmed data from the relevant group, keeping groups isolated.
        """
        bearing = self.registry.get_bearing(bearing_name)
        if not bearing:
            raise ValueError(f"Bearing '{bearing_name}' not in registry.")

        group = bearing.get("group")
        if not group:
            raise ValueError(
                f"Bearing '{bearing_name}' has no 'group' field in bearings.json."
            )

        source_folder = self.registry.source_path(bearing)
        features_path = os.path.join(source_folder, "features.csv")

        if os.path.exists(features_path):
            # Train/val bearing — features.csv exists on disk
            df = pd.read_csv(features_path)
        else:
            # Live bearing — features were streamed via SCADA and live in
            # the feature_store collection in MongoDB, not on disk.
            # Reconstruct a features DataFrame from the SCADA burst documents.
            logger.info(
                f"[{bearing_name}] features.csv not on disk — "
                f"reading live SCADA bursts from MongoDB feature_store."
            )
            mongo_cfg = self._mongo_config()
            if not mongo_cfg.get("enabled"):
                raise FileNotFoundError(
                    f"No features.csv for {bearing_name} at {features_path} "
                    f"and MongoDB is not enabled — cannot reconstruct features."
                )
            from pymongo import MongoClient
            from utils.db_collections import COL_FEATURE_STORE
            client = MongoClient(mongo_cfg["uri"], serverSelectionTimeoutMS=5000)
            try:
                col  = client[mongo_cfg["db_name"]][COL_FEATURE_STORE]
                docs = list(col.find(
                    {"bearing_name": bearing_name, "session_end": {"$exists": False}},
                    sort=[("burst_idx", 1)],
                ))
            finally:
                client.close()

            if not docs:
                raise FileNotFoundError(
                    f"No features.csv on disk and no SCADA bursts found in "
                    f"MongoDB feature_store for {bearing_name}. "
                    f"Ensure the bearing was served before confirming."
                )

            # Features are stored flat on the document (18 values written by
            # run_serving.py when marking consumed). Exclude MongoDB/metadata
            # fields to get only the feature columns.
            _exclude = {
                "_id", "bearing_name", "burst_idx", "time_s", "sent_at",
                "consumed", "consumed_at", "session_end",
            }
            rows = []
            for doc in docs:
                row = {"burst_idx": doc["burst_idx"], "time_s": doc["time_s"]}
                row.update({k: v for k, v in doc.items() if k not in _exclude})
                rows.append(row)
            df = pd.DataFrame(rows)
            logger.info(
                f"[{bearing_name}] Reconstructed {len(df)} bursts "
                f"from MongoDB feature_store."
            )
        failure_s      = float(df["time_s"].max())
        df["RUL_s"]    = (failure_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / failure_s if failure_s > 0 else 0.0

        confirmed_path = os.path.join(source_folder, "features_confirmed.csv")
        df.to_csv(confirmed_path, index=False)

        mongo_cfg = self._mongo_config()
        if mongo_cfg.get("enabled"):
            from utils.MongoDB import FeatureStore
            from utils.db_collections import COL_FEATURE_STORE_MIRRORED
            store = FeatureStore({
                "mongo_uri":       mongo_cfg["uri"],
                "db_name":         mongo_cfg["db_name"],
                "collection_name": COL_FEATURE_STORE_MIRRORED,
                "dataset_id":      bearing_name,
                "version":         run_id,
                "df_path":         confirmed_path,
                "metadata": {
                    "bearing_name":   bearing_name,
                    "group":          group,
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
                f"MongoDB feature_store_mirrored (Group {group})."
            )
        else:
            result = {"skipped": "MongoDB not enabled"}

        self.registry.set_status(bearing_name, "confirmed")
        logger.info(
            f"[{bearing_name}] Status → confirmed. "
            f"Group {group} retraining can now be triggered via run_preprod.py."
        )
        return {**result, "group": group, "bearing": bearing_name}

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Group-level training pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _run_group_training(self, run_id: str, group: str):
        """
        Run the full validate → train → select pipeline for one group.
        Called both from start_workflow() (parallel threads) and directly
        from _train_group_thread().
        """
        self.state_manager.init_state(run_id, f"group_{group}_training")

        # 1. Validate
        self._run_validation(run_id, group=group)

        # 2. Train
        self._run_training(run_id, group=group)

        # 3. Select & write group champion
        self._run_model_selection(run_id, group=group)

        self.state_manager.mark_workflow_complete(run_id)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Validation (group-scoped)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(self, run_id: str, group: str = None):
        """
        Validate val bearings. If group is specified, only validates val
        bearings from that group. Otherwise validates all val bearings.
        """
        step    = self._validation_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = f"validation_g{group}" if group else "validation"

        val_bearings = (
            self.registry.val_bearings_in_group(group)
            if group else self.registry.val_bearings()
        )

        def _validate(cfg):
            validator    = DataValidatorPHM(
                schema_path=cfg.get("schema_path"),
                log_path=cfg.get("log_path"),
            )
            results_list = []
            for bearing in val_bearings:
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

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Training (group-scoped)
    # ─────────────────────────────────────────────────────────────────────────

    def _confirmed_fault_dataframes(self, group: str = None) -> List[pd.DataFrame]:
        """
        Return DataFrames for confirmed live bearings from feature_store_mirrored.
        If group is specified, only returns confirmed data for that group.
        """
        mongo_cfg = self._mongo_config()
        if not mongo_cfg.get("enabled"):
            return []

        from pymongo import MongoClient
        from utils.db_collections import COL_FEATURE_STORE_MIRRORED

        confirmed_dfs = []
        try:
            client = MongoClient(mongo_cfg["uri"], serverSelectionTimeoutMS=5000)
            db     = client[mongo_cfg["db_name"]]
            col    = db[COL_FEATURE_STORE_MIRRORED]

            # Filter by group if specified — this is the key isolation mechanism
            query_filter = {}
            if group:
                query_filter["metadata.group"] = group

            bearings_to_check = (
                self.registry.live_bearings_in_group(group)
                if group else self.registry.live_bearings()
            )

            for b in bearings_to_check:
                if b.get("status") != "confirmed":
                    continue
                docs = list(col.find({"dataset_id": b["name"]}))
                if not docs:
                    logger.warning(
                        f"  [{b['name']}] Status 'confirmed' but no docs in "
                        f"feature_store_mirrored — skipping."
                    )
                    continue
                df = pd.DataFrame(docs).drop(
                    columns=["_id", "version", "metadata"], errors="ignore"
                )

                # Keep only the columns that match features.csv schema.
                # The flat MongoDB documents contain extra fields
                # (bearing_name, sent_at, consumed, consumed_at etc.) that
                # must be excluded before passing to the trainer — otherwise
                # _add_rolling_features() produces more columns than val data,
                # causing the scaler shape mismatch.
                _FEATURE_COLS = [
                    "file_id", "burst_idx", "time_s",
                    "h_max", "h_min", "h_mean", "h_sd", "h_rms",
                    "h_skew", "h_kurt", "h_crest", "h_form",
                    "v_max", "v_min", "v_mean", "v_sd", "v_rms",
                    "v_skew", "v_kurt", "v_crest", "v_form",
                    "RUL_s", "RUL_norm",
                ]
                keep = [c for c in _FEATURE_COLS if c in df.columns]
                df   = df[keep]
                confirmed_dfs.append(df)
                logger.info(
                    f"  [{b['name']}] Loaded {len(df)} confirmed fault rows "
                    f"(Group {group or 'all'})"
                )
            client.close()
        except Exception as e:
            logger.warning(
                f"  Could not load confirmed fault DataFrames: {e}. "
                f"Training will proceed with base train set only."
            )

        return confirmed_dfs

    def _run_training(self, run_id: str, group: str = None):
        """
        Load training/validation DataFrames from MongoDB and run the RUL trainer.

        If group is specified, only loads bearings from that group — this is
        what keeps each group's model trained on its own operating condition data.

        Data sources
        ────────────
        Train : factory_features (train-role bearings for this group)
              + feature_store_mirrored (confirmed faults for this group)
        Val   : factory_features (val-role bearings for this group)
        Test  : factory_features (test-role bearings for this group)
        """
        step    = self._training_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = f"training_g{group}" if group else "training"

        mongo_cfg = self._mongo_config()
        if not mongo_cfg.get("enabled"):
            logger.error(
                f"[{run_id}] MongoDB not enabled — cannot load training data."
            )
            self.state_manager.update_step_status(
                run_id, step_id, "FAILED", "MongoDB not enabled"
            )
            return

        from pymongo import MongoClient
        from utils.db_collections import COL_FACTORY_FEATURES

        client = MongoClient(mongo_cfg["uri"], serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
        except Exception as e:
            logger.error(f"[{run_id}] Cannot connect to MongoDB: {e}")
            self.state_manager.update_step_status(
                run_id, step_id, "FAILED", f"MongoDB connection failed: {e}"
            )
            return

        db = client[mongo_cfg["db_name"]]

        # ── Select which bearings to load ─────────────────────────────────────
        train_bearings = (
            self.registry.train_bearings_in_group(group)
            if group else self.registry.train_bearings()
        )

        # Val bearings are shared across ALL groups — every group validates
        # against the full set of val bearings regardless of which group
        # those bearings belong to.
        val_bearings = self.registry.val_bearings()
        all_for_test = (
            self.registry.bearings_in_group(group)
            if group else self.registry.all_bearings()
        )

        # ── Load train DataFrames ─────────────────────────────────────────────
        train_dfs: List[pd.DataFrame] = []
        for b in train_bearings:
            docs = list(db[COL_FACTORY_FEATURES].find({"dataset_id": b["name"]}))
            if not docs:
                logger.warning(
                    f"  [{b['name']}] No docs in factory_features — "
                    f"run seed_historical_data.py first."
                )
                continue
            df = pd.DataFrame(docs).drop(
                columns=["_id", "version", "metadata"], errors="ignore"
            )
            train_dfs.append(df)
            logger.info(
                f"  [{b['name']}] Loaded {len(df)} train rows (Group {group or 'all'})"
            )

        # ── Add confirmed fault DataFrames (group-scoped) ─────────────────────
        confirmed_dfs = self._confirmed_fault_dataframes(group=group)
        train_dfs.extend(confirmed_dfs)

        # ── Load val DataFrames ───────────────────────────────────────────────
        val_dfs: List[pd.DataFrame] = []
        for b in val_bearings:
            docs = list(db[COL_FACTORY_FEATURES].find({"dataset_id": b["name"]}))
            if not docs:
                logger.warning(
                    f"  [{b['name']}] No val docs in factory_features — skipping."
                )
                continue
            df = pd.DataFrame(docs).drop(
                columns=["_id", "version", "metadata"], errors="ignore"
            )
            val_dfs.append(df)
            logger.info(f"  [{b['name']}] Loaded {len(df)} val rows")

        # ── Load test DataFrames ──────────────────────────────────────────────
        test_dfs: List[Tuple[str, pd.DataFrame]] = []
        for b in all_for_test:
            if b["role"] != "test":
                continue
            docs = list(db[COL_FACTORY_FEATURES].find({"dataset_id": b["name"]}))
            if not docs:
                continue
            df = pd.DataFrame(docs).drop(
                columns=["_id", "version", "metadata"], errors="ignore"
            )
            test_dfs.append((b["name"], df))

        client.close()

        if not train_dfs:
            logger.warning(
                f"[{run_id}] No training data for Group {group or 'all'} — skipping. "
                f"Ensure seed_historical_data.py has been run."
            )
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        logger.info(
            f"[{run_id}] Group {group or 'all'}: training on {len(train_dfs)} "
            f"train bearing(s) ({len(confirmed_dfs)} confirmed fault), "
            f"{len(val_dfs)} val, {len(test_dfs)} test — all from MongoDB."
        )

        config["train_dataframes"] = train_dfs
        config["val_dataframes"]   = val_dfs
        config["test_dataframes"]  = test_dfs
        config.pop("train_files", None)
        config.pop("val_files",   None)
        config.pop("test_files",  None)

        self._execute(run_id, step_id, "training", config,
                      lambda cfg: RULTrainerPHM(cfg).run(run_id=run_id))

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Model selection (group-scoped champion)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_model_selection(self, run_id: str, group: str = None):
        """
        Select and deploy the best model for this training run.
        Writes the group-specific champion file if a better model is found.
        """
        step    = self._model_selection_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = f"model_selection_g{group}" if group else "model_selection"

        state           = self.state_manager.load_state(run_id)
        training_step   = f"training_g{group}" if group else "training"
        training_status = state.get("steps", {}).get(training_step, {}).get("status")
        if training_status == "SKIPPED":
            logger.info(f"[{run_id}] Model selection skipped — training was skipped.")
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        from utils.model_registry import ModelRegistry
        models_this_run = ModelRegistry().list_models(run_id=run_id, status="pending")
        if not models_this_run:
            champion_path = (
                self.registry.champion_path(group) if group
                else os.path.join(MODEL_REGISTRY_DIR, "champion.json")
            )
            if os.path.exists(champion_path):
                logger.info(
                    f"[{run_id}] No new models for this run — "
                    f"existing champion retained ({champion_path})."
                )
                self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
                return
            else:
                raise RuntimeError(
                    f"No trained models found and no champion at {champion_path}."
                )

        # Champion path for this group
        champion_path = (
            self.registry.champion_path(group) if group
            else os.path.join(MODEL_REGISTRY_DIR, "champion.json")
        )

        self._execute(
            run_id, step_id, "model_selection", config,
            lambda cfg: {
                "best_model_id": self.select_best_model(
                    run_id,
                    cfg.get("metric", "mae_s"),
                    champion_path=champion_path,
                )["model_id"]
            },
        )

    def select_best_model(
        self, run_id: str, metric: str = "mae_s", champion_path: str = None
    ) -> Dict:
        """
        Compare the best new model from this run against the current champion
        and promote if better.

        When champion_path is provided (group-specific training), comparison
        is done against that group's own champion file — NOT the shared
        deployed model in the registry. This avoids the race condition where
        3 parallel group training threads overwrite each other's promotion.

        When champion_path is None (single-group or legacy path), falls back
        to the standard compare_and_promote() behaviour.
        """
        import json, os
        from datetime import datetime, timezone
        from utils.model_registry import ModelRegistry

        registry = ModelRegistry()

        if not champion_path:
            # Legacy / single-group path — use registry's built-in comparison
            result = registry.compare_and_promote(run_id=run_id, metric=metric)
            logger.info(
                f"[{run_id}] Model selection: promoted={result['promoted']}  "
                f"model_id={result['model_id']}  reason={result['reason']}"
            )
            return result

        # ── Group-specific path — compare against this group's champion file ──
        pending = registry.list_models(run_id=run_id, status="pending")
        if not pending:
            msg = f"No pending models for run_id='{run_id}'."
            logger.warning(f"[{run_id}] {msg}")
            return {"promoted": False, "model_id": None, "reason": msg}

        # Pick best pending model by metric (lower is better)
        best       = min(
            pending,
            key=lambda m: m.get("metrics", {}).get(metric, float("inf"))
        )
        new_score   = best.get("metrics", {}).get(metric)
        new_metrics = best.get("metrics", {})
        model_id    = best["model_id"]
        model_path  = best["model_path"]

        # Read this group's current champion (not the shared registry deployed model)
        old_score   = None
        old_metrics = {}
        if os.path.exists(champion_path):
            try:
                with open(champion_path) as f:
                    current = json.load(f)
                old_score   = current.get("metrics", {}).get(metric)
                old_metrics = current.get("metrics", {})
            except Exception:
                pass

        # Decision
        if old_score is None:
            reason  = "No existing group champion — promoting new model."
            promote = True
        elif new_score is None:
            reason  = f"New model has no '{metric}' — left as PENDING."
            promote = False
        elif new_score < old_score:
            reason  = f"New model BETTER: {metric} {old_score:.2f} → {new_score:.2f}."
            promote = True
        else:
            reason  = f"New model NOT better: {new_score:.2f} >= {old_score:.2f}."
            promote = False

        logger.info(f"[{run_id}] Group champion comparison: {reason}")

        if promote:
            # Approve + deploy in registry (for audit trail)
            registry.approve_model(model_id, approved_by="auto_group_promote")
            registry.deploy_model(model_id)

            # Write group-specific champion file atomically
            champion = {
                "model_id":    model_id,
                "model_path":  model_path,
                "metrics":     new_metrics,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(champion_path), exist_ok=True)
            tmp = champion_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(champion, f, indent=2)
            os.replace(tmp, champion_path)
            logger.info(
                f"[{run_id}] Group champion written → {champion_path} ({model_id})"
            )

        return {
            "promoted":    promote,
            "model_id":    model_id,
            "new_score":   new_score,
            "old_score":   old_score,
            "new_metrics": new_metrics,
            "old_metrics": old_metrics,
            "reason":      reason,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Workflow step templates (read from workflow.yaml)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_step(self, step_type: str) -> Dict:
        for step in self.workflow_def.get("steps", []):
            if step["type"] == step_type:
                return step
        raise KeyError(f"No step of type '{step_type}' in workflow definition.")

    def _validation_step(self) -> Dict:
        return self._get_step("validation")

    def _training_step(self) -> Dict:
        return self._get_step("training")

    def _model_selection_step(self) -> Dict:
        return self._get_step("model_selection")

    def _mongo_config(self) -> Dict:
        return self.workflow_def.get("mongodb", {"enabled": False})

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Generic step executor
    # ─────────────────────────────────────────────────────────────────────────

    def _execute(self, run_id: str, step_id: str, step_type: str, config: Dict, fn):
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