"""
orchestrator_MongoDB.py
═══════════════════════════════════════════════════════════════════════════════
Workflow Orchestrator — MongoDB edition.

Identical to orchestrator.py with these additions:
  1. _mongo_config()        — reads MongoDB connection settings from workflow.yaml
  2. _run_extraction()      — pushes features to MongoDB Feature Store after extraction
  3. _run_mongo_backfill()  — pushes already-extracted features.csv files into MongoDB
  4. start_workflow()       — includes Phase 2b (backfill) and Phase 7 (Serving Pipeline)
  5. _run_serving_pipeline()— runs the 4-stage Serving Pipeline for all live bearings
                              and persists results to Serving History (MongoDB)
"""

import os
import json
import yaml
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

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
    Loads config/bearings.json and tracks data availability per bearing.

    roles:
        train     — run to failure, used for training
        val_test  — test bearing also used as validation during training
        test      — test bearing only
        live      — bearing whose data is streamed for live inference

    status values:
        available  — data exists on disk, ready to ingest
        missing    — data not yet available (sensor not yet run / files not copied)
        ingested   — raw CSVs consolidated into parquet
        extracted  — burst features extracted to CSV
        error      — a pipeline step failed for this bearing
    """

    VALID_STATUSES = {"available", "missing", "ingested", "extracted", "error"}

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load()

    def _load(self):
        with open(self.config_path, "r") as f:
            data = json.load(f)
        self.base_path = data["base_path"]
        self.bearings  = data["bearings"]

    def _save(self):
        with open(self.config_path, "r") as f:
            data = json.load(f)
        data["bearings"] = self.bearings
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Queries ───────────────────────────────────────────────────────────────

    def all_bearings(self) -> List[Dict]:
        return self.bearings

    def available_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b.get("status") == "available"]

    def missing_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b.get("status") == "missing"]

    def ready_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b.get("status") == "extracted"]

    def train_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "train"]

    def val_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "val_test"]

    def test_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] in ("test", "val_test")]

    def live_bearings(self) -> List[Dict]:
        return [b for b in self.bearings if b["role"] == "live"]

    def is_test(self, bearing: Dict) -> bool:
        return bearing["role"] != "train"

    def source_path(self, bearing: Dict) -> str:
        return os.path.join(self.base_path, bearing["name"])

    def print_status(self):
        from collections import Counter
        counts = Counter(b.get("status", "unknown") for b in self.bearings)
        logger.info("── Bearing registry status ──────────────────────")
        for bearing in self.bearings:
            status = bearing.get("status", "unknown")
            role   = bearing["role"]
            name   = bearing["name"]
            exists = "exists" if os.path.isdir(self.source_path(bearing)) else "NO FOLDER"
            logger.info(f"  {name:<14} role={role:<9} status={status:<10} disk={exists}")
        logger.info(f"  Summary: {dict(counts)}")
        logger.info("─────────────────────────────────────────────────")

    # ── Updates ───────────────────────────────────────────────────────────────

    def set_status(self, bearing_name: str, status: str):
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {self.VALID_STATUSES}")
        for b in self.bearings:
            if b["name"] == bearing_name:
                b["status"] = status
                self._save()
                logger.info(f"  Registry: {bearing_name} -> {status}")
                return
        raise ValueError(f"Bearing '{bearing_name}' not found in registry")

    def sync_from_disk(self):
        changed = False
        for b in self.bearings:
            if b.get("status") == "missing" and os.path.isdir(self.source_path(b)):
                b["status"] = "available"
                changed = True
                logger.info(f"  Auto-detected: {b['name']} -> available")
        if changed:
            self._save()


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowStateManager:
    def __init__(self, base_dir="workflow_data"):
        self.base_dir = base_dir

    def create_run_state(self, run_id: str, step_ids: List[str]) -> Dict[str, Any]:
        state = {
            "run_id":     run_id,
            "status":     "RUNNING",
            "start_time": datetime.now().isoformat(),
            "steps": {
                sid: {
                    "status":     "PENDING",
                    "start_time": None,
                    "end_time":   None,
                    "outputs":    {},
                    "error":      None,
                }
                for sid in step_ids
            },
        }
        self._save(run_id, state)
        return state

    def _path(self, run_id: str) -> str:
        return os.path.join(self.base_dir, run_id, "state", f"{run_id}.json")

    def _save(self, run_id: str, state: Dict):
        path = self._path(run_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, run_id: str) -> Dict[str, Any]:
        with open(self._path(run_id), "r") as f:
            return json.load(f)

    def update_step_status(self, run_id: str, step_id: str,
                           status: str, error: str = None):
        state = self.load_state(run_id)
        step  = state["steps"][step_id]
        step["status"] = status
        if status == "RUNNING":
            step["start_time"] = datetime.now().isoformat()
        elif status in ("COMPLETE", "FAILED"):
            step["end_time"] = datetime.now().isoformat()
        if error:
            step["error"] = str(error)
        self._save(run_id, state)

    def mark_step_outputs(self, run_id: str, step_id: str, outputs: Dict):
        state = self.load_state(run_id)
        state["steps"][step_id]["outputs"] = outputs
        self._save(run_id, state)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class DataContractManager:
    """Resolves {run_id} placeholders in path templates."""

    def resolve(self, template: str, run_id: str) -> str:
        if not template:
            return ""
        return template.format(run_id=run_id)

    def resolve_config(self, config: Dict, run_id: str) -> Dict:
        resolved = {}
        for key, val in config.items():
            if isinstance(val, str):
                resolved[key] = self.resolve(val, run_id)
            elif isinstance(val, list):
                resolved[key] = [
                    self.resolve(v, run_id) if isinstance(v, str) else v
                    for v in val
                ]
            elif isinstance(val, dict):
                resolved[key] = self.resolve_config(val, run_id)
            else:
                resolved[key] = val
        return resolved


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW EXECUTOR  (MongoDB edition)
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowExecutor:

    def __init__(
        self,
        workflow_name: str = "rul_prediction",
        yaml_path: str = "config/workflow.yaml",
    ):
        from utils.workflow_registry import WorkflowRegistry

        self.state_manager    = WorkflowStateManager()
        self.contract_manager = DataContractManager()

        # ── 1. Try the Workflow Registry first ────────────────────────────────
        wf_registry     = WorkflowRegistry()
        active_workflow = wf_registry.get_active_workflow(workflow_name)

        if active_workflow:
            self.workflow_def      = active_workflow["definition"]
            self._workflow_id      = active_workflow["workflow_id"]
            self._workflow_version = active_workflow["version"]
            logger.info(
                f"WorkflowExecutor: resolved '{workflow_name}' "
                f"v{self._workflow_version} (id={self._workflow_id}) "
                f"from WorkflowRegistry."
            )
        else:
            # ── 2. Fall back to local YAML ────────────────────────────────────
            logger.warning(
                f"WorkflowExecutor: no active workflow found in registry for "
                f"'{workflow_name}'. Falling back to '{yaml_path}'."
            )
            with open(yaml_path, "r") as f:
                workflows = yaml.safe_load(f)["workflows"]
            self.workflow_def      = workflows[workflow_name]
            self._workflow_id      = None
            self._workflow_version = "local"

        # ── 3. Bearing registry ───────────────────────────────────────────────
        bearing_config_path = self.workflow_def.get("bearing_config", "config/bearings.json")
        self.registry = BearingRegistry(bearing_config_path)

    # ── MongoDB config helper ─────────────────────────────────────────────────

    def _mongo_config(self) -> Dict:
        """
        Return MongoDB connection settings from the workflow definition.

        Expected in workflow.yaml under the workflow root:
            mongodb:
              enabled: true
              uri:     "mongodb://localhost:27017"
              db_name: "phm_mlops"

        Returns a dict with keys: enabled, uri, db_name.
        If the section is absent, returns {enabled: False}.
        """
        mongo = self.workflow_def.get("mongodb", {})
        return {
            "enabled": bool(mongo.get("enabled", False)),
            "uri":     mongo.get("uri",     "mongodb://localhost:27017"),
            "db_name": mongo.get("db_name", "phm_mlops"),
        }

    # ── Step template helpers ─────────────────────────────────────────────────

    def _ingestion_template(self) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == "ingestion")["config"]

    def _extraction_template(self) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == "feature_engineering")["config"]

    def _validation_step(self) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == "validation")

    def _training_step(self) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == "training")

    def _model_selection_step(self) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == "model_selection")

    def _live_serving_step(self) -> Dict:
        return next(s for s in self.workflow_def["steps"] if s["id"] == "live_serving")

    def _serving_pipeline_step(self) -> Dict:
        """Workflow step definition for the Serving Pipeline phase."""
        return {
            "id":   "serving_pipeline",
            "type": "serving_pipeline",
            "config": {
                "window_size":          40,
                "burst_period":         10.0,
                "realtime":             False,
                "critical_threshold_s": 3600,
                "warning_threshold_s":  14400,
                "baseline_path":        "model_registry/monitoring_baseline.json",
            },
        }

    # ── Entry point ───────────────────────────────────────────────────────────

    def start_workflow(self, run_id: str):
        logger.info(f"Starting RUL workflow (MongoDB) | run_id={run_id}")

        self.registry.sync_from_disk()
        self.registry.print_status()

        all_bearings = self.registry.all_bearings()
        missing      = self.registry.missing_bearings()

        if missing:
            logger.warning(
                f"  {len(missing)} bearing(s) registered but data missing: "
                f"{[b['name'] for b in missing]}"
            )

        def _parquet(b):
            return os.path.join(self.registry.source_path(b), "vibration_consolidated.parquet")
        def _features(b):
            return os.path.join(self.registry.source_path(b), "features.csv")

        non_live = [b for b in all_bearings if b["role"] != "live"]

        needs_ingestion  = [b for b in non_live
                            if b.get("status") != "missing"
                            and not os.path.exists(_parquet(b))
                            and not os.path.exists(_features(b))]
        needs_extraction = [b for b in non_live
                            if b.get("status") != "missing"
                            and os.path.exists(_parquet(b))
                            and not os.path.exists(_features(b))]
        done             = [b for b in non_live if os.path.exists(_features(b))]

        if not needs_ingestion and not needs_extraction and not done:
            logger.error("No bearings have data on disk. Aborting workflow.")
            return

        logger.info(
            f"  {len(needs_ingestion)} need ingestion+extraction, "
            f"{len(needs_extraction)} need extraction only, "
            f"{len(done)} already have features.csv"
        )

        ingest_ids   = [f"ingest_{b['name'].lower()}"  for b in needs_ingestion]
        extract_ids  = [f"extract_{b['name'].lower()}" for b in needs_ingestion + needs_extraction]
        all_step_ids = (
            ingest_ids + extract_ids
            + ["validation", "training", "model_selection", "serving_pipeline"]
        )

        self.state_manager.create_run_state(run_id, all_step_ids)

        # ── 1. Ingestion ──────────────────────────────────────────────────────
        if needs_ingestion:
            logger.info(f"[{run_id}] Phase 1: Ingestion ({len(needs_ingestion)} bearings)")
            for bearing in needs_ingestion:
                self._run_ingestion(run_id, bearing)
        else:
            logger.info(f"[{run_id}] Phase 1: Ingestion — skipped")

        # ── 2. Feature extraction ─────────────────────────────────────────────
        just_ingested     = [b for b in needs_ingestion if b.get("status") == "ingested"]
        needs_extract_now = just_ingested + needs_extraction
        if needs_extract_now:
            logger.info(f"[{run_id}] Phase 2: Feature extraction ({len(needs_extract_now)} bearings)")
            for bearing in needs_extract_now:
                self._run_extraction(run_id, bearing)
        else:
            logger.info(f"[{run_id}] Phase 2: Feature extraction — skipped")

        # ── 2b. MongoDB backfill ──────────────────────────────────────────────
        logger.info(f"[{run_id}] Phase 2b: MongoDB Feature Store backfill")
        self._run_mongo_backfill(run_id)

        # ── 3. Validation ─────────────────────────────────────────────────────
        logger.info(f"[{run_id}] Phase 3: Validation")
        self._run_validation(run_id)

        # ── 4. Training ───────────────────────────────────────────────────────
        logger.info(f"[{run_id}] Phase 4: Training")
        self._run_training(run_id)

        # ── 5. Model selection ────────────────────────────────────────────────
        logger.info(f"[{run_id}] Phase 5: Model selection")
        self._run_model_selection(run_id)

        # ── 6. Serving Pipeline ───────────────────────────────────────────────
        logger.info(f"[{run_id}] Phase 6: Serving Pipeline")
        self._run_serving_pipeline(run_id)

        logger.info(f"[{run_id}] Workflow complete!")

    # ── Per-phase helpers ─────────────────────────────────────────────────────

    def _run_ingestion(self, run_id: str, bearing: Dict):
        """Ingest one bearing — build config from template + registry."""
        step_id       = f"ingest_{bearing['name'].lower()}"
        template      = self.contract_manager.resolve_config(self._ingestion_template(), run_id)
        state_base    = template["state_base"]
        log_base      = template["log_base"]
        source_folder = self.registry.source_path(bearing)

        config = {
            "input_location":  source_folder,
            "output_location": source_folder,
            "state_location":  os.path.join(state_base, f"{step_id}.flag"),
            "save_format":     template.get("save_format", "parquet"),
            "log_path":        os.path.join(log_base, f"{step_id}.log"),
        }

        self._execute(run_id, step_id, "ingestion", config,
                      lambda cfg: DataIngestorPHM(cfg).run())

        state = self.state_manager.load_state(run_id)
        if state["steps"][step_id]["status"] == "COMPLETE":
            self.registry.set_status(bearing["name"], "ingested")
        else:
            self.registry.set_status(bearing["name"], "error")

    def _run_extraction(self, run_id: str, bearing: Dict):
        """Extract features for one bearing, then push to MongoDB Feature Store."""
        step_id       = f"extract_{bearing['name'].lower()}"
        template      = self.contract_manager.resolve_config(self._extraction_template(), run_id)
        state_base    = template["state_base"]
        log_base      = template["log_base"]
        source_folder = self.registry.source_path(bearing)

        config = {
            "input_location":    os.path.join(source_folder, "vibration_consolidated.parquet"),
            "output_location":   os.path.join(source_folder, "features.csv"),
            "state_location":    os.path.join(state_base, f"{step_id}.flag"),
            "log_path":          os.path.join(log_base, f"{step_id}.log"),
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
            # ── Push features to MongoDB Feature Store ────────────────────────
            mongo_cfg = self._mongo_config()
            if mongo_cfg.get("enabled"):
                from utils.MongoDB import FeatureStore
                features_path = os.path.join(source_folder, "features.csv")
                store = FeatureStore({
                    "mongo_uri":        mongo_cfg["uri"],
                    "db_name":          mongo_cfg["db_name"],
                    "collection_name":  "features",
                    "dataset_id":       bearing["name"],
                    "version":          run_id,
                    "df_path":          features_path,
                    "metadata": {
                        "bearing_name": bearing["name"],
                        "role":         bearing["role"],
                        "run_id":       run_id,
                    },
                })
                store.run()
                logger.info(f"  [{bearing['name']}] Features ingested into MongoDB")
        else:
            self.registry.set_status(bearing["name"], "error")

    def _run_mongo_backfill(self, run_id: str):
        """Push already-extracted features.csv files into MongoDB (one-time backfill)."""
        mongo_cfg = self._mongo_config()
        if not mongo_cfg.get("enabled"):
            logger.info("MongoDB not enabled — skipping backfill.")
            return

        from utils.MongoDB import FeatureStore

        logger.info(f"[{run_id}] MongoDB backfill: pushing existing features.csv files...")
        for bearing in self.registry.all_bearings():
            if bearing["role"] == "live":
                continue  # live bearing handled separately
            source_folder = self.registry.source_path(bearing)
            features_path = os.path.join(source_folder, "features.csv")

            if not os.path.exists(features_path):
                logger.warning(f"  [{bearing['name']}] No features.csv found — skipping.")
                continue

            store = FeatureStore({
                "mongo_uri":        mongo_cfg["uri"],
                "db_name":          mongo_cfg["db_name"],
                "collection_name":  "features",
                "dataset_id":       bearing["name"],
                "version":          run_id,
                "df_path":          features_path,
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
        """Validate all feature CSVs produced across all bearings."""
        step    = self._validation_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "validation"

        def _validate(cfg):
            validator    = DataValidatorPHM(schema_path=cfg.get("schema_path"),
                                            log_path=cfg.get("log_path"))
            results_list = []

            for bearing in self.registry.all_bearings():
                csv_path = os.path.join(
                    self.registry.source_path(bearing), "features.csv"
                )
                if not os.path.exists(csv_path):
                    logger.warning(f"  No features.csv for {bearing['name']} - skipping validation.")
                    continue
                df = pd.read_csv(csv_path)
                df, result = validator.validate_features(df)
                results_list.append({bearing["name"]: result})
                logger.info(f"  Validated {bearing['name']}")

            validator.save_results(results_list, cfg["output_location"])
            return {"validation_results": cfg["output_location"]}

        self._execute(run_id, step_id, "validation", config, _validate)

    def _run_training(self, run_id: str):
        """Build file lists from bearing roles and extracted status, then train."""
        step    = self._training_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "training"

        def _features_csv(bearing: Dict) -> str:
            return os.path.join(self.registry.source_path(bearing), "features.csv")

        def _has_features(bearing: Dict) -> bool:
            return os.path.exists(_features_csv(bearing))

        train_ready = [b for b in self.registry.train_bearings() if _has_features(b)]
        val_ready   = [b for b in self.registry.val_bearings()   if _has_features(b)]
        test_ready  = [b for b in self.registry.test_bearings()  if _has_features(b)]

        if not train_ready:
            logger.error("No training bearings have features.csv — skipping training.")
            return
        if not val_ready:
            logger.error("No validation bearings have features.csv — skipping training.")
            return

        config["train_files"] = [_features_csv(b) for b in train_ready]
        config["val_files"]   = [_features_csv(b) for b in val_ready]
        config["test_files"]  = [_features_csv(b) for b in test_ready]

        logger.info(f"  Train ({len(train_ready)}): {[b['name'] for b in train_ready]}")
        logger.info(f"  Val   ({len(val_ready)}):   {[b['name'] for b in val_ready]}")
        logger.info(f"  Test  ({len(test_ready)}):  {[b['name'] for b in test_ready]}")

        missing_features = [b["name"] for b in self.registry.all_bearings()
                            if not _has_features(b)]
        if missing_features:
            logger.warning(f"  No features.csv (skipped): {missing_features}")

        self._execute(run_id, step_id, "training", config,
                      lambda cfg: RULTrainerPHM(cfg).run(run_id=run_id))

    def _run_model_selection(self, run_id: str):
        """Select and deploy the best model by lowest mae_s."""
        step    = self._model_selection_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "model_selection"

        self._execute(run_id, step_id, "model_selection", config,
                      lambda cfg: {"best_model_id":
                                   self.select_best_model(run_id, cfg.get("metric", "mae_s"))["model_id"]})

    def _run_live_serving(self, run_id: str):
        """Stream live bearing data through the deployed model and save predictions."""
        live_bearings = self.registry.live_bearings()
        if not live_bearings:
            logger.info(f"[{run_id}] Phase 6: Live serving — skipped (no bearings with role='live')")
            return

        step    = self._live_serving_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "live_serving"

        def _serve(cfg):
            from Live_implementation.live_feature_buffer import LiveFeatureBuffer
            from Live_implementation.live_predictor      import LivePredictor
            from utils.model_registry                    import ModelRegistry

            registry    = ModelRegistry()
            model_entry = registry.get_deployed_model("RUL_s")
            if not model_entry:
                raise RuntimeError(
                    "No deployed model found. Ensure training + model_selection completed successfully."
                )

            predictor    = LivePredictor.from_path(model_entry["model_path"])
            window_size  = int(cfg.get("window_size", 40))
            burst_period = float(cfg.get("burst_period", 10.0))
            realtime     = bool(cfg.get("realtime", False))

            all_predictions = []

            for bearing in live_bearings:
                source_folder = self.registry.source_path(bearing)
                logger.info(
                    f"[{run_id}] Live serving: {bearing['name']} "
                    f"({'realtime' if realtime else 'fast replay'})"
                )

                buffer   = LiveFeatureBuffer(window_size=window_size)
                ingestor = DataIngestorPHM(config={
                    "input_location":  source_folder,
                    "output_location": source_folder,
                })

                predictions   = []
                live_features = []
                for burst in ingestor.stream_bursts(
                    source_folder,
                    burst_period=burst_period,
                    realtime=realtime,
                ):
                    vec = buffer.push_burst(burst["h_signal"], burst["v_signal"])

                    # Always save raw base features from burst 0 regardless
                    # of warmup state — full history needed for retraining.
                    raw = buffer._deque[-1]
                    feature_row = {
                        "burst_idx": burst["burst_idx"],
                        "time_s":    burst["time_s"],
                        **{k: v for k, v in raw.items() if k != "RUL_norm"},
                        "RUL_s":    None,
                        "RUL_norm": None,
                    }
                    live_features.append(feature_row)

                    if vec is None:
                        continue

                    rul_s = predictor.predict(vec)

                    entry = {
                        "bearing":   bearing["name"],
                        "burst_idx": burst["burst_idx"],
                        "time_s":    burst["time_s"],
                        "rul_s":     rul_s,
                        "rul_min":   rul_s / 60.0,
                        "h_max":     burst["h_max"],
                        "v_max":     burst["v_max"],
                    }
                    predictions.append(entry)

                    logger.info(
                        f"  [{bearing['name']}] Burst {burst['burst_idx']:>5} "
                        f"| t={burst['time_s']:>8.0f} s "
                        f"| RUL = {rul_s:>8.0f} s  ({rul_s/60:.1f} min)"
                    )

                out_base = cfg.get("output_base", f"workflow_data/{run_id}/live")
                os.makedirs(out_base, exist_ok=True)

                if predictions:
                    out_path = os.path.join(out_base, f"{bearing['name']}_predictions.csv")
                    pd.DataFrame(predictions).to_csv(out_path, index=False)
                    logger.info(
                        f"  [{bearing['name']}] {len(predictions)} predictions saved → {out_path}"
                    )
                    all_predictions.extend(predictions)

                if live_features:
                    features_path  = os.path.join(out_base, f"{bearing['name']}_live_features.csv")
                    features_df    = pd.DataFrame(live_features)
                    failure_time_s = features_df["time_s"].max()
                    features_df["RUL_s"]    = (failure_time_s - features_df["time_s"]).clip(lower=0.0)
                    features_df["RUL_norm"] = (
                        features_df["RUL_s"] / failure_time_s if failure_time_s > 0 else 0.0
                    )
                    features_df.drop(columns=["bearing"], inplace=True, errors="ignore")
                    features_df.to_csv(features_path, index=False)
                    logger.info(
                        f"  [{bearing['name']}] {len(features_df)} feature rows saved → {features_path}"
                    )
                    logger.info(
                        f"  [{bearing['name']}] RUL labelled: "
                        f"failure_time={failure_time_s:.0f} s | "
                        f"RUL range: {features_df['RUL_s'].max():.0f} s -> 0 s"
                    )

            return {
                "n_predictions":   len(all_predictions),
                "bearings_served": [b["name"] for b in live_bearings],
            }

        self._execute(run_id, step_id, "live_serving", config, _serve)

    def _run_serving_pipeline(self, run_id: str) -> None:
        """
        Phase 7: Run the full 4-stage Serving Pipeline for all live bearings.

        Wires (per diagram):
            WFOrch ──(step 5/6)──► ServPipeline ──(step 7)──► FeatStore
                                                 ──(step 9)──► ServHistory
                                                 ──────────►  ExportSvc (stub)
                                                 ──────────►  Dashboard (via monitoring)
        """
        live_bearings = self.registry.live_bearings()
        if not live_bearings:
            logger.info(
                f"[{run_id}] Phase 7: Serving Pipeline — skipped (no live bearings)"
            )
            return

        step   = self._serving_pipeline_step()
        config = self.contract_manager.resolve_config(step["config"], run_id)

        # ── Build pipeline config, injecting MongoDB settings if available ────
        mongo_cfg    = self._mongo_config()
        pipeline_cfg = {
            "window_size":            int(config.get("window_size", 40)),
            "burst_period":           float(config.get("burst_period", 10.0)),
            "realtime":               bool(config.get("realtime", False)),
            "critical_threshold_s":   int(config.get("critical_threshold_s", 3600)),
            "warning_threshold_s":    int(config.get("warning_threshold_s", 14400)),
            "baseline_path":          config.get(
                "baseline_path", "model_registry/monitoring_baseline.json"
            ),
            "enable_serving_history": True,
            "mongo_uri":              mongo_cfg["uri"],
            "db_name":                mongo_cfg["db_name"],
        }

        def _run_pipeline(cfg):
            from serving_pipeline.serving_pipeline import ServingPipeline

            pipeline    = ServingPipeline(config=pipeline_cfg)
            all_results = []

            for bearing in live_bearings:
                source_folder = self.registry.source_path(bearing)
                logger.info(
                    f"[{run_id}] Serving Pipeline: {bearing['name']} "
                    f"({'realtime' if pipeline_cfg['realtime'] else 'fast replay'})"
                )

                results = pipeline.run_bearing(
                    run_id        = run_id,
                    bearing_name  = bearing["name"],
                    source_folder = source_folder,
                    burst_period  = pipeline_cfg["burst_period"],
                    realtime      = pipeline_cfg["realtime"],
                )

                n_ready  = sum(1 for r in results if r.get("ready"))
                n_alerts = sum(
                    1 for r in results
                    if r.get("ready") and r.get("pm", {}).get("alert", False)
                )
                n_errors = sum(1 for r in results if not r.get("ok"))

                logger.info(
                    f"  [{bearing['name']}] bursts={len(results)}  "
                    f"ready={n_ready}  alerts={n_alerts}  errors={n_errors}"
                )
                all_results.extend(results)

            return {
                "bearings_served": [b["name"] for b in live_bearings],
                "total_bursts":    len(all_results),
                "ready_bursts":    sum(1 for r in all_results if r.get("ready")),
                "total_alerts":    sum(
                    1 for r in all_results
                    if r.get("ready") and r.get("pm", {}).get("alert", False)
                ),
            }

        self._execute(run_id, "serving_pipeline", "serving_pipeline", config, _run_pipeline)

    # ── Generic step executor ─────────────────────────────────────────────────

    def _execute(self, run_id: str, step_id: str, step_type: str,
                 config: Dict, fn):
        """
        Run a single step, update state, write flag file.
        fn receives the resolved config dict and returns an outputs dict.
        """
        logger.info(f"[{run_id}] Starting '{step_id}' ({step_type})")
        self.state_manager.update_step_status(run_id, step_id, "RUNNING")

        try:
            start    = time.time()
            outputs  = fn(config)
            duration = time.time() - start

            self.state_manager.mark_step_outputs(run_id, step_id, outputs or {})
            self.state_manager.update_step_status(run_id, step_id, "COMPLETE")
            logger.info(f"[{run_id}] '{step_id}' complete in {duration:.2f}s")

            if config.get("state_location"):
                flag = config["state_location"]
                os.makedirs(os.path.dirname(flag), exist_ok=True)
                with open(flag, "w") as f:
                    f.write(datetime.now().isoformat())

        except Exception as e:
            logger.error(f"[{run_id}] '{step_id}' failed: {e}", exc_info=True)
            self.state_manager.update_step_status(run_id, step_id, "FAILED", str(e))

    # ── Model selection ───────────────────────────────────────────────────────

    def select_best_model(self, run_id: str, metric: str = "mae_s") -> Dict:
        from utils.model_registry import ModelRegistry
        time.sleep(2)

        registry = ModelRegistry()
        models_this_run = (
            registry.list_models(status="pending",  run_id=run_id) +
            registry.list_models(status="approved", run_id=run_id)
        )

        if not models_this_run:
            raise ValueError(f"No models registered for run_id='{run_id}'")

        best = min(models_this_run,
                   key=lambda m: m["metrics"].get(metric, float("inf")))

        registry.approve_model(best["model_id"], approved_by="orchestrator")
        registry.deploy_model(best["model_id"])

        logger.info(
            f"[{run_id}] Best model: {best['model_id']} ({best['model_type']}) "
            f"with {metric}={best['metrics'].get(metric, 0):.1f} s"
        )
        return best


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    executor = WorkflowExecutor("config/workflow.yaml")
    executor.start_workflow()