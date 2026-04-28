"""
orchestrator.py
═══════════════════════════════════════════════════════════════════════════════
Workflow Orchestrator

Architecture clarification
──────────────────────────
The SCADA system is solely responsible for:
  - Raw data ingestion (acc_*.csv files)
  - Feature extraction
  - Writing features to MongoDB Feature Store ('features' collection)

The MLOps system (this orchestrator) is responsible for:
  - Validating features already in MongoDB
  - Training models on features already in MongoDB
  - Registering, comparing, and promoting models
  - Triggering serving via run_serving.py (hot-swap via champion.json)

There is NO ingestion, NO feature extraction, and NO local CSV reading
in this file. All training data comes from MongoDB exclusively.

start_workflow() behaviour
──────────────────────────
FIRST RUN  (no deployed model):
    1. Validate val bearings   (from MongoDB 'features' collection)
    2. Train on train bearings (from MongoDB 'features' collection)
    3. Select & deploy best model
    → Serving handled by run_serving.py (started by API ProcessManager)

SUBSEQUENT RUNS  (deployed model already exists):
    Steps 1–3 skipped. Serving continues via run_serving.py.

run_training_only() — called by run_preprod.py after fault confirmation:
    Reads confirmed_faults + train bearings from MongoDB.
    Does NOT touch the live Feature Store or Serving Pipeline.
    Registers new model as PENDING — run_preprod.py handles promotion.

confirm_fault_and_push_to_store() — called from API after fault confirm:
    Reads features from MongoDB 'features' collection,
    re-labels RUL, pushes to MongoDB 'confirmed_faults' collection.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import yaml
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

from scripts.data_validator import DataValidatorPHM
from models.model_trainer import RULTrainerPHM
from utils.db_collections import (
    COL_FACTORY_FEATURES        as _COL_FEATURES,
    COL_FEATURE_STORE_MIRRORED  as _COL_CONFIRMED,
    COL_SERVING_HISTORY         as _COL_SERV_HIST,
)
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

    roles         : train | val | test | live
    status values : available | confirmed | denied | error
    """

    VALID_STATUSES = {"available", "confirmed", "denied", "error"}

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

    def all_bearings(self)   -> List[Dict]: return self.bearings
    def train_bearings(self) -> List[Dict]: return [b for b in self.bearings if b["role"] == "train"]
    def val_bearings(self)   -> List[Dict]: return [b for b in self.bearings if b["role"] == "val"]
    def live_bearings(self)  -> List[Dict]: return [b for b in self.bearings if b["role"] == "live"]

    def current_live_bearing(self) -> Optional[Dict]:
        if not self.live_queue:
            return None
        if self.current_live_index >= len(self.live_queue):
            logger.warning("Live bearing queue exhausted.")
            return None
        return self.get_bearing(self.live_queue[self.current_live_index])

    def advance_live_bearing(self) -> Optional[Dict]:
        self.current_live_index += 1
        self._save()
        next_b = self.current_live_bearing()
        if next_b:
            logger.info(f"Live queue advanced → now serving: {next_b['name']}")
        else:
            logger.info("Live bearing queue exhausted.")
        return next_b

    def get_bearing(self, name: str) -> Optional[Dict]:
        for b in self.bearings:
            if b["name"] == name:
                return b
        return None

    def set_status(self, name: str, status: str):
        assert status in self.VALID_STATUSES, f"Invalid status: {status}"
        for b in self.bearings:
            if b["name"] == name:
                b["status"] = status
                self._save()
                return
        raise ValueError(f"Bearing '{name}' not found.")

    def source_path(self, bearing: Dict) -> str:
        sp = bearing.get("source_path", "")
        if sp and os.path.isdir(sp):
            return sp
        return os.path.join(self.base_path, bearing["name"])


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowStateManager:

    def __init__(self, state_base: str = "workflow_data"):
        self.state_base = state_base

    def _state_file(self, run_id: str) -> str:
        return os.path.join(self.state_base, run_id, "state", "workflow_state.json")

    def init_state(self, run_id: str, workflow_name: str):
        path = self._state_file(run_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "run_id":     run_id,
            "workflow":   workflow_name,
            "status":     "running",
            "start_time": datetime.now().isoformat(),
            "steps":      {},
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _load(self, run_id: str) -> Dict:
        path = self._state_file(run_id)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_state(self, run_id: str, state: Dict):
        with open(self._state_file(run_id), "w") as f:
            json.dump(state, f, indent=2)

    def update_step_status(self, run_id: str, step_id: str, status: str, error: str = None):
        state = self._load(run_id)
        state.setdefault("steps", {})[step_id] = {
            "status":     status,
            "updated_at": datetime.now().isoformat(),
            "error":      error,
        }
        self._save_state(run_id, state)

    def mark_step_outputs(self, run_id: str, step_id: str, outputs: Dict):
        state = self._load(run_id)
        state.setdefault("steps", {}).setdefault(step_id, {})["outputs"] = outputs
        self._save_state(run_id, state)

    def mark_workflow_complete(self, run_id: str):
        state = self._load(run_id)
        state["status"]   = "complete"
        state["end_time"] = datetime.now().isoformat()
        self._save_state(run_id, state)

    def mark_workflow_failed(self, run_id: str, tb: str):
        state = self._load(run_id)
        state["status"]    = "failed"
        state["end_time"]  = datetime.now().isoformat()
        state["traceback"] = tb
        self._save_state(run_id, state)

    def get_state(self, run_id: str) -> Dict:
        return self._load(run_id)


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ContractManager:

    def resolve_config(self, config_template: Dict, run_id: str) -> Dict:
        resolved = {}
        for k, v in config_template.items():
            if isinstance(v, str):
                resolved[k] = v.replace("{run_id}", run_id)
            elif isinstance(v, dict):
                resolved[k] = self.resolve_config(v, run_id)
            else:
                resolved[k] = v
        return resolved


# ─────────────────────────────────────────────────────────────────────────────
# MONGO HELPER — the ONLY data source for training/validation
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_bearing_from_mongo(
    bearing_name:    str,
    collection_name: str,
    mongo_uri:       str,
    db_name:         str,
) -> Optional[pd.DataFrame]:
    """
    Read all feature rows for a bearing from a MongoDB collection.
    Returns a DataFrame, or None if nothing is found.

    This is the ONLY path for reading training/validation data.
    SCADA writes features here; MLOps reads from here.
    """
    try:
        from pymongo import MongoClient
        client  = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        records = list(client[db_name][collection_name].find({"dataset_id": bearing_name}))
        if not records:
            return None
        for r in records:
            r.pop("_id",        None)
            r.pop("dataset_id", None)
            r.pop("version",    None)
            r.pop("metadata",   None)
        df = pd.DataFrame(records)
        df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
        logger.info(
            f"  [{bearing_name}] Loaded {len(df)} rows from "
            f"MongoDB '{collection_name}'."
        )
        return df
    except Exception as e:
        logger.warning(
            f"  [{bearing_name}] Could not read from MongoDB "
            f"'{collection_name}': {e}"
        )
        return None


# Columns the trainer must keep even though they look like non-features
_TRAINER_KEEP_COLS = {"time_s", "burst_idx", "RUL_s", "RUL_norm", "file_id"}

# Known junk columns that come back from MongoDB docs and must be stripped
_MONGO_JUNK_COLS = {
    "_id", "dataset_id", "version", "metadata",
    "source", "seeded_at", "bearing_name", "role", "run_id",
    "consumed", "consumed_at", "sent_at", "session_end",
    "scada_stats", "features",   # nested dict fields from live_features
}


def _sanitise_for_trainer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame pulled from MongoDB so it is safe to write as a
    training CSV and will not cause pd.concat() to crash.

    Steps
    ─────
    1. Drop known MongoDB metadata / junk columns.
    2. Drop columns whose dtype is object/dict/list (non-numeric) unless
       they are in _TRAINER_KEEP_COLS.
    3. Deduplicate column names — keep the first occurrence of each name
       (duplicate columns from MongoDB nested-doc expansion are the main
       cause of the InvalidIndexError in pd.concat).
    4. Reset the index to a clean 0-based RangeIndex.
    """
    # 1. Drop junk columns
    drop = [c for c in df.columns if c in _MONGO_JUNK_COLS]
    if drop:
        df = df.drop(columns=drop, errors="ignore")

    # 2. Drop non-numeric columns that are not trainer keep-cols
    bad_type = [
        c for c in df.columns
        if c not in _TRAINER_KEEP_COLS
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if bad_type:
        logger.debug(f"Dropping non-numeric columns from temp CSV: {bad_type}")
        df = df.drop(columns=bad_type, errors="ignore")

    # 3. Deduplicate column names (keep first occurrence)
    seen = set()
    keep = []
    for col in df.columns:
        if col not in seen:
            keep.append(col)
            seen.add(col)
        else:
            logger.debug(f"Dropping duplicate column: {col}")
    df = df[keep]

    # 4. Clean index
    df = df.reset_index(drop=True)
    return df


def _sanitise_for_trainer(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns and non-numeric junk before writing training CSVs."""
    _KEEP = {"time_s", "burst_idx", "RUL_s", "RUL_norm", "file_id"}
    _JUNK = {
        "_id", "dataset_id", "version", "metadata", "source", "seeded_at",
        "bearing_name", "role", "run_id", "consumed", "consumed_at",
        "sent_at", "session_end", "scada_stats", "features", "doc_type",
    }
    # Drop known junk
    df = df.drop(columns=[c for c in df.columns if c in _JUNK], errors="ignore")
    # Drop non-numeric columns that aren't needed by the trainer
    df = df.drop(columns=[
        c for c in df.columns
        if c not in _KEEP and not pd.api.types.is_numeric_dtype(df[c])
    ], errors="ignore")
    # Deduplicate column names — keep first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reset_index(drop=True)


def _df_to_temp_csv(df: pd.DataFrame, tmp_dir: str, name: str) -> str:
    """Sanitise a MongoDB-sourced DataFrame and write it as a trainer-ready CSV."""
    df = _sanitise_for_trainer(df)
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def _fetch_features_from_serving_history(
    bearing_name: str,
    mongo_uri:    str,
    db_name:      str,
) -> Optional[pd.DataFrame]:
    """
    Reconstruct a feature DataFrame from MongoDB 'serving_history' for a
    live bearing.

    run_serving.py writes one document per burst to serving_history,
    containing the full pipeline output including the feature vector used
    for inference. This function reassembles those per-burst records into
    a flat DataFrame that can be re-labelled with RUL and pushed to
    'confirmed_faults' for retraining.

    The key fields extracted are:
        time_s      — burst timestamp in seconds (from burst_idx * burst_period)
        burst_idx   — chronological burst index
        h_*/v_*     — the 18 base features from the FE stage
        rul_s       — the predicted RUL at that burst (used as a reference only)

    Returns None if no records found.
    """
    try:
        from pymongo import MongoClient, ASCENDING
        client  = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        col     = client[db_name][_COL_SERV_HIST]
        records = list(
            col.find({"bearing_name": bearing_name})
               .sort("burst_idx", ASCENDING)
        )
        if not records:
            return None

        rows = []
        for r in records:
            row = {}

            # ── Burst timing ──────────────────────────────────────────────────
            row["burst_idx"] = r.get("burst_idx", 0)
            # Reconstruct time_s from burst_idx assuming 10s per burst.
            # If the record has a timestamp we prefer that, but burst_idx * 10
            # is always consistent with what the trainer expects.
            row["time_s"]   = float(r.get("burst_idx", 0)) * 10.0

            # ── Features from the FE stage (stored in the pipeline output) ────
            # serving_history stores features under the 'features' sub-dict
            fe_features = r.get("features") or r.get("fe", {}) or {}
            for k, v in fe_features.items():
                if isinstance(v, (int, float)):
                    row[k] = v

            # ── Inference output (predicted RUL — reference only) ─────────────
            inf = r.get("inference") or {}
            row["rul_s"]   = inf.get("rul_s")   or r.get("rul_s")
            row["rul_min"] = inf.get("rul_min") or r.get("rul_min")

            rows.append(row)

        df = pd.DataFrame(rows).sort_values("burst_idx").reset_index(drop=True)

        # Strip rolling features (_mean/_std/_slope) — the trainer recomputes
        # these itself. Also strip serving pipeline outputs (rul_s, rul_min)
        # which are predictions, not ground truth labels.
        # Keep only: 18 base features + time_s + burst_idx + RUL_s + RUL_norm
        _STRIP_SUFFIXES = ("_mean", "_std", "_slope")
        _STRIP_EXACT = {"rul_s", "rul_min", "doc_type", "recorded_at",
                        "run_id", "model_version", "data_quality",
                        "drift_detected", "drift_features", "anomaly_flag",
                        "baseline_ready", "stats", "alert", "pm_status"}
        keep_cols = [
            c for c in df.columns
            if not any(c.endswith(s) for s in _STRIP_SUFFIXES)
               and c not in _STRIP_EXACT
        ]
        df = df[keep_cols]

        logger.info(
            f"  [{bearing_name}] Loaded {len(df)} bursts from "
            f"MongoDB 'serving_history' ({len(df.columns)} base feature columns)."
        )
        return df

    except Exception as e:
        logger.warning(
            f"  [{bearing_name}] Could not read from MongoDB "
            f"'serving_history': {e}"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowExecutor:

    def __init__(self, workflow_path: str = "config/workflow.yaml"):
        with open(workflow_path) as f:
            cfg = yaml.safe_load(f)
        self.workflow_def     = cfg["workflows"]["rul_prediction"]
        self.registry         = BearingRegistry(
            self.workflow_def.get("bearing_config", "config/bearings.json")
        )
        self.state_manager    = WorkflowStateManager()
        self.contract_manager = ContractManager()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Full workflow trigger
    # ─────────────────────────────────────────────────────────────────────────

    def start_workflow(self, run_id: str) -> str:
        """
        Entry point called by API on POST /workflow/trigger.

        FIRST RUN (no deployed model):
            Phase 1 — Validate val bearings (from MongoDB)
            Phase 2 — Train on train bearings (from MongoDB)
            Phase 3 — Select & deploy best model

        SUBSEQUENT RUNS:
            Skipped — deployed model exists, serving via run_serving.py.

        SCADA handles all ingestion and feature extraction.
        This workflow only trains and selects models.
        """
        logger.info(f"[{run_id}] Workflow started.")
        self.state_manager.init_state(run_id, "rul_prediction")

        try:
            from utils.model_registry import ModelRegistry
            deployed = ModelRegistry().get_deployed_model("RUL_s")

            if not deployed:
                logger.info(
                    f"[{run_id}] No deployed model found — running full pipeline."
                )

                logger.info(f"[{run_id}] Phase 1: Validation (from MongoDB)")
                self._run_validation(run_id)

                logger.info(f"[{run_id}] Phase 2: Training (from MongoDB)")
                self._run_training(run_id)

                logger.info(f"[{run_id}] Phase 3: Model Selection")
                self._run_model_selection(run_id)

            else:
                logger.info(
                    f"[{run_id}] Deployed model already exists — "
                    f"training skipped. Serving handled by run_serving.py."
                )

            logger.info(
                f"[{run_id}] Workflow complete. "
                f"Serving handled by run_serving.py (started by API)."
            )

        except Exception as e:
            self.state_manager.mark_workflow_failed(run_id, traceback.format_exc())
            logger.error(f"[{run_id}] Workflow FAILED: {e}", exc_info=True)
            raise

        self.state_manager.mark_workflow_complete(run_id)
        return run_id

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Pre-Production retraining
    # ─────────────────────────────────────────────────────────────────────────

    def run_training_only(
        self,
        run_id:    str,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name:   str = "phm_mlops",
    ) -> str:
        """
        Pre-Production retraining triggered after a fault is confirmed.

        Reads ALL data from MongoDB:
          - train-role bearings  → 'features' collection      (from SCADA)
          - confirmed fault data → 'confirmed_faults' collection

        Live Feature Store and Serving Pipeline are NOT touched.
        New model registered as PENDING; run_preprod.py handles promotion.
        """
        logger.info(f"[{run_id}] Pre-Production retraining started.")
        logger.info(f"[{run_id}] Data source: MongoDB only.")
        logger.info(f"[{run_id}] Live Feature Store and Serving Pipeline UNAFFECTED.")

        self.state_manager.init_state(run_id, "preprod_training")
        self._preprod_mongo_uri = mongo_uri
        self._preprod_db_name   = db_name

        try:
            logger.info(f"[{run_id}] Phase 1: Validation (from MongoDB)")
            self._run_validation(run_id, mongo_uri=mongo_uri, db_name=db_name)

            logger.info(f"[{run_id}] Phase 2: Training (from MongoDB)")
            self._run_training(run_id, mongo_uri=mongo_uri, db_name=db_name)

            self.state_manager.mark_workflow_complete(run_id)
            logger.info(
                f"[{run_id}] Pre-Production retraining complete. "
                f"New model registered as PENDING."
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

        Data source — depends on bearing role:
        ───────────────────────────────────────
        live bearings   → reads from MongoDB 'serving_history' collection.
                          This is where run_serving.py has written every
                          burst's full processed feature vector. This is the
                          correct source for a bearing that has just been
                          monitored live through the serving pipeline.

        train/val/test  → reads from MongoDB 'features' collection.
                          These are the historical run-to-failure features
                          seeded by seed_historical_data.py.

        After reading, re-labels RUL from the confirmed failure point
        and pushes the labelled DataFrame to MongoDB 'confirmed_faults'
        (Feature Store Mirrored) for use in pre-production retraining.

        No local disk reads at any point.
        """
        bearing = self.registry.get_bearing(bearing_name)
        if not bearing:
            raise ValueError(f"Bearing '{bearing_name}' not in registry.")

        mongo_cfg = self._mongo_config()
        if not mongo_cfg.get("enabled"):
            raise RuntimeError(
                "MongoDB is not enabled — cannot read features or push confirmed faults."
            )

        uri    = mongo_cfg["uri"]
        db     = mongo_cfg["db_name"]
        role   = bearing.get("role", "live")

        # ── Choose the correct source collection by bearing role ──────────────
        if role == "live":
            # Live bearings: features were processed by the serving pipeline
            # and written to serving_history one record per burst.
            logger.info(
                f"[{bearing_name}] Role=live — reading processed features "
                f"from MongoDB 'serving_history'.",
            )
            df = _fetch_features_from_serving_history(bearing_name, uri, db)
            if df is None or df.empty:
                raise FileNotFoundError(
                    f"No serving history found in MongoDB for live bearing '{bearing_name}'. "
                    f"Ensure run_serving.py has processed bursts for this bearing before "
                    f"confirming a fault."
                )
        else:
            # Train / val / test bearings: use the historical features
            # seeded by seed_historical_data.py.
            logger.info(
                f"[{bearing_name}] Role={role} — reading features "
                f"from MongoDB 'features'.",
            )
            df = _fetch_bearing_from_mongo(bearing_name, _COL_FEATURES, uri, db)
            if df is None or df.empty:
                raise FileNotFoundError(
                    f"No features found in MongoDB 'features' for bearing '{bearing_name}'. "
                    f"Run seed_historical_data.py to load historical features."
                )

        # ── Re-label RUL from the confirmed failure point ─────────────────────
        if "time_s" not in df.columns:
            raise ValueError(
                f"Features for '{bearing_name}' are missing the 'time_s' column. "
                f"Cannot re-label RUL."
            )
        failure_s      = float(df["time_s"].max())
        df["RUL_s"]    = (failure_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / failure_s if failure_s > 0 else 0.0

        logger.info(
            f"[{bearing_name}] Re-labelled RUL for {len(df)} bursts. "
            f"failure_s={failure_s:.0f}s, RUL range: "
            f"{df['RUL_s'].max():.0f}s → 0s"
        )

        # ── Write temp CSV for FeatureStore (expects a file path) ─────────────
        tmp_dir        = os.path.join("workflow_data", run_id, "tmp_confirmed")
        confirmed_path = _df_to_temp_csv(df, tmp_dir, f"confirmed_{bearing_name}")

        # ── Push to MongoDB 'confirmed_faults' (Feature Store Mirrored) ───────
        from utils.MongoDB import FeatureStore
        store = FeatureStore({
            "mongo_uri":       uri,
            "db_name":         db,
            "collection_name": _COL_CONFIRMED,
            "dataset_id":      bearing_name,
            "version":         run_id,
            "df_path":         confirmed_path,
            "metadata": {
                "bearing_name":   bearing_name,
                "role":           role,
                "run_id":         run_id,
                "confirmed":      True,
                "rul_at_failure": rul_at_failure,
                "confirmed_at":   datetime.now().isoformat(),
                "source":         "serving_history" if role == "live" else "features",
            },
        })
        result = store.run()
        logger.info(
            f"[{bearing_name}] Confirmed fault features pushed to "
            f"MongoDB 'confirmed_faults' (Feature Store Mirrored)."
        )

        self.registry.set_status(bearing_name, "confirmed")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Validation — reads from MongoDB only
    # ─────────────────────────────────────────────────────────────────────────

    def _run_validation(
        self,
        run_id:    str,
        mongo_uri: Optional[str] = None,
        db_name:   Optional[str] = None,
    ):
        """
        Validate val-bearing features read from MongoDB 'features' collection.
        SCADA writes features here; no disk reads.
        """
        step    = self._validation_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "validation"

        mongo_cfg = self._mongo_config()
        _uri = mongo_uri or mongo_cfg.get("uri", "mongodb://localhost:27017")
        _db  = db_name  or mongo_cfg.get("db_name", "phm_mlops")

        def _validate(cfg):
            validator    = DataValidatorPHM(
                schema_path=cfg.get("schema_path"),
                log_path=cfg.get("log_path"),
            )
            results_list = []
            for bearing in self.registry.val_bearings():
                df = _fetch_bearing_from_mongo(bearing["name"], _COL_FEATURES, _uri, _db)
                if df is None:
                    logger.warning(
                        f"  [{bearing['name']}] Not found in MongoDB 'features' — "
                        f"skipping. Ensure SCADA has written this bearing's data."
                    )
                    continue
                df, result = validator.validate_features(df)
                results_list.append({bearing["name"]: result})
                logger.info(f"  Validated {bearing['name']}")
            validator.save_results(results_list, cfg["output_location"])
            return {"validation_results": cfg["output_location"]}

        self._execute(run_id, step_id, "validation", config, _validate)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Training — reads from MongoDB only
    # ─────────────────────────────────────────────────────────────────────────

    def _run_training(
        self,
        run_id:    str,
        mongo_uri: Optional[str] = None,
        db_name:   Optional[str] = None,
    ):
        """
        Train on data pulled exclusively from MongoDB.

        Sources:
          - train-role bearings  → 'features' collection      (SCADA writes this)
          - confirmed fault data → 'confirmed_faults' collection
          - val-role bearings    → 'features' collection      (SCADA writes this)

        DataFrames are pulled from Mongo and written to temp CSVs so the
        existing RULTrainerPHM (which expects file paths) works unchanged.
        No local disk reads.
        """
        step    = self._training_step()
        config  = self.contract_manager.resolve_config(step["config"], run_id)
        step_id = "training"

        mongo_cfg = self._mongo_config()
        _uri = mongo_uri or mongo_cfg.get("uri", "mongodb://localhost:27017")
        _db  = db_name  or mongo_cfg.get("db_name", "phm_mlops")

        tmp_dir     = os.path.join("workflow_data", run_id, "tmp_training_csvs")
        train_files = []
        val_files   = []
        test_files  = []

        # ── Train bearings ────────────────────────────────────────────────────
        for bearing in self.registry.train_bearings():
            df = _fetch_bearing_from_mongo(bearing["name"], _COL_FEATURES, _uri, _db)
            if df is None:
                logger.warning(
                    f"  [{bearing['name']}] Not in MongoDB 'features' — skipping. "
                    f"Ensure SCADA has written this bearing's features."
                )
                continue
            train_files.append(_df_to_temp_csv(df, tmp_dir, f"train_{bearing['name']}"))

        # ── Confirmed fault data ──────────────────────────────────────────────
        try:
            from pymongo import MongoClient
            client        = MongoClient(_uri, serverSelectionTimeoutMS=5000)
            confirmed_col = client[_db][_COL_CONFIRMED]
            for bname in confirmed_col.distinct("dataset_id"):
                records = list(confirmed_col.find({"dataset_id": bname}))
                for r in records:
                    r.pop("_id", None)
                    r.pop("dataset_id", None)
                    r.pop("version", None)
                    r.pop("metadata", None)
                df = pd.DataFrame(records)
                if df.empty:
                    continue
                train_files.append(
                    _df_to_temp_csv(df, tmp_dir, f"confirmed_{bname}")
                )
                logger.info(
                    f"  [{bname}] Confirmed fault data loaded from MongoDB "
                    f"({len(df)} rows)."
                )
        except Exception as e:
            logger.warning(f"  Could not load confirmed faults from MongoDB: {e}")

        # ── Val bearings ──────────────────────────────────────────────────────
        for bearing in self.registry.val_bearings():
            df = _fetch_bearing_from_mongo(bearing["name"], _COL_FEATURES, _uri, _db)
            if df is None:
                logger.warning(
                    f"  [{bearing['name']}] Not in MongoDB 'features' — skipping val."
                )
                continue
            val_files.append(_df_to_temp_csv(df, tmp_dir, f"val_{bearing['name']}"))

        # ── Test bearings ─────────────────────────────────────────────────────
        for bearing in [b for b in self.registry.all_bearings() if b["role"] == "test"]:
            df = _fetch_bearing_from_mongo(bearing["name"], _COL_FEATURES, _uri, _db)
            if df is None:
                continue
            test_files.append(_df_to_temp_csv(df, tmp_dir, f"test_{bearing['name']}"))

        if not train_files:
            logger.warning(
                f"[{run_id}] No training data found in MongoDB — skipping training. "
                f"Ensure SCADA has written features for all train-role bearings."
            )
            self.state_manager.update_step_status(run_id, step_id, "SKIPPED")
            return

        n_confirmed = sum(1 for f in train_files if "confirmed_" in os.path.basename(f))
        logger.info(
            f"[{run_id}] Training: {len(train_files)} train file(s) "
            f"({n_confirmed} confirmed fault file(s)), "
            f"{len(val_files)} val, {len(test_files)} test — ALL from MongoDB."
        )
        config["train_files"] = train_files
        config["val_files"]   = val_files
        config["test_files"]  = test_files

        self._execute(run_id, step_id, "training", config,
                      lambda cfg: RULTrainerPHM(cfg).run(run_id=run_id))

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Model selection
    # ─────────────────────────────────────────────────────────────────────────

    def _run_model_selection(self, run_id: str):
        from utils.model_registry import ModelRegistry
        registry = ModelRegistry()
        result   = registry.compare_and_promote(run_id=run_id, metric="mae_s")
        if not result["model_id"]:
            raise RuntimeError(f"No pending models found for run_id={run_id}")
        logger.info(f"[{run_id}] Model selection: {result['reason']}")
        return {"model_id": result["model_id"], "promoted": result["promoted"]}

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Step templates
    # ─────────────────────────────────────────────────────────────────────────

    def _get_step(self, step_type: str) -> Dict:
        for step in self.workflow_def.get("steps", []):
            if step["type"] == step_type:
                return step
        raise KeyError(f"No step of type '{step_type}' in workflow definition.")

    def _validation_step(self)      -> Dict: return self._get_step("validation")
    def _training_step(self)        -> Dict: return self._get_step("training")
    def _model_selection_step(self) -> Dict: return self._get_step("model_selection")

    def _mongo_config(self) -> Dict:
        return self.workflow_def.get("mongodb", {"enabled": False})

    def _execute(self, run_id, step_id, step_type, config, fn):
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
            logger.error(f"  [{run_id}] Step '{step_id}' FAILED: {e}", exc_info=True)
            raise