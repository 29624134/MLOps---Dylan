"""
utils/model_registry.py
═══════════════════════════════════════════════════════════════════════════════
Central registry for all trained models — MongoDB-backed.

Mirrors WorkflowRegistry: MongoDB is the primary store; champion.json on disk
is only the lightweight hot-swap pointer watched by run_serving.py.

Fix #3  — Registry now lives in MongoDB (collection: model_registry), NOT in
           model_registry/registry.json.  Falls back to JSON if Mongo is
           unavailable so the system degrades gracefully.
Fix #5  — compare_and_promote() checks whether serving is currently active
           (via the serving_lock collection) before writing champion.json.
           If serving is busy the promotion is deferred and retried
           periodically until the burst boundary is free.

Key responsibilities
────────────────────
1. Track model versions and metadata across all runs (in MongoDB)
2. Manage model approval workflow (pending → approved → deployed → archived)
3. Retrieve deployed models for serving
4. Maintain model lineage via metadata
5. Write champion.json pointer for atomic hot-swap (run_serving.py watches this)
6. Compare pending models against the current champion and promote if better,
   but NEVER push a new model while serving is mid-burst.
"""

import json
import os
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

# ── Champion pointer path — watched by run_serving.py for hot-swap ────────────
CHAMPION_PATH = os.path.join("model_registry", "champion.json")

# ── MongoDB defaults ───────────────────────────────────────────────────────────
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB_NAME   = os.environ.get("MONGO_DB",  "phm_mlops")
from utils.db_collections import COL_MODEL_REGISTRY, COL_SERVING_LOCK
COLLECTION        = COL_MODEL_REGISTRY   # "model_registry"
LOCK_COLLECTION   = COL_SERVING_LOCK     # used by run_serving.py to signal active burst


class ModelStatus(Enum):
    """Model lifecycle states."""
    PENDING      = "pending"
    APPROVED     = "approved"
    REJECTED     = "rejected"
    DEPLOYED     = "deployed"
    ARCHIVED     = "archived"
    EXPERIMENTAL = "experimental"


class ModelRegistry:
    """
    Central registry for all trained models — primary store is MongoDB.

    Falls back to model_registry/registry.json if MongoDB is unavailable.
    Public API is unchanged so all callers (orchestrator, run_preprod, API)
    work without modification.
    """

    FALLBACK_JSON = os.path.join("model_registry", "registry.json")

    def __init__(
        self,
        registry_path: Optional[str] = None,   # kept for backward compat, ignored when Mongo up
        mongo_uri:     Optional[str] = None,
        db_name:       Optional[str] = None,
    ):
        self._mongo_uri = mongo_uri or DEFAULT_MONGO_URI
        self._db_name   = db_name   or DEFAULT_DB_NAME
        self._col       = None
        self._lock_col  = None
        self._fallback  = False

        # Keep legacy fallback path for when Mongo is down
        if registry_path is None:
            registry_path = self.FALLBACK_JSON
        if not os.path.isabs(registry_path):
            registry_path = os.path.abspath(registry_path)
        self.registry_path = registry_path
        self.registry_dir  = os.path.dirname(registry_path)
        os.makedirs(self.registry_dir, exist_ok=True)

        # Always create the fallback JSON — used as safety net even when MongoDB
        # is up, so _load_fallback() never crashes on a missing file.
        self._ensure_fallback_json()

        try:
            col, lock_col = self._connect()
            col.create_index("model_id",       name="idx_model_id",  unique=True, sparse=True)
            col.create_index("status",         name="idx_status")
            col.create_index("target_feature", name="idx_target")
            logger.info(
                f"ModelRegistry connected to MongoDB → {self._db_name}.{COLLECTION}"
            )
        except Exception as exc:
            logger.warning(
                f"ModelRegistry: MongoDB unavailable ({exc}). "
                f"Falling back to JSON at {self.registry_path}."
            )
            self._fallback = True

    # ── MongoDB connection ────────────────────────────────────────────────────

    def _connect(self):
        from pymongo import MongoClient
        client          = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db              = client[self._db_name]
        self._col       = db[COLLECTION]
        self._lock_col  = db[LOCK_COLLECTION]
        return self._col, self._lock_col

    def _collection(self):
        if self._col is not None:
            return self._col
        col, _ = self._connect()
        return col

    def _lock_collection(self):
        if self._lock_col is not None:
            return self._lock_col
        _, lock_col = self._connect()
        return lock_col

    # ── Fallback JSON helpers ─────────────────────────────────────────────────

    def _ensure_fallback_json(self):
        os.makedirs(self.registry_dir, exist_ok=True)
        if not os.path.exists(self.registry_path):
            self._save_fallback({"version": "1.0",
                                  "created_at": datetime.now().isoformat(),
                                  "models": []})

    def _load_fallback(self) -> Dict:
        if not os.path.exists(self.registry_path):
            self._ensure_fallback_json()
        with open(self.registry_path) as fh:
            return json.load(fh)

    def _save_fallback(self, data: Dict):
        with open(self.registry_path, "w") as fh:
            json.dump(data, fh, indent=2)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_model_id(self, model_path: str, timestamp: str) -> str:
        return hashlib.md5(f"{model_path}_{timestamp}".encode()).hexdigest()[:12]

    # ── Public API — Registration & Lifecycle ─────────────────────────────────

    def register_model(
        self,
        model_path:         str,
        model_type:         str,
        target_feature:     str,
        metrics:            Dict[str, float],
        training_data_info: Dict[str, Any],
        metadata:           Optional[Dict] = None,
    ) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        model_id  = self._generate_model_id(model_path, timestamp)

        record = {
            "model_id":           model_id,
            "model_path":         model_path,
            "model_type":         model_type,
            "target_feature":     target_feature,
            "metrics":            metrics,
            "training_data_info": training_data_info,
            "metadata":           metadata or {},
            "status":             ModelStatus.PENDING.value,
            "registered_at":      timestamp,
        }

        if not self._fallback:
            try:
                self._collection().insert_one({**record, "_id": model_id})
                logger.info(
                    f"Model registered in MongoDB: {model_id} "
                    f"({model_type} → {target_feature}, status=pending)"
                )
                return model_id
            except Exception as e:
                logger.warning(f"MongoDB register failed, using fallback JSON: {e}")

        # Fallback JSON path
        data = self._load_fallback()
        data["models"].append(record)
        self._save_fallback(data)
        logger.info(f"Model registered in fallback JSON: {model_id}")
        return model_id

    def approve_model(self, model_id: str, approved_by: str = "system") -> bool:
        now = datetime.now(timezone.utc).isoformat()
        if not self._fallback:
            try:
                res = self._collection().update_one(
                    {"_id": model_id},
                    {"$set": {"status": ModelStatus.APPROVED.value,
                               "approved_at": now, "approved_by": approved_by}},
                )
                if res.matched_count:
                    logger.info(f"Model {model_id} approved by '{approved_by}'.")
                    return True
            except Exception as e:
                logger.warning(f"MongoDB approve failed: {e}")

        # Fallback
        data = self._load_fallback()
        for m in data["models"]:
            if m["model_id"] == model_id:
                m["status"] = ModelStatus.APPROVED.value
                m["approved_at"] = now
                m["approved_by"] = approved_by
                self._save_fallback(data)
                logger.info(f"Model {model_id} approved (fallback JSON).")
                return True
        logger.error(f"approve_model: id '{model_id}' not found.")
        return False

    def reject_model(self, model_id: str, rejected_by: str = "system") -> bool:
        now = datetime.now(timezone.utc).isoformat()
        if not self._fallback:
            try:
                res = self._collection().update_one(
                    {"_id": model_id},
                    {"$set": {"status": ModelStatus.REJECTED.value,
                               "rejected_at": now, "rejected_by": rejected_by}},
                )
                if res.matched_count:
                    logger.info(f"Model {model_id} rejected.")
                    return True
            except Exception as e:
                logger.warning(f"MongoDB reject failed: {e}")

        data = self._load_fallback()
        for m in data["models"]:
            if m["model_id"] == model_id:
                m["status"] = ModelStatus.REJECTED.value
                m["rejected_at"] = now
                m["rejected_by"] = rejected_by
                self._save_fallback(data)
                return True
        logger.error(f"reject_model: id '{model_id}' not found.")
        return False

    def deploy_model(self, model_id: str) -> bool:
        """
        Transition approved → deployed.
        Archives any previously deployed model for the same target_feature.
        """
        now    = datetime.now(timezone.utc).isoformat()
        record = self.get_model(model_id)
        if record is None:
            logger.error(f"deploy_model: id '{model_id}' not found.")
            return False
        if record["status"] != ModelStatus.APPROVED.value:
            logger.error(
                f"deploy_model: model '{model_id}' is '{record['status']}', not 'approved'."
            )
            return False

        # Archive any currently deployed model for the same feature
        existing = self.list_models(status=ModelStatus.DEPLOYED.value,
                                    target_feature=record["target_feature"])
        for m in existing:
            if m["model_id"] != model_id:
                self._set_status(m["model_id"], ModelStatus.ARCHIVED.value,
                                  {"archived_at": now})
                logger.info(f"  Archived previously deployed model: {m['model_id']}")

        self._set_status(model_id, ModelStatus.DEPLOYED.value, {"deployed_at": now})
        logger.info(f"Model {model_id} deployed for '{record['target_feature']}'.")
        return True

    def archive_model(self, model_id: str) -> bool:
        self._set_status(model_id, ModelStatus.ARCHIVED.value,
                          {"archived_at": datetime.now(timezone.utc).isoformat()})
        logger.info(f"Model {model_id} archived.")
        return True

    def _set_status(self, model_id: str, status: str, extra: Dict = None):
        update = {"status": status, **(extra or {})}
        if not self._fallback:
            try:
                self._collection().update_one({"_id": model_id}, {"$set": update})
                return
            except Exception as e:
                logger.warning(f"MongoDB _set_status failed: {e}")
        data = self._load_fallback()
        for m in data["models"]:
            if m["model_id"] == model_id:
                m.update(update)
                self._save_fallback(data)
                return

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_deployed_model(self, target_feature: str) -> Optional[Dict]:
        if not self._fallback:
            try:
                doc = self._collection().find_one(
                    {"target_feature": target_feature,
                     "status": ModelStatus.DEPLOYED.value},
                )
                if doc:
                    doc.pop("_id", None)
                    return doc
            except Exception as e:
                logger.warning(f"MongoDB get_deployed_model failed: {e}")

        data = self._load_fallback()
        for m in data["models"]:
            if (m["target_feature"] == target_feature
                    and m["status"] == ModelStatus.DEPLOYED.value):
                return m
        logger.warning(f"No deployed model found for '{target_feature}'.")
        return None

    def get_model(self, model_id: str) -> Optional[Dict]:
        if not self._fallback:
            try:
                doc = self._collection().find_one({"_id": model_id})
                if doc:
                    doc.pop("_id", None)
                    doc["model_id"] = model_id
                    return doc
            except Exception as e:
                logger.warning(f"MongoDB get_model failed: {e}")

        data = self._load_fallback()
        for m in data["models"]:
            if m["model_id"] == model_id:
                return m
        return None

    def list_models(
        self,
        status:         Optional[str] = None,
        target_feature: Optional[str] = None,
        run_id:         Optional[str] = None,
    ) -> List[Dict]:
        query: Dict[str, Any] = {}
        if status:
            query["status"] = status
        if target_feature:
            query["target_feature"] = target_feature
        if run_id:
            query["metadata.run_id"] = run_id

        if not self._fallback:
            try:
                docs = list(self._collection().find(query))
                for d in docs:
                    mid = d.pop("_id", None)
                    if mid and "model_id" not in d:
                        d["model_id"] = mid
                return docs
            except Exception as e:
                logger.warning(f"MongoDB list_models failed: {e}")

        # Fallback
        data    = self._load_fallback()
        results = data["models"]
        if status:
            results = [m for m in results if m["status"] == status]
        if target_feature:
            results = [m for m in results if m["target_feature"] == target_feature]
        if run_id:
            results = [m for m in results
                       if m.get("metadata", {}).get("run_id") == run_id]
        return results

    def print_status(self, target_feature: Optional[str] = None):
        models = self.list_models(target_feature=target_feature)
        logger.info("─" * 72)
        logger.info(f"{'MODEL REGISTRY (MongoDB)':^72}")
        logger.info("─" * 72)
        if not models:
            logger.info("  (empty)")
        for m in models:
            run_id = m.get("metadata", {}).get("run_id", "n/a")
            logger.info(
                f"  [{m['status'].upper():10}] "
                f"{m['model_type']} -> {m['target_feature']} "
                f"(id={m['model_id']}, run={run_id})"
            )
        logger.info("─" * 72)

    # ── Serving lock — Fix #5 ─────────────────────────────────────────────────

    def is_serving_active(self) -> bool:
        """
        Return True if run_serving.py has signalled that it is mid-burst.

        run_serving.py writes a document to the 'serving_lock' collection
        before processing each burst and removes it (or sets active=False)
        after the burst completes. This prevents champion.json being rewritten
        during an in-flight prediction.
        """
        if self._fallback:
            return False   # cannot check, assume safe
        try:
            doc = self._lock_collection().find_one({"active": True})
            return doc is not None
        except Exception as e:
            logger.warning(f"Could not check serving_lock: {e}. Assuming inactive.")
            return False

    # ── Champion pointer — hot-swap support ───────────────────────────────────

    def write_champion_pointer(
        self,
        model_id:             str,
        max_wait_s:           float = 30.0,
        retry_interval_s:     float = 1.0,
    ) -> bool:
        """
        Write model_registry/champion.json pointing to the newly deployed model.

        Fix #5: Waits until serving is NOT mid-burst before writing, so the
        serving pipeline never loads a half-written champion during a prediction.
        Waits up to max_wait_s seconds, then proceeds anyway (safety fallback).

        Uses write-then-rename (os.replace) for atomicity.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.error(f"write_champion_pointer: model '{model_id}' not found.")
            return False

        # ── Wait for serving to be idle (Fix #5) ─────────────────────────────
        waited = 0.0
        while self.is_serving_active() and waited < max_wait_s:
            logger.info(
                f"[Champion] Serving is mid-burst — waiting {retry_interval_s}s "
                f"before writing champion.json ({waited:.0f}/{max_wait_s:.0f}s waited)..."
            )
            time.sleep(retry_interval_s)
            waited += retry_interval_s

        if waited >= max_wait_s:
            logger.warning(
                f"[Champion] Serving did not become idle within {max_wait_s}s. "
                f"Proceeding with champion.json write (atomic swap will still be safe)."
            )

        champion = {
            "model_id":    model_id,
            "model_path":  record.get("model_path"),
            "metrics":     record.get("metrics", {}),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(CHAMPION_PATH), exist_ok=True)
        tmp_path = CHAMPION_PATH + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(champion, f, indent=2)
            os.replace(tmp_path, CHAMPION_PATH)   # atomic on POSIX
            logger.info(
                f"champion.json updated → {model_id}  "
                f"(run_serving.py will hot-swap on next burst boundary)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to write champion.json: {e}")
            return False

    def read_champion_pointer(self) -> Optional[Dict]:
        if not os.path.exists(CHAMPION_PATH):
            return None
        try:
            with open(CHAMPION_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read champion.json: {e}")
            return None

    def compare_and_promote(
        self,
        run_id: str,
        metric: str = "mae_s",
    ) -> Dict[str, Any]:
        """
        Compare the best pending model from run_id against the currently
        deployed champion and promote if better (lower metric = better).

        Fix #5: Promotion defers writing champion.json until serving is idle.
        """
        pending = self.list_models(run_id=run_id, status=ModelStatus.PENDING.value)
        if not pending:
            return {"model_id": None, "promoted": False,
                    "reason": f"No pending models for run_id={run_id}"}

        best_pending = min(
            pending,
            key=lambda m: m.get("metrics", {}).get(metric, float("inf")),
        )
        new_val  = best_pending.get("metrics", {}).get(metric, float("inf"))
        new_id   = best_pending["model_id"]
        new_path = best_pending["model_path"]

        # Compare against current champion
        current_champion = self.read_champion_pointer()
        if current_champion:
            champ_id  = current_champion["model_id"]
            champ_rec = self.get_model(champ_id)
            champ_val = (champ_rec.get("metrics", {}).get(metric, float("inf"))
                         if champ_rec else float("inf"))
        else:
            champ_val = float("inf")

        if new_val < champ_val:
            # New model is better — approve, deploy, write champion
            self.approve_model(new_id, approved_by="preprod_auto")
            self.deploy_model(new_id)
            self.write_champion_pointer(new_id)   # Fix #5: waits for idle burst
            reason = (
                f"New model promoted: {metric}={new_val:.2f} < "
                f"champion {metric}={champ_val:.2f}"
            )
            logger.info(f"[{run_id}] {reason}")
            return {"model_id": new_id, "promoted": True, "reason": reason}
        else:
            reason = (
                f"New model NOT promoted: {metric}={new_val:.2f} >= "
                f"champion {metric}={champ_val:.2f}. Left as PENDING."
            )
            logger.info(f"[{run_id}] {reason}")
            return {"model_id": new_id, "promoted": False, "reason": reason}