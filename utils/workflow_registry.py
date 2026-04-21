"""
utils/workflow_registry.py
═══════════════════════════════════════════════════════════════════════════════
MongoDB-backed Workflow Registry.

Replaces the flat registry.json storage with a MongoDB collection while
keeping the exact same public API — the Scheduler, CI pipeline, and API
endpoints all work without any changes.

MongoDB layout
──────────────
database   : phm_mlops   (or configured db_name)
collection : workflow_registry

Each document is one workflow record, with _id = workflow_id (str).
The full workflow definition (DAG/steps dict) is embedded directly in the
document — no separate file needed.

Connection
──────────
Priority order:
  1. Constructor kwargs  (mongo_uri, db_name)
  2. Environment variables MONGO_URI / MONGO_DB
  3. Defaults: mongodb://localhost:27017 / phm_mlops

Backward compatibility
──────────────────────
If MongoDB is unavailable the registry falls back to the original JSON file
so the system degrades gracefully rather than crashing.
"""

import hashlib
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS ENUM  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowStatus(Enum):
    """Workflow lifecycle states.

    Flow:
        pending  ->  approved  ->  active  ->  deprecated
                 ->  rejected
                 ->  archived  (manual retirement)
    """
    PENDING    = "pending"
    APPROVED   = "approved"
    ACTIVE     = "active"
    REJECTED   = "rejected"
    DEPRECATED = "deprecated"
    ARCHIVED   = "archived"


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowRegistry:
    """
    Central registry for versioned, CI-approved workflows — backed by MongoDB.

    Mirrors the ModelRegistry pattern so the two registries feel consistent.

    Key responsibilities
    ───────────────────
    1. Store versioned workflow definitions (steps/DAG from workflow.yaml).
    2. Track lifecycle status: pending → approved → active → deprecated.
    3. Provide get_active_workflow() for the Scheduler.
    4. Maintain full lineage: who approved, when, which git hash.

    Public API is identical to the previous JSON-based registry.
    """

    COLLECTION   = "workflow_registry"
    FALLBACK_JSON = os.path.join("workflow_registry", "registry.json")

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name:   Optional[str] = None,
        # Legacy kwarg kept for backward compat — ignored when MongoDB is used
        registry_path: Optional[str] = None,
    ):
        self._mongo_uri = (
            mongo_uri
            or os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        )
        self._db_name = (
            db_name
            or os.environ.get("MONGO_DB", "phm_mlops")
        )
        self._col      = None
        self._fallback = False

        try:
            col = self._collection()
            # Ensure useful indexes (idempotent)
            col.create_index("status",        name="idx_status")
            col.create_index("workflow_name", name="idx_workflow_name")
            col.create_index(
                [("workflow_name", 1), ("status", 1)],
                name="idx_name_status",
            )
            col.create_index("git_hash", name="idx_git_hash", sparse=True)
            logger.info(
                f"WorkflowRegistry connected to MongoDB "
                f"→ {self._db_name}.{self.COLLECTION}"
            )
        except Exception as exc:
            logger.warning(
                f"WorkflowRegistry: MongoDB unavailable ({exc}). "
                f"Falling back to JSON at {self.FALLBACK_JSON}."
            )
            self._fallback = True
            self._ensure_fallback_json()

    # ── Connection ────────────────────────────────────────────────────────────

    def _collection(self):
        if self._col is not None:
            return self._col
        from pymongo import MongoClient
        client    = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        self._col = client[self._db_name][self.COLLECTION]
        return self._col

    # ── Fallback JSON helpers ─────────────────────────────────────────────────

    def _ensure_fallback_json(self):
        import json
        os.makedirs(os.path.dirname(self.FALLBACK_JSON), exist_ok=True)
        if not os.path.exists(self.FALLBACK_JSON):
            with open(self.FALLBACK_JSON, "w") as fh:
                json.dump(
                    {"version": "1.0", "created_at": datetime.now().isoformat(), "workflows": []},
                    fh, indent=2,
                )

    def _fallback_load(self) -> List[Dict]:
        import json
        with open(self.FALLBACK_JSON) as fh:
            return json.load(fh).get("workflows", [])

    def _fallback_save(self, workflows: List[Dict]):
        import json
        with open(self.FALLBACK_JSON) as fh:
            data = json.load(fh)
        data["workflows"] = workflows
        with open(self.FALLBACK_JSON, "w") as fh:
            json.dump(data, fh, indent=2)

    # ── ID generation ─────────────────────────────────────────────────────────

    @staticmethod
    def _generate_workflow_id(workflow_name: str, version: str, timestamp: str) -> str:
        unique = f"{workflow_name}_{version}_{timestamp}"
        return hashlib.md5(unique.encode()).hexdigest()[:12]

    # ── Internal MongoDB helpers ───────────────────────────────────────────────

    def _insert(self, record: Dict):
        doc = {**record, "_id": record["workflow_id"]}
        self._collection().insert_one(doc)

    def _update(self, workflow_id: str, updates: Dict):
        self._collection().update_one({"_id": workflow_id}, {"$set": updates})

    def _find_one_by_id(self, workflow_id: str) -> Optional[Dict]:
        doc = self._collection().find_one({"_id": workflow_id})
        return self._strip(doc)

    def _find_many(self, query: Dict) -> List[Dict]:
        return [self._strip(d) for d in self._collection().find(query)]

    @staticmethod
    def _strip(doc: Optional[Dict]) -> Optional[Dict]:
        if doc is None:
            return None
        doc = dict(doc)
        doc.pop("_id", None)
        return doc

    # ── Public API ────────────────────────────────────────────────────────────

    def register_workflow(
        self,
        workflow_name: str,
        version:       str,
        definition:    Dict[str, Any],
        trigger:       Optional[Dict]  = None,
        git_hash:      Optional[str]   = None,
        environment:   Optional[Dict]  = None,
        metadata:      Optional[Dict]  = None,
    ) -> str:
        """
        Register a new workflow version.

        Parameters
        ----------
        workflow_name : str
            Logical name (e.g. 'rul_prediction').
        version : str
            SemVer string (e.g. '1.3.0').
        definition : dict
            Full workflow DAG / step list from workflow.yaml.
        trigger : dict, optional
            Trigger config e.g. {'type': 'schedule', 'cron': '0 */6 * * *'}.
        git_hash : str, optional
            Git commit hash for lineage.
        environment : dict, optional
            Runtime environment snapshot e.g. {'python': '3.11'}.
        metadata : dict, optional
            Arbitrary extra fields.

        Returns
        -------
        str — generated workflow_id.
        """
        timestamp   = datetime.now().isoformat()
        workflow_id = self._generate_workflow_id(workflow_name, version, timestamp)

        record = {
            "workflow_id":   workflow_id,
            "workflow_name": workflow_name,
            "version":       version,
            "definition":    definition,
            "trigger":       trigger    or {},
            "git_hash":      git_hash   or "",
            "environment":   environment or {},
            "metadata":      metadata   or {},
            "status":        WorkflowStatus.PENDING.value,
            "registered_at": timestamp,
        }

        if self._fallback:
            workflows = self._fallback_load()
            workflows.append(record)
            self._fallback_save(workflows)
        else:
            self._insert(record)

        logger.info(
            f"Workflow registered: {workflow_id} "
            f"({workflow_name} v{version}, status=pending)"
        )
        return workflow_id

    def approve_workflow(self, workflow_id: str, approved_by: str = "ci-pipeline") -> bool:
        """Transition pending → approved."""
        updates = {
            "status":      WorkflowStatus.APPROVED.value,
            "approved_at": datetime.now().isoformat(),
            "approved_by": approved_by,
        }
        return self._transition(
            workflow_id, WorkflowStatus.PENDING, updates, "approved"
        )

    def reject_workflow(self, workflow_id: str, rejected_by: str = "ci-pipeline") -> bool:
        """Transition pending → rejected."""
        updates = {
            "status":      WorkflowStatus.REJECTED.value,
            "rejected_at": datetime.now().isoformat(),
            "rejected_by": rejected_by,
        }
        return self._transition(
            workflow_id, WorkflowStatus.PENDING, updates, "rejected"
        )

    def activate_workflow(self, workflow_id: str) -> bool:
        """
        Transition approved → active.
        Automatically deprecates any currently active version of the same
        workflow_name so only one is active at a time.
        """
        record = self.get_workflow(workflow_id)
        if record is None:
            logger.error(f"activate_workflow: id '{workflow_id}' not found.")
            return False
        if record["status"] != WorkflowStatus.APPROVED.value:
            logger.error(
                f"activate_workflow: workflow '{workflow_id}' is "
                f"'{record['status']}', not 'approved'."
            )
            return False

        now  = datetime.now().isoformat()
        name = record["workflow_name"]

        # Deprecate any currently active version of this workflow
        if not self._fallback:
            self._collection().update_many(
                {"workflow_name": name, "status": WorkflowStatus.ACTIVE.value},
                {"$set": {"status": WorkflowStatus.DEPRECATED.value, "deprecated_at": now}},
            )
        else:
            workflows = self._fallback_load()
            for w in workflows:
                if w["workflow_name"] == name and w["status"] == WorkflowStatus.ACTIVE.value:
                    w["status"]        = WorkflowStatus.DEPRECATED.value
                    w["deprecated_at"] = now
            self._fallback_save(workflows)

        updates = {"status": WorkflowStatus.ACTIVE.value, "activated_at": now}
        result  = self._transition(workflow_id, WorkflowStatus.APPROVED, updates, "active")
        if result:
            logger.info(f"Workflow {workflow_id} is now active for '{name}'.")
        return result

    def deprecate_workflow(self, workflow_id: str) -> bool:
        """Manually deprecate an active workflow."""
        updates = {
            "status":        WorkflowStatus.DEPRECATED.value,
            "deprecated_at": datetime.now().isoformat(),
        }
        return self._transition(
            workflow_id, WorkflowStatus.ACTIVE, updates, "deprecated"
        )

    def archive_workflow(self, workflow_id: str) -> bool:
        """Retire a workflow (any terminal state → archived)."""
        record = self.get_workflow(workflow_id)
        if record is None:
            logger.error(f"archive_workflow: id '{workflow_id}' not found.")
            return False
        updates = {
            "status":      WorkflowStatus.ARCHIVED.value,
            "archived_at": datetime.now().isoformat(),
        }
        if not self._fallback:
            self._update(workflow_id, updates)
        else:
            workflows = self._fallback_load()
            for w in workflows:
                if w["workflow_id"] == workflow_id:
                    w.update(updates)
            self._fallback_save(workflows)
        logger.info(f"Archived workflow {workflow_id}.")
        return True

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active_workflow(self, workflow_name: str) -> Optional[Dict]:
        """
        Return the currently active workflow for ``workflow_name``.
        Primary method called by the Scheduler (step 3 in the architecture).
        Returns None if no active version exists.
        """
        if self._fallback:
            for w in self._fallback_load():
                if w["workflow_name"] == workflow_name and w["status"] == WorkflowStatus.ACTIVE.value:
                    return w
            logger.warning(f"No active workflow found for '{workflow_name}'.")
            return None

        doc = self._collection().find_one({
            "workflow_name": workflow_name,
            "status":        WorkflowStatus.ACTIVE.value,
        })
        if doc is None:
            logger.warning(f"No active workflow found for '{workflow_name}'.")
        return self._strip(doc)

    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Return a workflow record by its ID."""
        if self._fallback:
            for w in self._fallback_load():
                if w["workflow_id"] == workflow_id:
                    return w
            return None
        return self._find_one_by_id(workflow_id)

    def get_latest_approved(self, workflow_name: str) -> Optional[Dict]:
        """Return the most recently approved (not yet active) workflow version."""
        if self._fallback:
            candidates = [
                w for w in self._fallback_load()
                if w["workflow_name"] == workflow_name
                and w["status"] == WorkflowStatus.APPROVED.value
            ]
            return max(candidates, key=lambda w: w["registered_at"]) if candidates else None

        docs = list(self._collection().find(
            {"workflow_name": workflow_name, "status": WorkflowStatus.APPROVED.value},
        ).sort("registered_at", -1).limit(1))
        return self._strip(docs[0]) if docs else None

    def list_workflows(
        self,
        workflow_name: Optional[str] = None,
        status:        Optional[str] = None,
    ) -> List[Dict]:
        """
        List workflow records with optional filters.

        Parameters
        ----------
        workflow_name : str, optional
            Filter by logical workflow name.
        status : str, optional
            Filter by status string (e.g. 'approved').
        """
        if self._fallback:
            results = self._fallback_load()
            if workflow_name:
                results = [w for w in results if w["workflow_name"] == workflow_name]
            if status:
                results = [w for w in results if w["status"] == status]
            return results

        query: Dict[str, Any] = {}
        if workflow_name:
            query["workflow_name"] = workflow_name
        if status:
            query["status"] = status
        return self._find_many(query)

    def print_status(self, workflow_name: Optional[str] = None):
        """Log a human-readable status summary."""
        workflows = self.list_workflows(workflow_name=workflow_name)
        logger.info("─" * 72)
        logger.info(f"{'WORKFLOW REGISTRY':^72}")
        logger.info("─" * 72)
        if not workflows:
            logger.info("  (empty)")
        for w in workflows:
            logger.info(
                f"  [{w['status'].upper():10}] "
                f"{w['workflow_name']} v{w['version']} "
                f"(id={w['workflow_id']}, git={w.get('git_hash', 'n/a')[:8]})"
            )
        logger.info("─" * 72)

    # ── Internal transition helper ─────────────────────────────────────────────

    def _transition(
        self,
        workflow_id: str,
        from_status: Optional[WorkflowStatus],
        updates:     Dict,
        label:       str,
    ) -> bool:
        if self._fallback:
            workflows = self._fallback_load()
            for w in workflows:
                if w["workflow_id"] == workflow_id:
                    if from_status and w["status"] != from_status.value:
                        logger.error(
                            f"_transition({label}): workflow '{workflow_id}' is "
                            f"'{w['status']}', expected '{from_status.value}'."
                        )
                        return False
                    w.update(updates)
                    self._fallback_save(workflows)
                    logger.info(f"Workflow {workflow_id} → {label}.")
                    return True
            logger.error(f"_transition({label}): id '{workflow_id}' not found.")
            return False

        record = self._find_one_by_id(workflow_id)
        if record is None:
            logger.error(f"_transition({label}): id '{workflow_id}' not found.")
            return False
        if from_status and record["status"] != from_status.value:
            logger.error(
                f"_transition({label}): workflow '{workflow_id}' is "
                f"'{record['status']}', expected '{from_status.value}'."
            )
            return False
        self._update(workflow_id, updates)
        logger.info(f"Workflow {workflow_id} → {label}.")
        return True