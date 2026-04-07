import json
import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW STATUS ENUM
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowStatus(Enum):
    """Workflow lifecycle states.

    Flow:
        pending  ->  approved  ->  active  ->  deprecated
                 ->  rejected
                 ->  archived  (manual retirement)
    """
    PENDING    = "pending"     # Pushed to registry, awaiting CI approval
    APPROVED   = "approved"    # CI passed; ready to be activated by Scheduler
    ACTIVE     = "active"      # Currently served to the Scheduler
    REJECTED   = "rejected"    # CI/domain expert rejected this version
    DEPRECATED = "deprecated"  # Superseded by a newer active version
    ARCHIVED   = "archived"    # Manually retired; kept for lineage/audit


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowRegistry:
    """
    Central registry for versioned, CI-approved workflows.

    Mirrors the ModelRegistry pattern so the two registries feel consistent.

    Key responsibilities
    -------------------
    1. Store versioned workflow definitions (steps/DAG from workflow.yaml).
    2. Track lifecycle status: pending → approved → active → deprecated.
    3. Provide ``get_active_workflow()`` for the Scheduler (analogous to
       ``get_deployed_model()`` in ModelRegistry).
    4. Maintain full lineage: who approved, when, which git hash.

    Storage layout
    --------------
    workflow_registry/
        registry.json          # master index of all workflow records

    Usage
    -----
    >>> reg = WorkflowRegistry()
    >>> wf_id = reg.register_workflow(
    ...     workflow_name="rul_prediction",
    ...     version="1.3.0",
    ...     definition=workflow_dict,
    ...     trigger={"type": "schedule", "cron": "0 */6 * * *"},
    ...     git_hash="abc1234",
    ...     environment={"python": "3.11", "torch": "2.2.0"},
    ... )
    >>> reg.approve_workflow(wf_id, approved_by="ci-pipeline")
    >>> reg.activate_workflow(wf_id)
    >>> active = reg.get_active_workflow("rul_prediction")
    """

    REGISTRY_FILENAME = "registry.json"

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = os.path.join("workflow_registry", self.REGISTRY_FILENAME)

        if not os.path.isabs(registry_path):
            registry_path = os.path.abspath(registry_path)

        self.registry_path = registry_path
        self.registry_dir  = os.path.dirname(registry_path)
        os.makedirs(self.registry_dir, exist_ok=True)

        logger.info(f"WorkflowRegistry initialised at: {self.registry_path}")

        if not os.path.exists(self.registry_path):
            logger.info("  Creating new workflow registry file.")
            self._initialise_registry()
        else:
            try:
                reg = self._load()
                logger.info(f"  Loaded existing registry with "
                            f"{len(reg.get('workflows', []))} workflow(s).")
            except Exception as exc:
                logger.error(f"  Failed to load existing registry: {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _initialise_registry(self):
        initial: Dict[str, Any] = {
            "version":    "1.0",
            "created_at": datetime.now().isoformat(),
            "workflows":  [],
        }
        self._save(initial)

    def _load(self) -> Dict:
        with open(self.registry_path, "r") as fh:
            return json.load(fh)

    def _save(self, registry: Dict):
        with open(self.registry_path, "w") as fh:
            json.dump(registry, fh, indent=2)

    def _generate_workflow_id(self, workflow_name: str, version: str) -> str:
        unique = f"{workflow_name}_{version}_{datetime.now().isoformat()}"
        return hashlib.md5(unique.encode()).hexdigest()[:12]

    # ── Public API ────────────────────────────────────────────────────────────

    def register_workflow(
        self,
        workflow_name:  str,
        version:        str,
        definition:     Dict[str, Any],
        trigger:        Optional[Dict[str, Any]] = None,
        git_hash:       Optional[str] = None,
        environment:    Optional[Dict[str, str]] = None,
        metadata:       Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new workflow version received from CI.

        Parameters
        ----------
        workflow_name : str
            Logical workflow name (e.g. ``"rul_prediction"``).
        version : str
            Semantic version string (e.g. ``"1.3.0"``).
        definition : dict
            Full workflow definition — typically the parsed YAML content
            (steps, depends_on, config blocks, etc.).
        trigger : dict, optional
            Trigger conditions, e.g. ``{"type": "schedule", "cron": "0 */6 * * *"}``
            or ``{"type": "event", "topic": "sensor.data.arrived"}``.
        git_hash : str, optional
            Git commit hash from CI for traceability.
        environment : dict, optional
            Runtime environment info, e.g. ``{"python": "3.11", "torch": "2.2.0"}``.
        metadata : dict, optional
            Any extra key/value pairs to store alongside the record.

        Returns
        -------
        str
            The generated ``workflow_id``.
        """
        timestamp   = datetime.now().isoformat()
        workflow_id = self._generate_workflow_id(workflow_name, version)

        record: Dict[str, Any] = {
            "workflow_id":   workflow_id,
            "workflow_name": workflow_name,
            "version":       version,
            "status":        WorkflowStatus.PENDING.value,
            "definition":    definition,
            "trigger":       trigger or {},
            "git_hash":      git_hash,
            "environment":   environment or {},
            "registered_at": timestamp,
            "approved_at":   None,
            "approved_by":   None,
            "activated_at":  None,
            "deprecated_at": None,
            "lineage": {
                "parent_version": self._get_latest_version(workflow_name),
            },
            "metadata": metadata or {},
        }

        registry = self._load()
        registry["workflows"].append(record)
        self._save(registry)

        logger.info(f"Registered workflow '{workflow_name}' v{version} "
                    f"(id={workflow_id}, status=pending)")
        return workflow_id

    # ── Lifecycle transitions ─────────────────────────────────────────────────

    def approve_workflow(self, workflow_id: str, approved_by: str) -> bool:
        """Transition pending → approved (typically called by CI pipeline)."""
        return self._transition(
            workflow_id,
            from_status=WorkflowStatus.PENDING,
            to_status=WorkflowStatus.APPROVED,
            update={
                "approved_at": datetime.now().isoformat(),
                "approved_by": approved_by,
            },
            log_msg=f"Approved by '{approved_by}'",
        )

    def reject_workflow(self, workflow_id: str, rejected_by: str,
                        reason: str = "") -> bool:
        """Transition pending → rejected."""
        return self._transition(
            workflow_id,
            from_status=WorkflowStatus.PENDING,
            to_status=WorkflowStatus.REJECTED,
            update={
                "rejected_at":  datetime.now().isoformat(),
                "rejected_by":  rejected_by,
                "reject_reason": reason,
            },
            log_msg=f"Rejected by '{rejected_by}': {reason}",
        )

    def activate_workflow(self, workflow_id: str) -> bool:
        """
        Transition approved → active.

        Any currently active workflow for the same ``workflow_name`` is
        automatically deprecated first, ensuring only one active version
        exists per workflow name.
        """
        registry = self._load()
        record   = self._find(registry, workflow_id)

        if record is None:
            logger.error(f"activate_workflow: id '{workflow_id}' not found.")
            return False

        if record["status"] != WorkflowStatus.APPROVED.value:
            logger.error(f"activate_workflow: workflow '{workflow_id}' is "
                         f"'{record['status']}', not 'approved'.")
            return False

        # Deprecate any currently active version for the same workflow
        now = datetime.now().isoformat()
        for wf in registry["workflows"]:
            if (wf["workflow_name"] == record["workflow_name"]
                    and wf["status"] == WorkflowStatus.ACTIVE.value
                    and wf["workflow_id"] != workflow_id):
                wf["status"]        = WorkflowStatus.DEPRECATED.value
                wf["deprecated_at"] = now
                logger.info(f"  Deprecated previous active workflow: "
                            f"{wf['workflow_id']} v{wf['version']}")

        record["status"]      = WorkflowStatus.ACTIVE.value
        record["activated_at"] = now
        self._save(registry)

        logger.info(f"Activated workflow '{record['workflow_name']}' "
                    f"v{record['version']} (id={workflow_id})")
        return True

    def deprecate_workflow(self, workflow_id: str) -> bool:
        """Manually transition active → deprecated."""
        return self._transition(
            workflow_id,
            from_status=WorkflowStatus.ACTIVE,
            to_status=WorkflowStatus.DEPRECATED,
            update={"deprecated_at": datetime.now().isoformat()},
            log_msg="Manually deprecated",
        )

    def archive_workflow(self, workflow_id: str) -> bool:
        """Retire a workflow (any terminal state → archived)."""
        registry = self._load()
        record   = self._find(registry, workflow_id)
        if record is None:
            logger.error(f"archive_workflow: id '{workflow_id}' not found.")
            return False
        record["status"]      = WorkflowStatus.ARCHIVED.value
        record["archived_at"] = datetime.now().isoformat()
        self._save(registry)
        logger.info(f"Archived workflow {workflow_id}")
        return True

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active_workflow(self, workflow_name: str) -> Optional[Dict]:
        """
        Return the currently active workflow for ``workflow_name``.

        This is the primary method called by the Scheduler (step 3 in the
        architecture diagram).

        Returns ``None`` if no active version exists.
        """
        registry = self._load()
        for wf in registry["workflows"]:
            if (wf["workflow_name"] == workflow_name
                    and wf["status"] == WorkflowStatus.ACTIVE.value):
                return wf
        logger.warning(f"No active workflow found for '{workflow_name}'.")
        return None

    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Return a workflow record by its ID."""
        registry = self._load()
        return self._find(registry, workflow_id)

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
            Filter by status string (e.g. ``"approved"``).
        """
        registry = self._load()
        results  = registry["workflows"]

        if workflow_name:
            results = [w for w in results if w["workflow_name"] == workflow_name]
        if status:
            results = [w for w in results if w["status"] == status]

        return results

    def get_latest_approved(self, workflow_name: str) -> Optional[Dict]:
        """Return the most recently approved (not yet active) workflow version."""
        candidates = self.list_workflows(
            workflow_name=workflow_name,
            status=WorkflowStatus.APPROVED.value,
        )
        if not candidates:
            return None
        return max(candidates, key=lambda w: w.get("approved_at") or "")

    def print_status(self, workflow_name: Optional[str] = None):
        """Log a human-readable status table (useful for debugging)."""
        workflows = self.list_workflows(workflow_name=workflow_name)
        logger.info("─" * 72)
        logger.info(f"{'WORKFLOW REGISTRY':^72}")
        logger.info("─" * 72)
        if not workflows:
            logger.info("  (empty)")
        for wf in workflows:
            logger.info(
                f"  [{wf['status'].upper():10}] "
                f"{wf['workflow_name']} v{wf['version']} "
                f"(id={wf['workflow_id']}, "
                f"git={wf.get('git_hash', 'n/a')})"
            )
        logger.info("─" * 72)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find(self, registry: Dict, workflow_id: str) -> Optional[Dict]:
        for wf in registry["workflows"]:
            if wf["workflow_id"] == workflow_id:
                return wf
        return None

    def _transition(
        self,
        workflow_id:  str,
        from_status:  WorkflowStatus,
        to_status:    WorkflowStatus,
        update:       Dict,
        log_msg:      str,
    ) -> bool:
        registry = self._load()
        record   = self._find(registry, workflow_id)

        if record is None:
            logger.error(f"Transition failed: id '{workflow_id}' not found.")
            return False

        if record["status"] != from_status.value:
            logger.error(
                f"Transition failed: workflow '{workflow_id}' is "
                f"'{record['status']}', expected '{from_status.value}'."
            )
            return False

        record["status"] = to_status.value
        record.update(update)
        self._save(registry)

        logger.info(f"Workflow {workflow_id}: {from_status.value} → "
                    f"{to_status.value}. {log_msg}")
        return True

    def _get_latest_version(self, workflow_name: str) -> Optional[str]:
        """Return the version string of the most recently registered workflow."""
        existing = self.list_workflows(workflow_name=workflow_name)
        if not existing:
            return None
        latest = max(existing, key=lambda w: w["registered_at"])
        return latest["version"]