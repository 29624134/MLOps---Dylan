"""
utils/model_registry.py
═══════════════════════════════════════════════════════════════════════════════
Central registry for all trained models.

Stores globally at model_registry/registry.json — not scoped to a run_id —
so all runs share a single registry.

Key responsibilities
────────────────────
1. Track model versions and metadata across all runs
2. Manage model approval workflow (pending → approved → deployed → archived)
3. Retrieve deployed models for serving
4. Maintain model lineage via metadata
5. Write champion.json pointer for atomic hot-swap (run_serving.py watches this)
6. Compare pending models against the current champion and promote if better
"""

import json
import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

# ── Champion pointer path — watched by run_serving.py for hot-swap ────────────
CHAMPION_PATH = os.path.join("model_registry", "champion.json")


class ModelStatus(Enum):
    """Model lifecycle states."""
    PENDING      = "pending"        # Just trained, awaiting review/comparison
    APPROVED     = "approved"       # Approved for production
    REJECTED     = "rejected"       # Rejected by domain expert
    DEPLOYED     = "deployed"       # Currently in production
    ARCHIVED     = "archived"       # Retired from production
    EXPERIMENTAL = "experimental"   # From external research


class ModelRegistry:
    """
    Central registry for all trained models.

    Stores globally at model_registry/registry.json — not scoped to a
    run_id — so all runs share a single registry, mirroring the
    WorkflowRegistry pattern.

    run_id is preserved for lineage by passing it in metadata when calling
    register_model(), and is filterable via list_models(run_id=...).
    """

    REGISTRY_FILENAME = "registry.json"

    def __init__(self, registry_path: Optional[str] = None):
        """
        Parameters
        ----------
        registry_path : str, optional
            Override the registry file path. Defaults to
            ``model_registry/registry.json`` relative to cwd.
        """
        if registry_path is None:
            registry_path = os.path.join("model_registry", self.REGISTRY_FILENAME)

        if not os.path.isabs(registry_path):
            registry_path = os.path.abspath(registry_path)

        self.registry_path = registry_path
        self.registry_dir  = os.path.dirname(registry_path)
        os.makedirs(self.registry_dir, exist_ok=True)

        logger.info(f"ModelRegistry initialised at: {self.registry_path}")

        if not os.path.exists(self.registry_path):
            logger.info("  Creating new model registry file.")
            self._initialize_registry()
        else:
            try:
                reg = self._load_registry()
                logger.info(
                    f"  Loaded existing registry with "
                    f"{len(reg.get('models', []))} model(s)."
                )
            except Exception as exc:
                logger.error(f"  Failed to load existing registry: {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _initialize_registry(self):
        initial = {
            "version":    "1.0",
            "created_at": datetime.now().isoformat(),
            "models":     [],
        }
        self._save_registry(initial)

    def _load_registry(self) -> Dict:
        with open(self.registry_path, "r") as fh:
            return json.load(fh)

    def _save_registry(self, registry: Dict):
        with open(self.registry_path, "w") as fh:
            json.dump(registry, fh, indent=2)

    def _generate_model_id(self, model_path: str, timestamp: str) -> str:
        unique = f"{model_path}_{timestamp}"
        return hashlib.md5(unique.encode()).hexdigest()[:12]

    def _find(self, registry: Dict, model_id: str) -> Optional[Dict]:
        for m in registry["models"]:
            if m["model_id"] == model_id:
                return m
        return None

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
        """
        Register a newly trained model.

        Parameters
        ----------
        model_path : str
            Path to the saved model file (.pt).
        model_type : str
            Model class name (e.g. 'RULNetModel').
        target_feature : str
            Feature being predicted (e.g. 'RUL_s').
        metrics : dict
            Performance metrics e.g. {'mae_s': 1200.0, 'mean_abs_pct': 0.12}.
        training_data_info : dict
            Info about training data (num files, window size, source, etc.).
        metadata : dict, optional
            Arbitrary extra fields. Pass run_id here for lineage:
            ``metadata={"run_id": run_id, "hyperparameters": {...}}``.

        Returns
        -------
        str — the generated model_id.
        """
        timestamp = datetime.now().isoformat()
        model_id  = self._generate_model_id(model_path, timestamp)

        record: Dict[str, Any] = {
            "model_id":           model_id,
            "model_path":         model_path,
            "model_type":         model_type,
            "target_feature":     target_feature,
            "status":             ModelStatus.PENDING.value,
            "metrics":            metrics,
            "training_data_info": training_data_info,
            "registered_at":      timestamp,
            "approved_at":        None,
            "approved_by":        None,
            "deployed_at":        None,
            "archived_at":        None,
            "metadata":           metadata or {},
        }

        registry = self._load_registry()
        registry["models"].append(record)
        self._save_registry(registry)

        logger.info(
            f"Registered model '{model_type}' -> '{target_feature}' "
            f"(id={model_id}, run={metadata.get('run_id', 'n/a') if metadata else 'n/a'})"
        )
        return model_id

    def approve_model(self, model_id: str, approved_by: str) -> bool:
        """Transition pending → approved."""
        registry = self._load_registry()
        record   = self._find(registry, model_id)
        if record is None:
            logger.error(f"approve_model: id '{model_id}' not found.")
            return False
        record["status"]      = ModelStatus.APPROVED.value
        record["approved_at"] = datetime.now().isoformat()
        record["approved_by"] = approved_by
        self._save_registry(registry)
        logger.info(f"Model {model_id} approved by '{approved_by}'.")
        return True

    def reject_model(self, model_id: str, rejected_by: str) -> bool:
        """Transition pending → rejected."""
        registry = self._load_registry()
        record   = self._find(registry, model_id)
        if record is None:
            logger.error(f"reject_model: id '{model_id}' not found.")
            return False
        record["status"]      = ModelStatus.REJECTED.value
        record["rejected_at"] = datetime.now().isoformat()
        record["rejected_by"] = rejected_by
        self._save_registry(registry)
        logger.info(f"Model {model_id} rejected by '{rejected_by}'.")
        return True

    def deploy_model(self, model_id: str) -> bool:
        """
        Transition approved → deployed.
        Automatically archives any previously deployed model for the same
        target_feature so only one deployed model exists per feature.
        """
        registry = self._load_registry()
        record   = self._find(registry, model_id)

        if record is None:
            logger.error(f"deploy_model: id '{model_id}' not found.")
            return False
        if record["status"] != ModelStatus.APPROVED.value:
            logger.error(
                f"deploy_model: model '{model_id}' is '{record['status']}', "
                f"not 'approved'."
            )
            return False

        now = datetime.now().isoformat()
        for m in registry["models"]:
            if (m["target_feature"] == record["target_feature"]
                    and m["status"] == ModelStatus.DEPLOYED.value
                    and m["model_id"] != model_id):
                m["status"]      = ModelStatus.ARCHIVED.value
                m["archived_at"] = now
                logger.info(f"  Archived previously deployed model: {m['model_id']}")

        record["status"]      = ModelStatus.DEPLOYED.value
        record["deployed_at"] = now
        self._save_registry(registry)
        logger.info(f"Model {model_id} deployed for '{record['target_feature']}'.")
        return True

    def archive_model(self, model_id: str) -> bool:
        """Manually archive a model."""
        registry = self._load_registry()
        record   = self._find(registry, model_id)
        if record is None:
            logger.error(f"archive_model: id '{model_id}' not found.")
            return False
        record["status"]      = ModelStatus.ARCHIVED.value
        record["archived_at"] = datetime.now().isoformat()
        self._save_registry(registry)
        logger.info(f"Model {model_id} archived.")
        return True

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_deployed_model(self, target_feature: str) -> Optional[Dict]:
        """
        Return the currently deployed model for a given target feature.
        Primary method used by live serving and the predict endpoint.
        Returns None if no deployed model exists.
        """
        registry = self._load_registry()
        for m in registry["models"]:
            if (m["target_feature"] == target_feature
                    and m["status"] == ModelStatus.DEPLOYED.value):
                return m
        logger.warning(f"No deployed model found for '{target_feature}'.")
        return None

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Return a model record by its ID."""
        registry = self._load_registry()
        return self._find(registry, model_id)

    def list_models(
        self,
        status:         Optional[str] = None,
        target_feature: Optional[str] = None,
        run_id:         Optional[str] = None,
    ) -> List[Dict]:
        """
        List model records with optional filters.

        Parameters
        ----------
        status : str, optional
            Filter by status string (e.g. 'pending', 'deployed').
        target_feature : str, optional
            Filter by target feature (e.g. 'RUL_s').
        run_id : str, optional
            Filter by run_id stored in metadata.
        """
        registry = self._load_registry()
        results  = registry["models"]

        if status:
            results = [m for m in results if m["status"] == status]
        if target_feature:
            results = [m for m in results if m["target_feature"] == target_feature]
        if run_id:
            results = [
                m for m in results
                if m.get("metadata", {}).get("run_id") == run_id
            ]
        return results

    def print_status(self, target_feature: Optional[str] = None):
        """Log a human-readable status summary."""
        models = self.list_models(target_feature=target_feature)
        logger.info("─" * 72)
        logger.info(f"{'MODEL REGISTRY':^72}")
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

    # ── Champion pointer — hot-swap support ───────────────────────────────────

    def write_champion_pointer(self, model_id: str) -> bool:
        """
        Write model_registry/champion.json pointing to the newly deployed model.

        Called automatically by compare_and_promote() after deploy_model().
        Can also be called manually after a manual deploy.

        run_serving.py watches this file for mtime changes between bursts.
        When detected it calls pipeline.reload_model() to hot-swap without
        missing a prediction.

        Uses write-then-rename (os.replace) for atomicity — the serving
        pipeline can never read a half-written file.

        Returns True on success.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.error(f"write_champion_pointer: model '{model_id}' not found.")
            return False

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
        """
        Read the current champion.json pointer.
        Returns the champion dict or None if no champion file exists.
        """
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

        Flow
        ────
        1. Find all pending models for this run_id
        2. Pick the one with the best (lowest) metric value
        3. Compare against the currently deployed model's metric
        4. If new model is better (or no champion exists):
               approve → deploy → write champion.json
           Else:
               leave as pending for manual review

        Parameters
        ----------
        run_id : str
            The training run whose pending models to compare.
        metric : str
            The metric to compare on (default: 'mae_s', lower is better).

        Returns
        -------
        dict with keys:
            promoted   : bool         — True if new model was deployed
            model_id   : str | None   — ID of the best pending model evaluated
            new_score  : float | None — new model's metric value
            old_score  : float | None — current champion's metric value (or None)
            reason     : str          — human-readable explanation of decision
        """
        pending = self.list_models(run_id=run_id, status="pending")
        if not pending:
            msg = f"No pending models found for run_id='{run_id}'."
            logger.warning(f"[Registry] compare_and_promote: {msg}")
            return {
                "promoted":  False,
                "model_id":  None,
                "new_score": None,
                "old_score": None,
                "reason":    msg,
            }

        # Pick best pending model by metric (lower is better)
        best      = min(
            pending,
            key=lambda m: m.get("metrics", {}).get(metric, float("inf"))
        )
        new_score    = best.get("metrics", {}).get(metric)
        new_metrics  = best.get("metrics", {})
        model_id     = best["model_id"]

        # Get currently deployed champion for this feature
        target       = best.get("target_feature", "RUL_s")
        current      = self.get_deployed_model(target)
        old_score    = current.get("metrics", {}).get(metric) if current else None
        old_metrics  = current.get("metrics", {}) if current else {}

        # Log full metric comparison across all 4 metrics
        logger.info(
            f"[Registry] ── Model Comparison ──────────────────────────────\n"
            f"  Metric      │ New Model          │ Current Champion\n"
            f"  ────────────┼────────────────────┼──────────────────\n"
            f"  mae_s       │ {str(round(new_metrics.get('mae_s', None), 1) if new_metrics.get('mae_s') is not None else 'None'):<18} │ "
            f"{str(round(old_metrics.get('mae_s', None), 1) if old_metrics.get('mae_s') is not None else 'None')}\n"
            f"  rmse_s      │ {str(round(new_metrics.get('rmse_s', None), 1) if new_metrics.get('rmse_s') is not None else 'None'):<18} │ "
            f"{str(round(old_metrics.get('rmse_s', None), 1) if old_metrics.get('rmse_s') is not None else 'None')}\n"
            f"  mape        │ {str(round(new_metrics.get('mape', None), 2) if new_metrics.get('mape') is not None else 'None'):<18} │ "
            f"{str(round(old_metrics.get('mape', None), 2) if old_metrics.get('mape') is not None else 'None')}\n"
        )

        # Decision logic
        if current is None:
            reason  = "No existing champion — promoting new model unconditionally."
            promote = True
        elif new_score is None:
            reason  = (
                f"New model '{model_id}' has no '{metric}' metric — "
                f"cannot compare. Left as PENDING for manual review."
            )
            promote = False
        elif new_score < old_score:
            reason  = (
                f"New model is BETTER: {metric} {old_score:.2f} → {new_score:.2f}. "
                f"Promoting to champion."
            )
            promote = True
        else:
            reason  = (
                f"New model is NOT better: new {metric}={new_score:.2f} >= "
                f"current {old_score:.2f}. Retaining existing champion."
            )
            promote = False

        logger.info(f"[Registry] Decision: {reason}")

        if promote:
            self.approve_model(model_id, approved_by="auto_compare_and_promote")
            self.deploy_model(model_id)
            self.write_champion_pointer(model_id)

        return {
            "promoted":    promote,
            "model_id":    model_id,
            "new_score":   new_score,
            "old_score":   old_score,
            "new_metrics": new_metrics,
            "old_metrics": old_metrics,
            "reason":      reason,
        }