"""
utils/model_registry.py
═══════════════════════════════════════════════════════════════════════════════
Central registry for all trained models.

Storage
───────
Primary  : MongoDB  phm_mlops.model_registry  (one document per model)
Fallback : model_registry/registry.json       (if MongoDB unavailable)

Follows the same MongoDB-primary / JSON-fallback pattern as WorkflowRegistry
so the two registries feel consistent.

Connection priority
───────────────────
1. Constructor kwargs  (mongo_uri, db_name)
2. Environment variables MONGO_URI / MONGO_DB
3. Defaults: mongodb://localhost:27017 / phm_mlops

Key responsibilities
────────────────────
1. Track model versions and metadata across all runs
2. Manage model approval workflow:
       pending → approved → deployed (champion) → archived (retired)
       pending → rejected (when not better than current champion)
3. Retrieve deployed models for serving
4. Maintain model lineage via metadata
5. Write champion.json pointer for atomic hot-swap (run_serving.py watches this)
6. Compare pending models against the current champion and promote if better

Metrics policy
──────────────
Only THREE metrics are recorded in the registry for every model:

    mae_s   — Mean Absolute Error in seconds
              Primary accuracy metric and the basis for champion selection.
    rmse_s  — Root Mean Square Error in seconds
              Penalises large prediction errors more heavily; important in a
              prognosis context where late RUL predictions are more
              consequential than small early errors.
    mcra    — Mean Cumulative Relative Accuracy
              Trajectory-based formulation (Lei et al., 2018). Aggregates
              relative accuracy across the full degradation trajectory,
              rewarding models that track decline consistently rather than
              only performing well at a single point.

Backward compatibility
──────────────────────
The full public API is unchanged — every caller (model_trainer.py,
run_preprod.py, orchestrator.py, API endpoints) works without modification.
If MongoDB is unavailable, the registry transparently falls back to the
original registry.json file so the system degrades gracefully.
═══════════════════════════════════════════════════════════════════════════════
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
    REJECTED     = "rejected"       # Rejected (not better than current champion, or denied)
    DEPLOYED     = "deployed"       # Currently in production (CHAMPION)
    ARCHIVED     = "archived"       # Retired from production
    EXPERIMENTAL = "experimental"   # From external research


# ─────────────────────────────────────────────────────────────────────────────
# Metric whitelist — ONLY these three are persisted in the registry
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_METRICS = ("mae_s", "rmse_s", "mcra")


def _filter_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the whitelisted metrics — silently drop everything else."""
    if not metrics:
        return {}
    return {k: metrics[k] for k in ALLOWED_METRICS if k in metrics and metrics[k] is not None}


class ModelRegistry:
    """
    Central registry for all trained models.

    Stores in MongoDB phm_mlops.model_registry (one document per model).
    Falls back to model_registry/registry.json if MongoDB is unavailable.

    run_id is preserved for lineage by passing it in metadata when calling
    register_model(), and is filterable via list_models(run_id=...).
    """

    REGISTRY_FILENAME = "registry.json"

    def __init__(
        self,
        registry_path: Optional[str] = None,
        mongo_uri:     Optional[str] = None,
        db_name:       Optional[str] = None,
    ):
        """
        Parameters
        ----------
        registry_path : str, optional
            Override the JSON fallback file path.
        mongo_uri : str, optional
            MongoDB connection string. Defaults to env MONGO_URI or
            mongodb://localhost:27017.
        db_name : str, optional
            MongoDB database name. Defaults to env MONGO_DB or phm_mlops.
        """
        self._mongo_uri = (
            mongo_uri
            or os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        )
        self._db_name = (
            db_name
            or os.environ.get("MONGO_DB", "phm_mlops")
        )
        self.registry_path = registry_path or os.path.join(
            "model_registry", self.REGISTRY_FILENAME
        )

        self._col      = None
        self._fallback = False

        try:
            col = self._collection()
            col.create_index("model_id",       name="idx_model_id", unique=True)
            col.create_index("status",         name="idx_status")
            col.create_index("target_feature", name="idx_target_feature")
            col.create_index(
                [("metadata.run_id", 1)],
                name="idx_run_id",
                sparse=True,
            )
            logger.info(
                f"ModelRegistry connected to MongoDB "
                f"→ {self._db_name}.model_registry"
            )
        except Exception as exc:
            logger.warning(
                f"ModelRegistry: MongoDB unavailable ({exc}). "
                f"Falling back to JSON at {self.registry_path}."
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
        self._col = client[self._db_name]["model_registry"]
        return self._col

    # ── Storage helpers ───────────────────────────────────────────────────────

    def _ensure_fallback_json(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as fh:
                json.dump(
                    {
                        "version":    "1.0",
                        "created_at": datetime.now().isoformat(),
                        "models":     [],
                    },
                    fh,
                    indent=2,
                )

    def _load_registry(self) -> Dict:
        """
        Load full registry — MongoDB primary, JSON fallback.
        Returns a dict with shape {"version": "...", "models": [...]}.
        """
        if not self._fallback:
            try:
                docs = list(self._collection().find({}))
                for d in docs:
                    d.pop("_id", None)
                return {"version": "1.0", "models": docs}
            except Exception as exc:
                logger.warning(
                    f"ModelRegistry: MongoDB read failed ({exc}). "
                    f"Using JSON fallback."
                )
        # JSON fallback
        with open(self.registry_path, "r") as fh:
            return json.load(fh)

    def _save_registry(self, registry: Dict):
        """
        Persist registry — MongoDB primary, JSON fallback.

        For MongoDB, each model record is upserted by model_id so this is
        safe to call repeatedly (idempotent).
        """
        if not self._fallback:
            try:
                col = self._collection()
                for record in registry.get("models", []):
                    col.replace_one(
                        {"model_id": record["model_id"]},
                        record,
                        upsert=True,
                    )
                return
            except Exception as exc:
                logger.warning(
                    f"ModelRegistry: MongoDB write failed ({exc}). "
                    f"Using JSON fallback."
                )
        # JSON fallback
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

        Only the three whitelisted metrics (mae_s, rmse_s, mcra) are stored;
        anything else passed in `metrics` is silently dropped.
        """
        timestamp = datetime.now().isoformat()
        model_id  = self._generate_model_id(model_path, timestamp)

        record: Dict[str, Any] = {
            "model_id":           model_id,
            "model_path":         model_path,
            "model_type":         model_type,
            "target_feature":     target_feature,
            "status":             ModelStatus.PENDING.value,
            "metrics":            _filter_metrics(metrics),
            "training_data_info": training_data_info,
            "registered_at":      timestamp,
            "approved_at":        None,
            "approved_by":        None,
            "deployed_at":        None,
            "archived_at":        None,
            "rejected_at":        None,
            "rejected_by":        None,
            "rejection_reason":   None,
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

    def reject_model(
        self,
        model_id:    str,
        rejected_by: str,
        reason:      Optional[str] = None,
    ) -> bool:
        """
        Transition pending → rejected.

        Use when the new model is NOT better than the current champion, or
        when a domain expert explicitly rejects it. The optional `reason`
        is stored on the record for audit traceability.
        """
        registry = self._load_registry()
        record   = self._find(registry, model_id)
        if record is None:
            logger.error(f"reject_model: id '{model_id}' not found.")
            return False
        record["status"]           = ModelStatus.REJECTED.value
        record["rejected_at"]      = datetime.now().isoformat()
        record["rejected_by"]      = rejected_by
        record["rejection_reason"] = reason or ""
        self._save_registry(registry)
        logger.info(
            f"Model {model_id} rejected by '{rejected_by}'"
            + (f" — {reason}" if reason else ".")
        )
        return True

    def deploy_model(self, model_id: str) -> bool:
        """
        Transition approved → deployed (becomes the CHAMPION).
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
                logger.info(
                    f"  Archived previously deployed (retired) model: {m['model_id']}"
                )

        record["status"]      = ModelStatus.DEPLOYED.value
        record["deployed_at"] = now
        self._save_registry(registry)
        logger.info(
            f"Model {model_id} deployed (CHAMPION) for '{record['target_feature']}'."
        )
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
        Return the currently deployed (champion) model for a target feature.
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
        """List model records with optional filters."""
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
        """
        Log a human-readable status summary.

        The deployed model is explicitly tagged as CHAMPION in the display,
        rejected models show as REJECTED, archived as ARCHIVED, etc — so the
        audit trail never leaves the question "which one is currently live?"
        ambiguous.
        """
        models = self.list_models(target_feature=target_feature)
        logger.info("─" * 88)
        logger.info(f"{'MODEL REGISTRY':^88}")
        logger.info("─" * 88)
        if not models:
            logger.info("  (empty)")
            logger.info("─" * 88)
            return

        # Sort: deployed (CHAMPION) first, then approved, pending, rejected, archived
        order = {
            ModelStatus.DEPLOYED.value: 0,
            ModelStatus.APPROVED.value: 1,
            ModelStatus.PENDING.value:  2,
            ModelStatus.REJECTED.value: 3,
            ModelStatus.ARCHIVED.value: 4,
        }
        sorted_models = sorted(
            models,
            key=lambda m: (order.get(m.get("status"), 9),
                           m.get("registered_at", "")),
        )

        for m in sorted_models:
            run_id    = m.get("metadata", {}).get("run_id", "n/a")
            status    = m.get("status", "?")
            # Tag the live champion explicitly so it's never ambiguous
            if status == ModelStatus.DEPLOYED.value:
                tag = "🏆 CHAMPION"
            elif status == ModelStatus.REJECTED.value:
                tag = "❌ REJECTED"
            elif status == ModelStatus.ARCHIVED.value:
                tag = "📦 ARCHIVED"
            elif status == ModelStatus.APPROVED.value:
                tag = "✅ APPROVED"
            elif status == ModelStatus.PENDING.value:
                tag = "⏳ PENDING"
            else:
                tag = status.upper()

            logger.info(
                f"  [{tag:<14}] "
                f"{m['model_type']} -> {m['target_feature']} "
                f"(id={m['model_id']}, run={run_id})"
            )
        logger.info("─" * 88)

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
        """Read the current champion.json pointer. Returns None if no file."""
        if not os.path.exists(CHAMPION_PATH):
            return None
        try:
            with open(CHAMPION_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read champion.json: {e}")
            return None

    # ── Compare & promote ─────────────────────────────────────────────────────

    def compare_and_promote(
        self,
        run_id: str,
        metric: str = "mae_s",
    ) -> Dict[str, Any]:
        """
        Compare the best pending model from run_id against the currently
        deployed champion and promote if better (lower metric = better).

        Decision logic
        ──────────────
        - No existing champion        → promote new model (approve+deploy).
        - New model better            → promote new model. Old champion is
                                        automatically archived by deploy_model().
        - New model NOT better        → REJECT new model (explicit rejection,
                                        not left as pending).
        - New model missing metric    → leave pending for manual review.
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
        best = min(
            pending,
            key=lambda m: m.get("metrics", {}).get(metric, float("inf"))
        )
        new_score   = best.get("metrics", {}).get(metric)
        new_metrics = best.get("metrics", {})
        model_id    = best["model_id"]

        # Get currently deployed champion for this feature
        target      = best.get("target_feature", "RUL_s")
        current     = self.get_deployed_model(target)
        old_score   = current.get("metrics", {}).get(metric) if current else None
        old_metrics = current.get("metrics", {}) if current else {}

        # Log metric comparison — only the three whitelisted metrics
        def _fmt(val, decimals=2):
            if val is None:
                return "N/A"
            try:
                return f"{float(val):.{decimals}f}"
            except (TypeError, ValueError):
                return str(val)

        logger.info(
            f"[Registry] ── Model Comparison ──────────────────────────────\n"
            f"  Metric        │ New Model          │ Current Champion\n"
            f"  ──────────────┼────────────────────┼──────────────────\n"
            f"  MAE_s  (s)    │ {_fmt(new_metrics.get('mae_s'),  1):<18} │ {_fmt(old_metrics.get('mae_s'),  1)}\n"
            f"  RMSE_s (s)    │ {_fmt(new_metrics.get('rmse_s'), 1):<18} │ {_fmt(old_metrics.get('rmse_s'), 1)}\n"
            f"  MCRA   (0-1)  │ {_fmt(new_metrics.get('mcra'),   4):<18} │ {_fmt(old_metrics.get('mcra'),   4)}\n"
        )

        # Decision logic
        if current is None:
            reason  = "No existing champion — promoting new model unconditionally."
            promote = True
            reject  = False
        elif new_score is None:
            reason  = (
                f"New model '{model_id}' has no '{metric}' metric — "
                f"cannot compare. Left as PENDING for manual review."
            )
            promote = False
            reject  = False
        elif new_score < old_score:
            reason  = (
                f"New model is BETTER: {metric} {old_score:.2f} → {new_score:.2f}. "
                f"Promoting to champion."
            )
            promote = True
            reject  = False
        else:
            reason  = (
                f"New model is NOT better: new {metric}={new_score:.2f} >= "
                f"current {old_score:.2f}. Rejecting new model; "
                f"current champion retained."
            )
            promote = False
            reject  = True

        logger.info(f"[Registry] Decision: {reason}")

        if promote:
            self.approve_model(model_id, approved_by="auto_compare_and_promote")
            self.deploy_model(model_id)
            self.write_champion_pointer(model_id)
        elif reject:
            self.reject_model(
                model_id,
                rejected_by="auto_compare_and_promote",
                reason=reason,
            )

        return {
            "promoted":    promote,
            "rejected":    reject,
            "model_id":    model_id,
            "new_score":   new_score,
            "old_score":   old_score,
            "new_metrics": new_metrics,
            "old_metrics": old_metrics,
            "reason":      reason,
        }