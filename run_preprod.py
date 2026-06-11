"""
run_preprod.py
═══════════════════════════════════════════════════════════════════════════════
Pre-Production Entry Point — group-scoped retraining.

Triggered AFTER a maintenance worker confirms a fault on the dashboard.
Retrains ONLY the model for the bearing group that had the fault confirmed.
Other groups' models and serving pipelines are completely unaffected.

Flow
────
1. Validate confirmed fault data exists in FS Mirrored for this group
2. Retrain the group-specific model via WorkflowExecutor.run_training_only(group)
3. Compare new model vs the group's current champion (lower MAE_s = better)
4. If better → approve + deploy in registry (auto-archives old champion)
              → write group champion file atomically
              → run_serving.py for that group detects the change and hot-swaps
5. If not better → REJECT new model in the registry (explicit, audit-traceable)
                 → current champion retained, no champion file rewrite

Status outcomes in the Model Registry
─────────────────────────────────────
After a preprod run the new model is always in one of:
    deployed  — the new champion (and old champion is archived)
    rejected  — explicitly rejected because not better than current champion
    pending   — only if metric was missing (data error → manual review needed)

Champion files (one per group):
    model_registry/champion_bearing1.json  ← Group 1
    model_registry/champion_bearing2.json  ← Group 2
    model_registry/champion_bearing3.json  ← Group 3

Usage
─────
    # Triggered automatically by the API after fault confirm:
    python run_preprod.py --run_id <run_id> --group 1

    # Manual trigger:
    python run_preprod.py --group 2

    # Dry run (compare only, do not promote or reject):
    python run_preprod.py --group 1 --dry_run
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PREPROD] %(levelname)s — %(message)s",
)
logger = logging.getLogger("run_preprod")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MONGO_URI  = "mongodb://localhost:27017"
DEFAULT_DB_NAME    = "phm_mlops"
MODEL_REGISTRY_DIR = "model_registry"
COMPARISON_METRIC  = "mae_s"   # lower is better


def _champion_path(group: str) -> str:
    """Return the champion file path for a given group."""
    return os.path.join(MODEL_REGISTRY_DIR, f"champion_bearing{group}.json")


# ─────────────────────────────────────────────────────────────────────────────
# Champion pointer management
# ─────────────────────────────────────────────────────────────────────────────

def write_champion(
    group: str, model_id: str, model_path: str, metrics: dict
) -> None:
    """
    Atomically write the group-specific champion pointer file.

    run_serving.py watches this file for changes between bursts and reloads
    the model when it detects a new champion (hot-swap).

    Uses write-then-rename for atomicity — no partial reads possible.
    """
    path     = _champion_path(group)
    tmp_path = path + ".tmp"
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

    champion = {
        "group":       group,
        "model_id":    model_id,
        "model_path":  model_path,
        "metrics":     metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(tmp_path, "w") as f:
        json.dump(champion, f, indent=2)
    os.replace(tmp_path, path)   # atomic on POSIX
    logger.info(f"[Champion] champion_bearing{group}.json written → {model_id}")


def read_champion(group: str) -> Optional[dict]:
    """Return the current champion dict for a group, or None if it doesn't exist."""
    path = _champion_path(group)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison
# ─────────────────────────────────────────────────────────────────────────────

def is_new_model_better(
    new_metrics:      dict,
    champion_metrics: dict,
    metric:           str = COMPARISON_METRIC,
) -> bool:
    """Returns True if the new model is better. Lower is better for MAE/RMSE."""
    new_score      = new_metrics.get(metric)
    champion_score = champion_metrics.get(metric) if champion_metrics else None

    if new_score is None:
        logger.warning(f"New model has no '{metric}' metric — cannot compare.")
        return False

    if champion_score is None:
        logger.info("No champion metric to compare against — new model wins by default.")
        return True

    logger.info(
        f"[Compare] {metric}: new={new_score:.2f}  vs  champion={champion_score:.2f}"
    )
    return new_score < champion_score


# ─────────────────────────────────────────────────────────────────────────────
# Main Pre-Production flow
# ─────────────────────────────────────────────────────────────────────────────

def run_preprod(
    group:     str,
    run_id:    Optional[str],
    mongo_uri: str,
    db_name:   str,
    dry_run:   bool,
) -> bool:
    """
    Full pre-production flow for a single bearing group.

    Parameters
    ----------
    group     : str  — "1", "2", or "3"
    run_id    : str  — unique identifier (auto-generated if None)
    mongo_uri : str  — MongoDB connection string
    db_name   : str  — database name
    dry_run   : bool — compare only, do not write champion file or reject
    """
    run_id = run_id or f"preprod_g{group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 60)
    logger.info(f"  PHM MLOps — Pre-Production Retraining (Group {group})")
    logger.info("=" * 60)
    logger.info(f"  Run ID        : {run_id}")
    logger.info(f"  Group         : {group}")
    logger.info(f"  Champion file : {_champion_path(group)}")
    logger.info(f"  MongoDB       : {mongo_uri} / {db_name}")
    logger.info(f"  Dry run       : {dry_run}")
    logger.info("=" * 60)

    # ── Step 1: Verify confirmed fault data exists for this group ─────────────
    logger.info(f"\n[Phase 1] Checking confirmed fault data for Group {group}...")
    try:
        from pymongo import MongoClient
        from utils.db_collections import COL_FEATURE_STORE_MIRRORED

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db     = client[db_name]
        col    = db[COL_FEATURE_STORE_MIRRORED]

        from orchestrator import BearingRegistry
        reg = BearingRegistry("config/bearings.json")

        confirmed_in_group = [
            b["name"] for b in reg.live_bearings_in_group(group)
            if b.get("status") == "confirmed"
        ]

        if not confirmed_in_group:
            logger.warning(
                f"  No confirmed bearings found for Group {group} in bearings.json. "
                f"Retraining will use base train set only."
            )
        else:
            for bname in confirmed_in_group:
                count = col.count_documents({"dataset_id": bname})
                logger.info(
                    f"  [{bname}] {count} confirmed fault documents in FS Mirrored"
                )

        client.close()

    except Exception as e:
        logger.error(f"Cannot connect to MongoDB or check confirmed faults: {e}")
        return False

    # ── Step 2: Run group-scoped retraining ───────────────────────────────────
    logger.info(f"\n[Phase 2] Retraining Group {group} model...")
    logger.info(f"  Training data: factory_features (Group {group}) + confirmed faults (Group {group})")
    logger.info(f"  Other groups: UNAFFECTED — their serving continues ✓")

    try:
        from orchestrator import WorkflowExecutor
        executor = WorkflowExecutor()
        executor.run_training_only(
            run_id=run_id,
            group=group,
            mongo_uri=mongo_uri,
            db_name=db_name,
        )
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False

    # ── Step 3: Get new model metrics ─────────────────────────────────────────
    logger.info(f"\n[Phase 3] Comparing new Group {group} model vs current champion...")
    try:
        from utils.model_registry import ModelRegistry
        registry = ModelRegistry()
        pending_models = registry.list_models(run_id=run_id, status="pending")
        if not pending_models:
            logger.error(
                f"No pending models found for run_id={run_id}. "
                f"Did training complete successfully?"
            )
            return False

        best_new = min(
            pending_models,
            key=lambda m: m.get("metrics", {}).get(COMPARISON_METRIC, float("inf"))
        )
        new_metrics  = best_new.get("metrics", {})
        new_model_id = best_new["model_id"]
        new_path     = best_new["model_path"]
        logger.info(f"  Best new model : {new_model_id}")
        logger.info(f"  Metrics        : {new_metrics}")

    except Exception as e:
        logger.error(f"Model registry error: {e}", exc_info=True)
        return False

    # ── Step 4: Compare and promote OR reject ─────────────────────────────────
    current_champion = read_champion(group)
    champion_metrics = current_champion.get("metrics", {}) if current_champion else {}

    if is_new_model_better(new_metrics, champion_metrics, COMPARISON_METRIC):
        new_mae      = new_metrics.get(COMPARISON_METRIC)
        champ_mae    = champion_metrics.get(COMPARISON_METRIC, "N/A")
        new_mae_str  = f"{new_mae:.2f}"   if isinstance(new_mae,   (int, float)) else "N/A"
        champ_mae_str = f"{champ_mae:.2f}" if isinstance(champ_mae, (int, float)) else "N/A"

        logger.info(
            f"\n  ✅ New model is BETTER — promoting to Group {group} champion.\n"
            f"  Model ID : {new_model_id}\n"
            f"  {COMPARISON_METRIC}: {new_mae_str} (was: {champ_mae_str})"
        )

        if not dry_run:
            # approve_model + deploy_model — deploy_model auto-archives the
            # previous champion (sets its status to 'archived').
            registry.approve_model(new_model_id, approved_by="preprod_auto")
            registry.deploy_model(new_model_id)

            # Write group-specific champion file (the file run_serving.py watches)
            write_champion(
                group=group,
                model_id=new_model_id,
                model_path=new_path,
                metrics=new_metrics,
            )
            logger.info(
                f"  champion_bearing{group}.json updated → "
                f"run_serving.py (Group {group}) will hot-swap on next burst ✓"
            )
        else:
            logger.info(
                f"  [DRY RUN] Promotion skipped — "
                f"champion_bearing{group}.json NOT written, registry NOT updated."
            )

    else:
        # New model is NOT better → mark it REJECTED in the registry.
        # This makes the registry status explicit and traceable instead of
        # leaving the model floating in PENDING forever.
        new_mae   = new_metrics.get(COMPARISON_METRIC)
        champ_mae = champion_metrics.get(COMPARISON_METRIC)
        reason = (
            f"New model {COMPARISON_METRIC}={new_mae} is not better than "
            f"current Group {group} champion {COMPARISON_METRIC}={champ_mae}."
        )

        logger.info(
            f"\n  ℹ️  New model is NOT better — rejecting and retaining current Group {group} champion.\n"
            f"  New {COMPARISON_METRIC}      : {new_mae}\n"
            f"  Champion {COMPARISON_METRIC} : {champ_mae}"
        )

        if not dry_run:
            registry.reject_model(
                new_model_id,
                rejected_by="preprod_auto",
                reason=reason,
            )
            logger.info(
                f"  Model {new_model_id} marked as REJECTED in the registry."
            )
        else:
            logger.info(
                f"  [DRY RUN] Rejection skipped — model {new_model_id} "
                f"left as PENDING."
            )

    logger.info(f"\n[Pre-Production complete — Group {group}]")
    logger.info(
        f"  Group {group} Serving Pipeline will pick up the new champion\n"
        f"  on the next burst boundary — no restart required.\n"
        f"  Groups 1, 2, 3 (minus Group {group}) were never interrupted."
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-Production — retrains the model for a specific bearing group.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic trigger (from API after fault confirmation):
  python run_preprod.py --run_id <run_id> --group 1

  # Manual trigger:
  python run_preprod.py --group 2

  # Dry run (no champion file written, no model rejected/promoted):
  python run_preprod.py --group 1 --dry_run
        """,
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help='Bearing group to retrain: "1", "2", or "3".',
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier (auto-generated if omitted).",
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default=DEFAULT_MONGO_URI,
        help=f"MongoDB URI (default: {DEFAULT_MONGO_URI}).",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default=DEFAULT_DB_NAME,
        help=f"Database name (default: {DEFAULT_DB_NAME}).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compare only — do not promote or reject.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ok = run_preprod(
        group     = args.group,
        run_id    = args.run_id,
        mongo_uri = args.mongo_uri,
        db_name   = args.db_name,
        dry_run   = args.dry_run,
    )
    sys.exit(0 if ok else 1)