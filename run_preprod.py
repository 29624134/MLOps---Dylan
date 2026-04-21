"""
run_preprod.py
═══════════════════════════════════════════════════════════════════════════════
Pre-Production Entry Point — runs independently of Serving.

This script is triggered AFTER a maintenance worker confirms or denies a fault
on the dashboard. It:

1. Reads confirmed/labelled fault data from Feature Store Mirrored
   (MongoDB 'confirmed_faults' collection)
2. Runs automated model retraining on FS Mirrored data
3. Compares new model metrics (MAE etc.) against the current champion
4. If new model is better → writes model_registry/champion.json atomically
   → run_serving.py detects this and hot-swaps the model between bursts
5. If new model is not better → logs result, current champion retained

The Serving Pipeline continues uninterrupted during this entire process.
Training runs on FS Mirrored data only — the live FS is never touched.

Architecture:
    [Dashboard: confirm fault] --> [FS Mirrored: confirmed_faults]
                                             |
                                      [run_preprod.py]
                                             |
                               [ModelRegistry + champion.json]
                                             |
                                    [run_serving.py detects]

Usage
─────
    # Triggered automatically by the dashboard after fault confirm:
    python run_preprod.py --run_id <run_id>

    # Manual trigger (e.g. after manually adding confirmed data):
    python run_preprod.py

    # Dry run (compare only, do not promote):
    python run_preprod.py --dry_run
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
import time
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
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB_NAME   = "phm_mlops"
CHAMPION_PATH     = os.path.join("model_registry", "champion.json")
COMPARISON_METRIC = "mae_s"   # lower is better


# ─────────────────────────────────────────────────────────────────────────────
# Champion pointer management
# ─────────────────────────────────────────────────────────────────────────────

def write_champion(model_id: str, model_path: str, metrics: dict) -> None:
    """
    Atomically write the champion pointer file.

    run_serving.py watches this file for changes between bursts and reloads
    the model when it detects a new champion (hot-swap).

    Uses write-then-rename for atomicity — no partial reads possible.
    """
    os.makedirs(os.path.dirname(CHAMPION_PATH), exist_ok=True)
    tmp_path = CHAMPION_PATH + ".tmp"
    champion = {
        "model_id":    model_id,
        "model_path":  model_path,
        "metrics":     metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(tmp_path, "w") as f:
        json.dump(champion, f, indent=2)
    os.replace(tmp_path, CHAMPION_PATH)   # atomic on POSIX
    logger.info(f"[Champion] champion.json written → {model_id}")


def read_champion() -> Optional[dict]:
    """Return the current champion dict or None if no champion exists."""
    if not os.path.exists(CHAMPION_PATH):
        return None
    try:
        with open(CHAMPION_PATH) as f:
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
    """
    Returns True if the new model is better than the current champion.
    Lower is better for MAE/RMSE metrics.
    """
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
    run_id:   Optional[str],
    mongo_uri: str,
    db_name:   str,
    dry_run:   bool,
):
    """
    Full pre-production flow:
      1. Validate confirmed fault data exists in FS Mirrored
      2. Run retraining via WorkflowExecutor (trains on confirmed_faults + train set)
      3. Compare new model vs current champion
      4. Promote if better (write champion.json)
    """
    run_id = run_id or f"preprod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 60)
    logger.info("  PHM MLOps — Pre-Production (Retraining)")
    logger.info("=" * 60)
    logger.info(f"  Run ID   : {run_id}")
    logger.info(f"  MongoDB  : {mongo_uri} / {db_name}")
    logger.info(f"  Dry run  : {dry_run}")
    logger.info("=" * 60)

    # ── Step 1: Verify new confirmed fault data exists and hasn't been used ──────
    logger.info("\n[Phase 1] Verifying new confirmed fault data in FS Mirrored...")
    try:
        from pymongo import MongoClient
        client        = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db            = client[db_name]
        confirmed_col = db["confirmed_faults"]
        preprod_col   = db["preprod_runs"]   # tracks which data was used in each run

        # 1a. Poll until confirmed fault data arrives in MongoDB.
        #     run_preprod.py starts almost immediately after confirm is pressed,
        #     but confirm_fault_and_push_to_store() may still be writing to MongoDB.
        #     We retry for up to DATA_WAIT_TIMEOUT_S seconds before giving up.
        DATA_WAIT_TIMEOUT_S = 60   # max seconds to wait for data to appear
        DATA_POLL_INTERVAL  = 2    # seconds between polls
        waited              = 0
        n_confirmed         = 0

        logger.info(
            f"  Waiting for confirmed fault data to arrive in FS Mirrored "
            f"(timeout={DATA_WAIT_TIMEOUT_S}s, polling every {DATA_POLL_INTERVAL}s)..."
        )
        while waited < DATA_WAIT_TIMEOUT_S:
            n_confirmed = confirmed_col.count_documents({})
            if n_confirmed > 0:
                logger.info(
                    f"  ✓ Data arrived after {waited}s — "
                    f"{n_confirmed} confirmed fault record(s) found."
                )
                break
            logger.info(
                f"  No data yet — retrying in {DATA_POLL_INTERVAL}s "
                f"({waited}/{DATA_WAIT_TIMEOUT_S}s elapsed)..."
            )
            time.sleep(DATA_POLL_INTERVAL)
            waited += DATA_POLL_INTERVAL

        if n_confirmed == 0:
            logger.warning(
                f"  No confirmed fault data found after {DATA_WAIT_TIMEOUT_S}s. "
                f"confirm_fault_and_push_to_store() may have failed. "
                f"Aborting retraining."
            )
            return False

        # 1b. Get the distinct bearing names that have confirmed fault data
        confirmed_bearings = confirmed_col.distinct("dataset_id")
        logger.info(
            f"  Found {n_confirmed} confirmed fault records across "
            f"{len(confirmed_bearings)} bearing(s): {confirmed_bearings}"
        )

        # 1c. Check which bearings were already used in a previous preprod run
        already_used = set()
        for bearing in confirmed_bearings:
            used = preprod_col.find_one({"bearing_name": bearing, "used": True})
            if used:
                already_used.add(bearing)

        new_bearings = [b for b in confirmed_bearings if b not in already_used]

        if not new_bearings:
            logger.warning(
                f"  All confirmed fault data has already been used in a previous "
                f"retraining run: {list(already_used)}. "
                f"No new data to train on — aborting retraining."
            )
            return False

        logger.info(
            f"  ✓ New confirmed fault data found for: {new_bearings}\n"
            f"  Already used in previous runs: {list(already_used)}\n"
            f"  Proceeding with retraining."
        )

        # 1d. Mark these bearings as used so we don't retrain on them again
        # (This is written now so even if training fails we don't double-train
        #  on the same data. If you want to retry failed runs, delete these records.)
        if not dry_run:
            for bearing in new_bearings:
                preprod_col.update_one(
                    {"bearing_name": bearing},
                    {"$set": {
                        "bearing_name": bearing,
                        "used":         True,
                        "run_id":       run_id,
                        "marked_at":    datetime.now(timezone.utc).isoformat(),
                    }},
                    upsert=True,
                )
            logger.info(
                f"  Marked {len(new_bearings)} bearing(s) as used in "
                f"'preprod_runs' collection."
            )

    except Exception as e:
        logger.error(f"Cannot connect to MongoDB or check confirmed faults: {e}")
        return False

    # ── Step 2: Run retraining ────────────────────────────────────────────────
    logger.info("\n[Phase 2] Starting automated model retraining...")
    logger.info("  Training on: confirmed_faults (FS Mirrored) + train-role bearings")
    logger.info("  NOTE: Live Feature Store is NOT touched — serving continues ✓")

    try:
        from orchestrator import WorkflowExecutor
        executor = WorkflowExecutor()
        executor.run_training_only(run_id=run_id, mongo_uri=mongo_uri, db_name=db_name)
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False

    # ── Step 3: Get new model metrics from registry ───────────────────────────
    logger.info("\n[Phase 3] Comparing new model vs current champion...")
    try:
        from utils.model_registry import ModelRegistry
        registry = ModelRegistry()
        pending_models = registry.list_models(run_id=run_id, status="pending")
        if not pending_models:
            logger.error(
                f"No pending models found for run_id={run_id}. "
                "Did training complete successfully?"
            )
            return False

        # Pick best pending model by metric
        best_new = min(
            pending_models,
            key=lambda m: m.get("metrics", {}).get(COMPARISON_METRIC, float("inf"))
        )
        new_metrics  = best_new.get("metrics", {})
        new_model_id = best_new["model_id"]
        new_path     = best_new["model_path"]
        logger.info(f"  Best new model: {new_model_id}")
        logger.info(f"  Metrics: {new_metrics}")

    except Exception as e:
        logger.error(f"Model registry error: {e}", exc_info=True)
        return False

    # ── Step 4: Compare and promote ───────────────────────────────────────────
    current_champion = read_champion()
    champion_metrics = current_champion.get("metrics", {}) if current_champion else {}

    if is_new_model_better(new_metrics, champion_metrics, COMPARISON_METRIC):
        logger.info(
            f"\n  ✅ New model is BETTER — promoting to champion.\n"
            f"  Model ID : {new_model_id}\n"
            f"  {COMPARISON_METRIC}: {new_metrics.get(COMPARISON_METRIC):.2f} "
            f"(was: {champion_metrics.get(COMPARISON_METRIC, 'N/A')})"
        )

        if not dry_run:
            # Approve + deploy in Model Registry
            registry.approve_model(new_model_id, approved_by="preprod_auto")
            registry.deploy_model(new_model_id)

            # Write champion.json — run_serving.py will detect this and hot-swap
            write_champion(
                model_id=new_model_id,
                model_path=new_path,
                metrics=new_metrics,
            )
            logger.info(
                "  champion.json updated → run_serving.py will hot-swap "
                "on next burst boundary ✓"
            )
        else:
            logger.info("  [DRY RUN] Promotion skipped — champion.json NOT written.")

    else:
        logger.info(
            f"\n  ℹ️  New model is NOT better — keeping current champion.\n"
            f"  New {COMPARISON_METRIC}      : {new_metrics.get(COMPARISON_METRIC)}\n"
            f"  Champion {COMPARISON_METRIC} : {champion_metrics.get(COMPARISON_METRIC)}"
        )
        # Leave new model as PENDING for manual review
        logger.info(
            f"  New model {new_model_id} left as PENDING for manual review."
        )

    logger.info("\n[Pre-Production complete]")
    logger.info(
        "  The Serving Pipeline will automatically pick up the new champion\n"
        "  on the next burst cycle — no restart required.\n"
        "  When SCADA starts sending data again, run_serving.py will\n"
        "  resume predictions with the latest champion model."
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-Production — retrains and promotes model after fault confirmation.",
    )
    parser.add_argument("--run_id",    type=str, default=None,
                        help="Run ID (auto-generated if not provided)")
    parser.add_argument("--mongo_uri", type=str, default=DEFAULT_MONGO_URI)
    parser.add_argument("--db_name",   type=str, default=DEFAULT_DB_NAME)
    parser.add_argument("--dry_run",   action="store_true", default=False,
                        help="Compare models but do NOT write champion.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    success = run_preprod(
        run_id=args.run_id,
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)