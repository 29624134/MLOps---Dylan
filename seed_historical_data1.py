"""
seed_historical_data.py
═══════════════════════════════════════════════════════════════════════════════
One-time historical data seeder.

The MLOps orchestrator reads ALL training/validation data from MongoDB.
This script seeds the MongoDB 'features' collection with the pre-extracted
features.csv files for all train and val role bearings (IEEE PHM 2012
historical run-to-failure datasets).

This runs ONCE before the first POST /workflow/trigger call. After that,
the orchestrator reads from MongoDB and never touches the local CSV files.

When to re-run
──────────────
- First-time setup on a new machine
- After dropping / clearing the MongoDB 'features' collection
- After adding new train/val bearings to config/bearings.json

When NOT to run
───────────────
- Live bearings — SCADA handles those (streams to 'live_features')
- After confirmed faults — the API's confirm-fault endpoint handles those

Usage
─────
    python seed_historical_data.py
    python seed_historical_data.py --force        # re-seed even if already present
    python seed_historical_data.py --dry_run      # check what would be seeded
    python seed_historical_data.py --mongo_uri mongodb://localhost:27017
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEED] %(levelname)s — %(message)s",
)
logger = logging.getLogger("seed_historical_data")

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MONGO_URI   = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB_NAME     = os.environ.get("MONGO_DB",  "phm_mlops")
from utils.db_collections import COL_FACTORY_FEATURES
COLLECTION_NAME     = COL_FACTORY_FEATURES   # "factory_features"
BEARINGS_CONFIG     = "config/bearings.json"
SEED_ROLES          = {"train", "val", "test"}   # live bearings excluded


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bearings(config_path: str) -> List[Dict]:
    with open(config_path) as f:
        cfg = json.load(f)
    base_path = cfg["base_path"]
    bearings  = []
    for b in cfg["bearings"]:
        if b["role"] not in SEED_ROLES:
            continue
        # Resolve source path — explicit source_path wins, else base_path/name
        sp = b.get("source_path", "")
        if not sp or not os.path.isdir(sp):
            sp = os.path.join(base_path, b["name"])
        b["_resolved_path"] = sp
        bearings.append(b)
    return bearings


def _already_seeded(col, bearing_name: str) -> int:
    """Return the number of documents already in MongoDB for this bearing."""
    return col.count_documents({"dataset_id": bearing_name})


def _seed_bearing(
    col,
    bearing:  Dict,
    run_id:   str,
    force:    bool,
    dry_run:  bool,
) -> Dict:
    """
    Seed one bearing's features.csv into MongoDB.

    Returns a status dict describing what happened.
    """
    name         = bearing["name"]
    role         = bearing["role"]
    source_path  = bearing["_resolved_path"]
    features_csv = os.path.join(source_path, "features.csv")

    # ── Check features.csv exists ─────────────────────────────────────────────
    if not os.path.exists(features_csv):
        logger.warning(
            f"  [{name}] features.csv not found at {features_csv} — skipping."
        )
        return {"bearing": name, "status": "missing_csv", "rows": 0}

    # ── Check if already seeded ───────────────────────────────────────────────
    existing_count = _already_seeded(col, name)
    if existing_count > 0 and not force:
        logger.info(
            f"  [{name}] Already in MongoDB ({existing_count} rows) — skipping. "
            f"Use --force to re-seed."
        )
        return {"bearing": name, "status": "already_seeded", "rows": existing_count}

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(features_csv)
    logger.info(f"  [{name}] Loaded {len(df)} rows from {features_csv}")

    if dry_run:
        logger.info(f"  [{name}] [DRY RUN] Would insert {len(df)} rows into MongoDB.")
        return {"bearing": name, "status": "dry_run", "rows": len(df)}

    # ── Clear existing if force re-seeding ────────────────────────────────────
    if existing_count > 0 and force:
        deleted = col.delete_many({"dataset_id": name}).deleted_count
        logger.info(f"  [{name}] Cleared {deleted} existing rows (--force).")

    # ── Build MongoDB documents ───────────────────────────────────────────────
    now  = datetime.now(timezone.utc).isoformat()
    docs = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        doc["dataset_id"] = name
        doc["version"]    = run_id
        doc["metadata"]   = {
            "bearing_name": name,
            "role":         role,
            "run_id":       run_id,
            "source":       "seed_historical_data",
            "seeded_at":    now,
        }
        docs.append(doc)

    # ── Insert in batches of 1000 ─────────────────────────────────────────────
    batch_size  = 1000
    inserted    = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        col.insert_many(batch, ordered=False)
        inserted += len(batch)
        logger.info(f"  [{name}] Inserted {inserted}/{len(docs)} rows...")

    logger.info(f"  [{name}] ✓ Seeded {inserted} rows into MongoDB '{COLLECTION_NAME}'.")
    return {"bearing": name, "status": "seeded", "rows": inserted}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def seed(
    mongo_uri:   str  = DEFAULT_MONGO_URI,
    db_name:     str  = DEFAULT_DB_NAME,
    force:       bool = False,
    dry_run:     bool = False,
    config_path: str  = BEARINGS_CONFIG,
) -> List[Dict]:

    logger.info("=" * 60)
    logger.info("  Historical Data Seeder")
    logger.info("=" * 60)
    logger.info(f"  MongoDB  : {mongo_uri} / {db_name}")
    logger.info(f"  Collection: {COLLECTION_NAME}")
    logger.info(f"  Roles    : {SEED_ROLES}")
    logger.info(f"  Force    : {force}")
    logger.info(f"  Dry run  : {dry_run}")
    logger.info("=" * 60)

    # ── Connect ───────────────────────────────────────────────────────────────
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        logger.info("MongoDB connection OK.")
    except Exception as e:
        logger.error(f"Cannot connect to MongoDB: {e}")
        sys.exit(1)

    db  = client[db_name]
    col = db[COLLECTION_NAME]

    # Ensure indexes
    col.create_index("dataset_id", name="idx_dataset_id")
    col.create_index(
        [("dataset_id", 1), ("time_s", 1)],
        name="idx_dataset_time",
    )

    # ── Load bearing config ───────────────────────────────────────────────────
    if not os.path.exists(config_path):
        logger.error(f"Bearings config not found: {config_path}")
        sys.exit(1)

    bearings = _load_bearings(config_path)
    if not bearings:
        logger.warning("No train/val/test bearings found in config.")
        return []

    logger.info(
        f"Found {len(bearings)} bearing(s) to seed: "
        f"{[b['name'] for b in bearings]}"
    )

    # ── Run ID for lineage ────────────────────────────────────────────────────
    run_id = f"seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Seed each bearing ─────────────────────────────────────────────────────
    results = []
    for bearing in bearings:
        logger.info(f"\nSeeding [{bearing['role'].upper()}] {bearing['name']} ...")
        result = _seed_bearing(col, bearing, run_id, force, dry_run)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Seeding Summary")
    logger.info("=" * 60)
    total_rows = 0
    for r in results:
        status_icon = {
            "seeded":         "✓",
            "already_seeded": "–",
            "missing_csv":    "✗",
            "dry_run":        "○",
        }.get(r["status"], "?")
        logger.info(
            f"  {status_icon} {r['bearing']:<20} "
            f"{r['status']:<16} {r['rows']:>6} rows"
        )
        if r["status"] == "seeded":
            total_rows += r["rows"]
    logger.info("=" * 60)
    logger.info(f"  Total rows inserted: {total_rows}")
    if dry_run:
        logger.info("  (Dry run — nothing was written to MongoDB)")
    logger.info("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-time seeder: loads train/val bearing features.csv files "
            "into MongoDB 'features' collection so the orchestrator can "
            "train without reading local disk."
        )
    )
    parser.add_argument(
        "--mongo_uri", default=DEFAULT_MONGO_URI,
        help=f"MongoDB URI (default: {DEFAULT_MONGO_URI})",
    )
    parser.add_argument(
        "--db_name", default=DEFAULT_DB_NAME,
        help=f"MongoDB database name (default: {DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-seed bearings that are already in MongoDB (clears and re-inserts)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Check what would be seeded without writing anything",
    )
    parser.add_argument(
        "--config", default=BEARINGS_CONFIG,
        help=f"Path to bearings config (default: {BEARINGS_CONFIG})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    seed(
        mongo_uri   = args.mongo_uri,
        db_name     = args.db_name,
        force       = args.force,
        dry_run     = args.dry_run,
        config_path = args.config,
    )