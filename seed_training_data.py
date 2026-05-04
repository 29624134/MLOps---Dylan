"""
seed_historical_data.py
═══════════════════════════════════════════════════════════════════════════════
One-time seeder: loads train/val bearing features.csv files into MongoDB
'factory_features' collection so the orchestrator can train without reading
local disk during training.

Auto-extraction
───────────────
If features.csv is missing for a bearing, the seeder will attempt to extract
it automatically using FeatureExtractorPHM before seeding. This requires the
raw acc_*.csv files to be present in the bearing's source folder.

Extraction config mirrors what the orchestrator uses (workflow.yaml defaults):
    burst_period       : 10.0 s
    failure_threshold  : 20.0 g
    n_consecutive      : 1 burst

If extraction also fails (e.g. no raw acc_*.csv files present), the bearing
is skipped and logged as 'missing_csv' in the summary.

Usage
─────
    python seed_historical_data.py
    python seed_historical_data.py --force
    python seed_historical_data.py --dry_run
    python seed_historical_data.py --group 1          # seed only group 1 bearings
    python seed_historical_data.py --role train val   # seed only train + val
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEED] %(levelname)s — %(message)s",
)
logger = logging.getLogger("seed_historical_data")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB_NAME   = os.environ.get("MONGO_DB",  "phm_mlops")

from utils.db_collections import COL_FACTORY_FEATURES
COLLECTION_NAME = COL_FACTORY_FEATURES   # "factory_features"
BEARINGS_CONFIG = "config/bearings.json"
SEED_ROLES      = {"train", "val", "test"}   # live bearings excluded

# Extraction defaults — must match workflow.yaml
_BURST_PERIOD       = 10.0
_FAILURE_THRESHOLD  = 20.0
_N_CONSECUTIVE      = 1


# ─────────────────────────────────────────────────────────────────────────────
# Auto-extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(name: str, source_path: str, features_csv: str) -> bool:
    """
    Extract features.csv from raw acc_*.csv files using FeatureExtractorPHM.

    Returns True if extraction succeeded and features.csv now exists.
    Returns False if extraction failed (e.g. no raw data present).
    """
    logger.info(
        f"  [{name}] features.csv not found — attempting extraction from "
        f"raw acc_*.csv files in {source_path}"
    )

    try:
        from scripts.feature_extractor import FeatureExtractorPHM
    except ImportError as e:
        logger.error(
            f"  [{name}] Cannot import FeatureExtractorPHM: {e}. "
            f"Run this script from the project root directory."
        )
        return False

    # Check raw data exists
    acc_files = sorted(Path(source_path).glob("acc_*.csv"))
    if not acc_files:
        logger.warning(
            f"  [{name}] No acc_*.csv files found in {source_path} — "
            f"cannot extract. Skipping."
        )
        return False

    logger.info(f"  [{name}] Found {len(acc_files)} acc_*.csv files — extracting...")

    try:
        extractor = FeatureExtractorPHM(config={
            "input_location":    source_path,
            "output_location":   features_csv,
            "bearing_name":      name,
            "is_test":           True,     # safe default — RUL measured from last burst
            "burst_period":      _BURST_PERIOD,
            "failure_threshold": _FAILURE_THRESHOLD,
            "n_consecutive":     _N_CONSECUTIVE,
        })
        extractor.run()
    except Exception as e:
        logger.error(f"  [{name}] Extraction FAILED: {e}")
        return False

    if os.path.exists(features_csv):
        df = pd.read_csv(features_csv)
        logger.info(
            f"  [{name}] Extraction complete — {len(df)} bursts, "
            f"{df.shape[1]} columns → {features_csv}"
        )
        return True
    else:
        logger.error(
            f"  [{name}] Extraction ran but features.csv not found at {features_csv}"
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_bearings(
    config_path:    str,
    filter_roles:   Optional[set] = None,
    filter_groups:  Optional[set] = None,
) -> List[Dict]:
    """
    Load bearings from bearings.json filtered by role and/or group.

    Parameters
    ----------
    config_path   : str  — path to bearings.json
    filter_roles  : set  — if set, only include bearings with these roles
    filter_groups : set  — if set, only include bearings from these groups
    """
    with open(config_path) as f:
        cfg = json.load(f)
    base_path = cfg["base_path"]
    roles     = filter_roles or SEED_ROLES

    bearings = []
    for b in cfg["bearings"]:
        if b["role"] not in roles:
            continue
        if filter_groups and b.get("group") not in filter_groups:
            continue

        # Resolve source path — explicit source_path wins, else base_path/name
        sp = b.get("source_path", "")
        if not sp or not os.path.isdir(sp):
            sp = os.path.join(base_path, b["name"])
        b["_resolved_path"] = sp
        bearings.append(b)
    return bearings


# ─────────────────────────────────────────────────────────────────────────────
# Per-bearing seeding
# ─────────────────────────────────────────────────────────────────────────────

def _already_seeded(col, bearing_name: str) -> int:
    """Return the number of documents already in MongoDB for this bearing."""
    return col.count_documents({"dataset_id": bearing_name})


def _seed_bearing(
    col,
    bearing:    Dict,
    run_id:     str,
    force:      bool,
    dry_run:    bool,
    no_extract: bool,
) -> Dict:
    """
    Seed one bearing's features into MongoDB.

    If features.csv is missing and no_extract is False, extraction is
    attempted automatically from raw acc_*.csv files.

    Returns a status dict describing what happened.
    """
    name        = bearing["name"]
    role        = bearing["role"]
    group       = bearing.get("group", "?")
    source_path = bearing["_resolved_path"]
    features_csv = os.path.join(source_path, "features.csv")

    # ── Auto-extract if features.csv is missing ───────────────────────────────
    if not os.path.exists(features_csv):
        if no_extract:
            logger.warning(
                f"  [{name}] features.csv not found and --no_extract set — skipping."
            )
            return {"bearing": name, "group": group, "status": "missing_csv", "rows": 0}

        extracted = _extract_features(name, source_path, features_csv)
        if not extracted:
            return {"bearing": name, "group": group, "status": "missing_csv", "rows": 0}

    # ── Check if already seeded ───────────────────────────────────────────────
    existing_count = _already_seeded(col, name)
    if existing_count > 0 and not force:
        logger.info(
            f"  [{name}] Already in MongoDB ({existing_count} rows) — skipping. "
            f"Use --force to re-seed."
        )
        return {
            "bearing": name, "group": group,
            "status": "already_seeded", "rows": existing_count,
        }

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(features_csv)
    logger.info(f"  [{name}] Loaded {len(df)} rows from {features_csv}")

    if dry_run:
        logger.info(f"  [{name}] [DRY RUN] Would insert {len(df)} rows into MongoDB.")
        return {"bearing": name, "group": group, "status": "dry_run", "rows": len(df)}

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
            "group":        group,
            "run_id":       run_id,
            "source":       "seed_historical_data",
            "seeded_at":    now,
        }
        docs.append(doc)

    # ── Insert in batches of 1000 ─────────────────────────────────────────────
    batch_size = 1000
    inserted   = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        col.insert_many(batch, ordered=False)
        inserted += len(batch)
        logger.info(f"  [{name}] Inserted {inserted}/{len(docs)} rows...")

    logger.info(
        f"  [{name}] ✓ Seeded {inserted} rows into MongoDB '{COLLECTION_NAME}' "
        f"(Group {group}, role={role})."
    )
    return {"bearing": name, "group": group, "status": "seeded", "rows": inserted}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def seed(
    mongo_uri:      str           = DEFAULT_MONGO_URI,
    db_name:        str           = DEFAULT_DB_NAME,
    force:          bool          = False,
    dry_run:        bool          = False,
    config_path:    str           = BEARINGS_CONFIG,
    filter_groups:  Optional[set] = None,
    filter_roles:   Optional[set] = None,
    no_extract:     bool          = False,
) -> List[Dict]:
    """
    Seed all qualifying bearings from bearings.json into MongoDB factory_features.

    Parameters
    ----------
    mongo_uri     : str           — MongoDB connection string
    db_name       : str           — database name
    force         : bool          — re-seed bearings already in MongoDB
    dry_run       : bool          — check only, do not write
    config_path   : str           — path to bearings.json
    filter_groups : set or None   — if set, only seed these groups (e.g. {"1", "2"})
    filter_roles  : set or None   — if set, only seed these roles (default: train/val/test)
    no_extract    : bool          — if True, skip auto-extraction and skip missing CSVs
    """
    logger.info("=" * 60)
    logger.info("  PHM MLOps — Historical Data Seeder")
    logger.info("=" * 60)
    logger.info(f"  MongoDB    : {mongo_uri} / {db_name}")
    logger.info(f"  Collection : {COLLECTION_NAME}")
    logger.info(f"  Roles      : {filter_roles or SEED_ROLES}")
    logger.info(f"  Groups     : {filter_groups or 'all'}")
    logger.info(f"  Force      : {force}")
    logger.info(f"  Dry run    : {dry_run}")
    logger.info(f"  Auto-extract: {not no_extract}")
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
    col.create_index(
        [("dataset_id", 1), ("metadata.group", 1)],
        name="idx_dataset_group",
    )

    # ── Load bearing config ───────────────────────────────────────────────────
    if not os.path.exists(config_path):
        logger.error(f"Bearings config not found: {config_path}")
        sys.exit(1)

    bearings = _load_bearings(
        config_path,
        filter_roles=filter_roles,
        filter_groups=filter_groups,
    )
    if not bearings:
        logger.warning("No bearings found matching the specified filters.")
        return []

    logger.info(
        f"Found {len(bearings)} bearing(s) to seed: "
        f"{[b['name'] for b in bearings]}"
    )

    run_id = f"seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ── Seed each bearing ─────────────────────────────────────────────────────
    results = []
    for bearing in bearings:
        logger.info(
            f"\nSeeding [{bearing['role'].upper()}] {bearing['name']} "
            f"(Group {bearing.get('group', '?')}) ..."
        )
        result = _seed_bearing(col, bearing, run_id, force, dry_run, no_extract)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Seeding Summary")
    logger.info("=" * 60)

    STATUS_ICONS = {
        "seeded":         "✓",
        "already_seeded": "–",
        "missing_csv":    "✗",
        "dry_run":        "○",
        "extracted":      "↑",
    }

    total_rows     = 0
    extracted_list = []
    failed_list    = []

    for r in results:
        icon = STATUS_ICONS.get(r["status"], "?")
        logger.info(
            f"  {icon} {r['bearing']:<16} group={r.get('group','?')}  "
            f"{r['status']:<16} {r['rows']:>6} rows"
        )
        if r["status"] == "seeded":
            total_rows += r["rows"]
        elif r["status"] == "missing_csv":
            failed_list.append(r["bearing"])

    logger.info("=" * 60)
    logger.info(f"  Total rows inserted : {total_rows}")
    logger.info(f"  Failed / missing    : {len(failed_list)}")
    if failed_list:
        logger.info(f"    {failed_list}")
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
            "Seed train/val bearing features into MongoDB factory_features. "
            "Auto-extracts features.csv from raw acc_*.csv files if missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Seed everything (auto-extracts if needed):
  python seed_historical_data.py

  # Re-seed all bearings (clears existing data):
  python seed_historical_data.py --force

  # Check what would be seeded without writing:
  python seed_historical_data.py --dry_run

  # Seed only Group 1 bearings:
  python seed_historical_data.py --group 1

  # Seed only train-role bearings across all groups:
  python seed_historical_data.py --role train

  # Seed without auto-extracting (skip bearings with missing features.csv):
  python seed_historical_data.py --no_extract
        """,
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
        help="Re-seed bearings already in MongoDB (clears and re-inserts)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Check what would be seeded without writing anything",
    )
    parser.add_argument(
        "--config", default=BEARINGS_CONFIG,
        help=f"Path to bearings config (default: {BEARINGS_CONFIG})",
    )
    parser.add_argument(
        "--group", type=str, nargs="+", metavar="GROUP",
        help="Only seed bearings from these groups (e.g. --group 1 2)",
    )
    parser.add_argument(
        "--role", type=str, nargs="+", metavar="ROLE",
        default=None,
        choices=["train", "val", "test"],
        help="Only seed bearings with these roles (default: train val test)",
    )
    parser.add_argument(
        "--no_extract", action="store_true", default=False,
        help=(
            "Skip auto-extraction. Bearings with missing features.csv "
            "are logged as failed instead of being extracted."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    seed(
        mongo_uri     = args.mongo_uri,
        db_name       = args.db_name,
        force         = args.force,
        dry_run       = args.dry_run,
        config_path   = args.config,
        filter_groups = set(args.group) if args.group else None,
        filter_roles  = set(args.role)  if args.role  else None,
        no_extract    = args.no_extract,
    )