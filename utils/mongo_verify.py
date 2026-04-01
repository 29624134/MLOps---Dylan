"""
utils/mongo_verify.py
─────────────────────────────────────────────────────────────────────────────
Verification script for the PHM MLOps MongoDB Feature Store.

Usage:
    python utils/mongo_verify.py                      # default db: phm_mlops
    python utils/mongo_verify.py --db my_db_name
    python utils/mongo_verify.py --uri mongodb://localhost:27017 --db phm_mlops
    python utils/mongo_verify.py --bearing Bearing1_1
    python utils/mongo_verify.py --export                # dump summary to JSON

─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

# ── Bearings expected in the system (from bearings.json) ─────────────────────
KNOWN_BEARINGS = [
    "Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2",
    "Bearing3_1", "Bearing3_2", "Bearing1_3", "Bearing1_4",
    "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3",
    "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]

BEARING_ROLES = {
    "Bearing1_1": "train",    "Bearing1_2": "train",
    "Bearing2_1": "train",    "Bearing2_2": "train",
    "Bearing3_1": "train",    "Bearing3_2": "train",
    "Bearing1_3": "train",    "Bearing1_4": "test",
    "Bearing1_5": "live",     "Bearing1_6": "test",
    "Bearing1_7": "test",     "Bearing2_3": "test",
    "Bearing2_4": "test",     "Bearing2_5": "val_test",
    "Bearing2_6": "test",     "Bearing2_7": "test",
    "Bearing3_3": "test",
}

EXPECTED_FEATURE_COLS = [
    "file_id", "burst_idx", "time_s",
    "h_max", "h_min", "h_mean", "h_sd", "h_rms", "h_skew", "h_kurt", "h_crest", "h_form",
    "v_max", "v_min", "v_mean", "v_sd", "v_rms", "v_skew", "v_kurt", "v_crest", "v_form",
    "RUL_s", "RUL_norm",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: int, max_value: int, width: int = 30) -> str:
    filled = int(width * value / max_value) if max_value > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _fmt_ts(ts) -> str:
    """Format a MongoDB ObjectId timestamp or datetime."""
    if ts is None:
        return "n/a"
    if hasattr(ts, "generation_time"):
        return ts.generation_time.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def connect(uri: str, db_name: str):
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    except ImportError:
        print("✗  pymongo is not installed. Run:  pip install pymongo")
        raise SystemExit(1)

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")          # fast liveness check
        db = client[db_name]
        print(f"✓  Connected to MongoDB  →  {uri}  /  {db_name}\n")
        return db
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"✗  Cannot connect to MongoDB at {uri}")
        print(f"   Error: {e}")
        raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_collections(db) -> dict:
    """List all collections and their document counts."""
    collections = db.list_collection_names()
    print("=" * 60)
    print("  COLLECTIONS")
    print("=" * 60)
    if not collections:
        print("  (no collections found — has the pipeline run yet?)")
        return {}

    counts = {}
    for col in sorted(collections):
        n = db[col].count_documents({})
        counts[col] = n
        print(f"  {col:<35}  {n:>8,} documents")
    print()
    return counts


def check_feature_store(db, filter_bearing: str = None) -> dict:
    """
    Per-bearing breakdown for the 'features' collection.
    Reports record count, run_id, and column schema validation.
    """
    col = db["features"]
    total = col.count_documents({})

    print("=" * 60)
    print("  FEATURE STORE  ('features' collection)")
    print("=" * 60)

    if total == 0:
        print("  (empty — no features ingested yet)")
        return {}

    # Distinct bearings present
    bearings_in_db = col.distinct("dataset_id")
    if filter_bearing:
        bearings_in_db = [b for b in bearings_in_db if b == filter_bearing]

    max_count = max(col.count_documents({"dataset_id": b}) for b in bearings_in_db) if bearings_in_db else 1

    summary = {}
    for bearing in sorted(bearings_in_db):
        n      = col.count_documents({"dataset_id": bearing})
        runs   = col.distinct("version", {"dataset_id": bearing})
        sample = col.find_one({"dataset_id": bearing})
        role   = BEARING_ROLES.get(bearing, "unknown")

        # Schema check: which expected columns are present / missing?
        if sample:
            present_cols = set(sample.keys()) - {"_id", "dataset_id", "version", "metadata"}
            missing_cols = [c for c in EXPECTED_FEATURE_COLS if c not in present_cols]
            extra_cols   = [c for c in present_cols if c not in EXPECTED_FEATURE_COLS]
            schema_ok    = "✓" if not missing_cols else "✗"
        else:
            missing_cols = []
            extra_cols   = []
            schema_ok    = "?"

        bar = _bar(n, max_count)
        print(f"\n  {bearing}  [{role}]")
        print(f"    Records  : {n:,}  {bar}")
        print(f"    Run IDs  : {', '.join(runs) if runs else 'none'}")
        print(f"    Schema   : {schema_ok}", end="")
        if missing_cols:
            print(f"  — MISSING cols: {missing_cols}", end="")
        if extra_cols:
            print(f"  — extra cols: {extra_cols}", end="")
        print()

        summary[bearing] = {
            "records": n,
            "run_ids": runs,
            "schema_ok": not bool(missing_cols),
            "missing_cols": missing_cols,
        }

    # Bearings registered but NOT yet in DB
    missing_from_db = [b for b in KNOWN_BEARINGS if b not in bearings_in_db]
    if missing_from_db and not filter_bearing:
        print(f"\n  Not yet ingested ({len(missing_from_db)}): {', '.join(missing_from_db)}")

    print(f"\n  Total feature documents: {total:,}")
    print()
    return summary


def check_metadata(db) -> None:
    """Print the metadata collection contents."""
    col = db["metadata"]
    total = col.count_documents({})

    print("=" * 60)
    print("  METADATA  ('metadata' collection)")
    print("=" * 60)

    if total == 0:
        print("  (empty)\n")
        return

    docs = list(col.find({}, {"_id": 0}).sort("dataset_id", 1))
    for doc in docs:
        bearing      = doc.get("dataset_id", "?")
        version      = doc.get("version", "?")
        n_records    = doc.get("n_records", "?")
        feature_type = doc.get("feature_type", "default")
        columns      = doc.get("columns", [])
        print(f"  {bearing:<14}  run={version:<22}  type={feature_type:<12}  "
              f"rows={str(n_records):<6}  cols={len(columns)}")
    print()


def check_latest_run(db) -> None:
    """Show the most recent run_id that appears in the features collection."""
    col = db["features"]
    if col.count_documents({}) == 0:
        return

    # Get the most recently inserted document
    latest = col.find_one(sort=[("_id", -1)])
    if latest:
        print("=" * 60)
        print("  LATEST INGESTION")
    print("=" * 60)
    print(f"  Run ID   : {latest.get('version', 'n/a')}")
    print(f"  Bearing  : {latest.get('dataset_id', 'n/a')}")
    print(f"  Inserted : {_fmt_ts(latest.get('_id'))}")
    print()


def check_indexes(db) -> None:
    """List indexes on key collections."""
    print("=" * 60)
    print("  INDEXES")
    print("=" * 60)
    for col_name in ["features", "metadata"]:
        col = db[col_name]
        if col_name not in db.list_collection_names():
            continue
        indexes = list(col.index_information().values())
        print(f"  {col_name}:")
        for idx in indexes:
            keys = list(idx["key"])
            print(f"    {idx['name']:<30}  keys={keys}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED INDEXES  (run once to speed up queries)
# ─────────────────────────────────────────────────────────────────────────────

def create_recommended_indexes(db) -> None:
    """
    Create indexes on fields most commonly used in queries.
    Safe to call multiple times (index creation is idempotent).
    """
    from pymongo import ASCENDING

    features_col = db["features"]
    features_col.create_index([("dataset_id", ASCENDING)], name="idx_dataset_id")
    features_col.create_index([("version", ASCENDING)],    name="idx_version")
    features_col.create_index(
        [("dataset_id", ASCENDING), ("version", ASCENDING)],
        name="idx_dataset_version",
    )

    meta_col = db["metadata"]
    meta_col.create_index(
        [("dataset_id", ASCENDING), ("version", ASCENDING)],
        name="idx_meta_dataset_version",
        unique=True,
    )

    print("✓  Recommended indexes created (or already existed)\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_summary(db, path: str = "mongo_summary.json") -> None:
    """Write a JSON summary of the feature store state to disk."""
    col = db["features"]
    bearings = col.distinct("dataset_id")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_feature_docs": col.count_documents({}),
        "bearings": {},
    }
    for bearing in sorted(bearings):
        n    = col.count_documents({"dataset_id": bearing})
        runs = col.distinct("version", {"dataset_id": bearing})
        summary["bearings"][bearing] = {
            "records": n,
            "run_ids": runs,
            "role":    BEARING_ROLES.get(bearing, "unknown"),
        }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓  Summary exported to {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify the PHM MLOps MongoDB Feature Store"
    )
    parser.add_argument("--uri",      default="mongodb://localhost:27017",
                        help="MongoDB connection URI")
    parser.add_argument("--db",       default="phm_mlops",
                        help="Database name (default: phm_mlops)")
    parser.add_argument("--bearing",  default=None,
                        help="Filter output to a single bearing (e.g. Bearing1_1)")
    parser.add_argument("--indexes",  action="store_true",
                        help="Create recommended indexes and exit")
    parser.add_argument("--export",   action="store_true",
                        help="Export summary to mongo_summary.json")
    args = parser.parse_args()

    db = connect(args.uri, args.db)

    if args.indexes:
        create_recommended_indexes(db)
        return

    check_collections(db)
    check_feature_store(db, filter_bearing=args.bearing)
    check_metadata(db)
    check_latest_run(db)
    check_indexes(db)

    if args.export:
        export_summary(db)

    print("─" * 60)
    print("  Run with --indexes to create recommended indexes.")
    print("  Run with --export  to save a JSON summary.")
    print("  Run with --bearing <name> to inspect one bearing.")
    print("─" * 60)


if __name__ == "__main__":
    main()