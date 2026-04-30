"""
scada_simulator.py
═══════════════════════════════════════════════════════════════════════════════
SCADA System Simulator — Step 1 of the MLOps pipeline.

Simulates a real SCADA system sending vibration data by reading the IEEE PHM
2012 acc_*.csv files one burst (2560 samples / 10 s) at a time.

Feature computation split (matching real SCADA behaviour):
  SCADA sends   → 5 simple stats per axis (max, min, mean, sd, rms)  [10 values]
  System adds   → 4 derived stats per axis (skew, kurt, crest, form)  [8 values]
                                                               Total = 18 features

The 5 simple stats are computed here from raw signals and written to the live
Feature Store (MongoDB 'feature_store'). run_serving.py reads them and computes
the 4 derived stats before passing the full 18-feature vector into the pipeline.

The Serving Pipeline (run_serving.py) polls 'feature_store' independently and
makes predictions as new bursts arrive. These two processes are fully decoupled
— the simulator does NOT call the serving pipeline directly.

Architecture position:
    [SCADA Simulator] --burst features--> [FS: feature_store] <-- [run_serving.py]

Usage
─────
    # Fast replay (no sleep — for testing):
    python scada_simulator.py --bearing Bearing1_5

    # Realtime (sleep 10 s between bursts, matching real sensor rate):
    python scada_simulator.py --bearing Bearing1_5 --realtime

    # Custom bearing path:
    python scada_simulator.py --bearing_path /data/PHM/Bearing1_5 --realtime

    # Override MongoDB connection:
    python scada_simulator.py --bearing Bearing1_5 --mongo_uri mongodb://localhost:27017

Stop with Ctrl+C. The simulator is idempotent — it tracks which burst_idx it
last sent via a state file so it can resume after interruption.
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

import numpy as np

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCADA] %(levelname)s — %(message)s",
)
logger = logging.getLogger("scada_simulator")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MONGO_URI  = "mongodb://localhost:27017"
DEFAULT_DB_NAME    = "phm_mlops"
STATE_DIR          = "scada_state"
BURST_PERIOD_S     = 10.0
SAMPLES_PER_BURST  = 2560

# ── Collection name — single source of truth ──────────────────────────────────
# Live SCADA bursts go into 'feature_store' (COL_FEATURE_STORE in db_collections).
# run_serving.py reads from the same collection.
from utils.db_collections import COL_FEATURE_STORE
FS_LIVE_COLLECTION = COL_FEATURE_STORE   # "feature_store"


# ─────────────────────────────────────────────────────────────────────────────
# SCADA-side feature extraction — simple stats only
# ─────────────────────────────────────────────────────────────────────────────

def _scada_stats(h: np.ndarray, v: np.ndarray) -> dict:
    """
    Compute the 5 simple statistics per axis that the SCADA system sends.

    Sends (10 values total, 5 per axis):
        {prefix}_max   — peak absolute value
        {prefix}_min   — minimum value
        {prefix}_mean  — arithmetic mean
        {prefix}_sd    — standard deviation (ddof=1)
        {prefix}_rms   — root mean square

    Does NOT compute: skew, kurt, crest, form — those are computed by
    run_serving.py using _derive_features() after reading from the FS.
    """
    stats = {}
    for prefix, sig in (("h", h), ("v", v)):
        stats[f"{prefix}_max"]  = float(np.max(sig))
        stats[f"{prefix}_min"]  = float(np.min(sig))
        stats[f"{prefix}_mean"] = float(np.mean(sig))
        stats[f"{prefix}_sd"]   = float(np.std(sig, ddof=1))
        stats[f"{prefix}_rms"]  = float(np.sqrt(np.mean(sig ** 2)))
    return stats   # 10 values


# ─────────────────────────────────────────────────────────────────────────────
# State management — allows resume after Ctrl+C
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorState:
    """Persist last-sent burst_idx to disk so the simulator can resume."""

    def __init__(self, bearing_name: str):
        os.makedirs(STATE_DIR, exist_ok=True)
        self._path  = os.path.join(STATE_DIR, f"{bearing_name}_state.json")
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_burst_idx": -1, "started_at": None, "bearing": None}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2)

    def last_burst_idx(self) -> int:
        return self._state.get("last_burst_idx", -1)

    def update(self, burst_idx: int, bearing: str):
        self._state["last_burst_idx"] = burst_idx
        self._state["bearing"]        = bearing
        if not self._state.get("started_at"):
            self._state["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def reset(self):
        self._state = {"last_burst_idx": -1, "started_at": None, "bearing": None}
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB writer
# ─────────────────────────────────────────────────────────────────────────────

class LiveFeatureStoreWriter:
    """
    Writes one SCADA stats row per burst into MongoDB 'feature_store' collection.

    Each document schema:
    {
      bearing_name : str
      burst_idx    : int
      time_s       : float   — seconds since start of bearing life
      sent_at      : str     — ISO-8601 UTC timestamp
      scada_stats  : dict    — 10 simple stats (5 per axis: max/min/mean/sd/rms)
                               run_serving.py derives the remaining 8 features
                               (skew/kurt/crest/form per axis) from these values
      consumed     : bool    — False until serving pipeline reads it
    }
    """

    def __init__(self, mongo_uri: str, db_name: str):
        from pymongo import MongoClient, ASCENDING
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._db     = self._client[db_name]
        self._col    = self._db[FS_LIVE_COLLECTION]

        # Ensure indexes for fast polling by serving pipeline
        self._col.create_index(
            [("bearing_name", ASCENDING), ("burst_idx", ASCENDING)],
            unique=True, name="idx_bearing_burst",
        )
        self._col.create_index(
            [("bearing_name", ASCENDING), ("consumed", ASCENDING)],
            name="idx_bearing_consumed",
        )
        logger.info(
            f"[FeatureStore] Connected → {db_name}.{FS_LIVE_COLLECTION}"
        )

    def write_burst(
        self,
        bearing_name: str,
        burst_idx:    int,
        time_s:       float,
        scada_stats:  dict,
    ) -> bool:
        """
        Insert one burst's SCADA stats (10 values). Returns True on success.
        Silently skips duplicate burst_idx (idempotent).
        run_serving.py will derive the remaining 8 features on receipt.
        """
        doc = {
            "bearing_name": bearing_name,
            "burst_idx":    burst_idx,
            "time_s":       time_s,
            "sent_at":      datetime.now(timezone.utc).isoformat(),
            "scada_stats":  scada_stats,
            "consumed":     False,
        }
        try:
            self._col.insert_one(doc)
            return True
        except Exception as e:
            if "duplicate" in str(e).lower() or "E11000" in str(e):
                logger.debug(f"Burst {burst_idx} already in FS — skipping.")
                return False
            logger.error(f"FS write failed for burst {burst_idx}: {e}")
            return False

    def mark_session_end(self, bearing_name: str):
        """Insert a sentinel document signalling end of data for this bearing."""
        try:
            self._col.insert_one({
                "bearing_name": bearing_name,
                "burst_idx":    -1,
                "time_s":       -1.0,
                "sent_at":      datetime.now(timezone.utc).isoformat(),
                "features":     {},
                "consumed":     False,
                "session_end":  True,
            })
            logger.info(
                f"[FeatureStore] Session-end sentinel written for {bearing_name}."
            )
        except Exception as e:
            logger.warning(f"Could not write session-end sentinel: {e}")

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Bearing path resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_bearing_path(
    bearing_name: Optional[str],
    bearing_path: Optional[str],
) -> str:
    if bearing_path:
        return bearing_path

    config_path = os.path.join("config", "bearings.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

        for entry in cfg.get("bearings", []):
            if entry.get("name") == bearing_name:
                sp = entry.get("source_path", "")
                if sp and os.path.isdir(sp):
                    return sp
                break

        base_path = cfg.get("base_path", "")
        if base_path:
            candidate = os.path.join(base_path, bearing_name)
            if os.path.isdir(candidate):
                logger.info(f"Found bearing data at: {candidate}")
                return candidate

    raise FileNotFoundError(
        f"Cannot locate data folder for bearing '{bearing_name}'. "
        f"Use --bearing_path to specify it explicitly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulator(
    bearing_name:  str,
    bearing_path:  str,
    mongo_uri:     str,
    db_name:       str,
    realtime:      bool,
    burst_period:  float,
    resume:        bool,
):
    """
    Core simulator loop.

    1. Connects to MongoDB feature_store
    2. Resolves which burst to start from (resume support)
    3. Streams bursts from acc_*.csv files
    4. Computes SCADA-side simple stats (5 per axis = 10 values)
    5. Writes stats to feature_store collection
    6. Sleeps burst_period between bursts (if realtime=True)
    7. Writes session-end sentinel when all bursts are done
    """
    logger.info("=" * 60)
    logger.info("  PHM MLOps — SCADA Simulator")
    logger.info("=" * 60)
    logger.info(f"  Bearing   : {bearing_name}")
    logger.info(f"  Data path : {bearing_path}")
    logger.info(f"  Mode      : {'realtime (10 s sleep)' if realtime else 'fast replay'}")
    logger.info(f"  MongoDB   : {mongo_uri} / {db_name}")
    logger.info(f"  FS coll   : {FS_LIVE_COLLECTION}")
    logger.info(f"  Resume    : {resume}")
    logger.info("=" * 60)

    fs_writer = LiveFeatureStoreWriter(mongo_uri=mongo_uri, db_name=db_name)
    if not fs_writer.ping():
        logger.error("Cannot reach MongoDB. Is it running?")
        sys.exit(1)

    state = SimulatorState(bearing_name)
    if not resume:
        state.reset()
    start_from = state.last_burst_idx() + 1
    if start_from > 0:
        logger.info(f"Resuming from burst {start_from}")

    try:
        from scripts.data_ingestor import DataIngestorPHM
    except ImportError:
        logger.error(
            "Cannot import DataIngestorPHM. "
            "Run this script from the project root directory."
        )
        sys.exit(1)

    ingestor = DataIngestorPHM(config={
        "input_location":  bearing_path,
        "output_location": bearing_path,
    })

    sent    = 0
    skipped = 0
    try:
        for burst in ingestor.stream_bursts(
            bearing_path,
            burst_period=burst_period,
            realtime=False,
        ):
            burst_idx = burst["burst_idx"]

            if burst_idx < start_from:
                skipped += 1
                continue

            scada_stats = _scada_stats(
                h=burst["h_signal"],
                v=burst["v_signal"],
            )

            ok = fs_writer.write_burst(
                bearing_name=bearing_name,
                burst_idx=burst_idx,
                time_s=burst["time_s"],
                scada_stats=scada_stats,
            )

            if ok:
                sent += 1
                state.update(burst_idx, bearing_name)
                logger.info(
                    f"  → Burst {burst_idx:>4} | "
                    f"time={burst['time_s']:>8.1f}s | "
                    f"h_rms={scada_stats.get('h_rms', 0):.4f} | "
                    f"v_rms={scada_stats.get('v_rms', 0):.4f} | "
                    f"sent to {FS_LIVE_COLLECTION} ✓  [10 SCADA stats]"
                )

            if realtime:
                time.sleep(burst_period)

    except KeyboardInterrupt:
        logger.info(
            f"\nStopped by user. Sent {sent} bursts "
            f"(skipped {skipped} already sent)."
        )
        logger.info(f"Resume from burst {state.last_burst_idx() + 1} next time.")
        sys.exit(0)

    logger.info(
        f"\n{'='*60}\n"
        f"  All {sent} bursts sent for {bearing_name}.\n"
        f"  Writing session-end sentinel to {FS_LIVE_COLLECTION}...\n"
        f"{'='*60}"
    )
    fs_writer.mark_session_end(bearing_name)
    logger.info("Done. The serving pipeline will detect the session end.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SCADA Simulator — streams bearing data into the Feature Store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scada_simulator.py --bearing Bearing1_5
  python scada_simulator.py --bearing Bearing1_5 --realtime
  python scada_simulator.py --bearing_path /data/PHM/Bearing1_5 --realtime
  python scada_simulator.py --bearing Bearing1_5 --resume
        """,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--bearing",      type=str,
                     help="Bearing name (looked up in config/bearings.json)")
    src.add_argument("--bearing_path", type=str,
                     help="Explicit path to bearing data folder")

    parser.add_argument("--realtime",     action="store_true", default=False,
                        help="Sleep burst_period seconds between bursts")
    parser.add_argument("--burst_period", type=float, default=BURST_PERIOD_S,
                        help=f"Seconds between bursts (default: {BURST_PERIOD_S})")
    parser.add_argument("--mongo_uri",    type=str,   default=DEFAULT_MONGO_URI)
    parser.add_argument("--db_name",      type=str,   default=DEFAULT_DB_NAME)
    parser.add_argument("--resume",       action="store_true", default=False,
                        help="Resume from last sent burst")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.bearing and not args.bearing_path:
        print("Error: provide --bearing <name> or --bearing_path <path>")
        sys.exit(1)

    bearing_name = args.bearing or Path(args.bearing_path).name
    bearing_path = _resolve_bearing_path(args.bearing, args.bearing_path)

    run_simulator(
        bearing_name  = bearing_name,
        bearing_path  = bearing_path,
        mongo_uri     = args.mongo_uri,
        db_name       = args.db_name,
        realtime      = args.realtime,
        burst_period  = args.burst_period,
        resume        = args.resume,
    )