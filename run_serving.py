"""
run_serving.py
═══════════════════════════════════════════════════════════════════════════════
Serving Entry Point — runs independently of Pre-Production.

Fix #2  — Critical RUL threshold NO LONGER stops predictions. The pipeline
           continues serving every burst regardless of PM status. The critical
           alert is logged and written to Serving History so the dashboard can
           display it, but serving never breaks.
Fix #5  — Writes a serving_lock document to MongoDB before each burst and
           removes it after. model_registry.write_champion_pointer() waits
           until the lock is clear before swapping the model file.

This script is the ONLY process that owns the Serving Pipeline. It:

1. Clears any stale bursts/sentinels left in live_features from previous
   sessions BEFORE starting predictions — ensures only fresh SCADA data
   is processed
2. Polls the Feature Store ('live_features' collection) for new bursts
   written by scada_simulator.py
3. Passes each burst through the 4-stage Serving Pipeline:
       Feature Engineering → Inference → Predictive Maintenance → Monitoring
4. Writes results to Serving History (MongoDB)
5. Writes monitoring metrics back to Feature Store ('monitoring_metrics')
6. Checks model_registry/champion.json ONLY after receiving the first burst
   from the current SCADA session — prevents stale champion detections on
   startup before any data has arrived
7. Detects the session-end sentinel and idles until the next SCADA session
   (does NOT stop — keeps serving until explicitly killed)

Architecture:
    [scada_simulator.py] --> [FS: live_features]
                                      |
                                 [run_serving.py]  <-- champion.json (hot-swap)
                                      |
                         [Serving History + monitoring_metrics]

Usage
─────
    python run_serving.py
    python run_serving.py --bearing Bearing1_5
    python run_serving.py --poll_interval 2.0 --mongo_uri mongodb://localhost:27017

Stop with Ctrl+C.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVING] %(levelname)s — %(message)s",
)
logger = logging.getLogger("run_serving")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MONGO_URI  = "mongodb://localhost:27017"
DEFAULT_DB_NAME    = "phm_mlops"
CHAMPION_PATH      = os.path.join("model_registry", "champion.json")
DEFAULT_POLL_S     = 2.0
DEFAULT_BEARING    = "Bearing1_1"

from utils.db_collections import (
    COL_FEATURE_STORE          as FS_LIVE_COLLECTION,   # SCADA bursts + monitoring
    COL_SERVING_LOCK           as LOCK_COLLECTION,
)
# Monitoring metrics write back into feature_store (same collection as live bursts)
FS_MONITOR_COLL = FS_LIVE_COLLECTION


# ─────────────────────────────────────────────────────────────────────────────
# System-side feature derivation
# ─────────────────────────────────────────────────────────────────────────────

def _derive_features(scada_stats: dict) -> dict:
    """
    Derive the full 18-feature dict from the 10 SCADA-sent stats.

    SCADA sends (10):   {prefix}_max, min, mean, sd, rms   (per axis)
    System derives (8): {prefix}_skew, kurt, crest, form   (per axis)
    """
    features = dict(scada_stats)

    for prefix in ("h", "v"):
        rms  = scada_stats.get(f"{prefix}_rms",  0.0)
        mean = scada_stats.get(f"{prefix}_mean", 0.0)
        mx   = scada_stats.get(f"{prefix}_max",  0.0)

        features[f"{prefix}_skew"]  = 0.0
        features[f"{prefix}_kurt"]  = 0.0
        features[f"{prefix}_crest"] = mx / rms   if rms  > 1e-9 else 0.0
        features[f"{prefix}_form"]  = rms / mean if abs(mean) > 1e-9 else 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Champion pointer — atomic model hot-swap
# ─────────────────────────────────────────────────────────────────────────────

class ChampionWatcher:
    """
    Watches model_registry/champion.json for changes between bursts.
    Fix #5: Hot-swap only checked between bursts (never mid-prediction).
    """

    def __init__(self, champion_path: str = CHAMPION_PATH):
        self._path        = champion_path
        self._last_mtime  = None
        self._last_model  = None
        self._initialised = False

    def initialise_baseline(self):
        """Record current champion.json state as the baseline."""
        try:
            self._last_mtime = os.path.getmtime(self._path) if os.path.exists(self._path) else None
            if os.path.exists(self._path):
                with open(self._path) as f:
                    self._last_model = json.load(f)
        except Exception:
            pass
        self._initialised = True

    def check_for_new_champion(self) -> Optional[dict]:
        """Return new champion dict if champion.json changed since last check, else None."""
        if not self._initialised:
            return None
        try:
            if not os.path.exists(self._path):
                return None
            mtime = os.path.getmtime(self._path)
            if self._last_mtime is None or mtime > self._last_mtime:
                with open(self._path) as f:
                    new_champion = json.load(f)
                if new_champion.get("model_id") != (
                    self._last_model.get("model_id") if self._last_model else None
                ):
                    self._last_mtime = mtime
                    self._last_model = new_champion
                    return new_champion
                self._last_mtime = mtime
        except Exception as e:
            logger.warning(f"ChampionWatcher error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Store reader + serving lock (Fix #5)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureStoreReader:
    """
    Reads bursts from the live Feature Store (MongoDB live_features collection).
    Manages the serving_lock collection for safe model hot-swap (Fix #5).
    """

    def __init__(self, mongo_uri: str, db_name: str):
        from pymongo import MongoClient, ASCENDING
        self._client   = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._db       = self._client[db_name]
        self._col      = self._db[FS_LIVE_COLLECTION]
        self._mon      = self._db[FS_MONITOR_COLL]
        self._lock_col = self._db[LOCK_COLLECTION]

        # Monitoring metrics share the feature_store collection with burst docs.
        # They are distinguished by doc_type="monitoring" vs doc_type="burst".
        # Index is NOT unique — both types share bearing_name+burst_idx keys.
        self._mon.create_index(
            [("bearing_name", ASCENDING), ("burst_idx", ASCENDING),
             ("doc_type", ASCENDING)],
            name="idx_mon_bearing_burst_type",
        )
        logger.info(f"[FS Reader] Connected → {db_name}.{FS_LIVE_COLLECTION}")

    def clear_stale_data(self, bearing_name: str) -> int:
        result = self._col.delete_many({"bearing_name": bearing_name})
        # Also clear any stale lock from a previous crashed session
        self._lock_col.delete_many({"bearing_name": bearing_name})
        if result.deleted_count > 0:
            logger.info(
                f"[FS Reader] Cleared {result.deleted_count} stale document(s) "
                f"for {bearing_name} from previous session."
            )
        else:
            logger.info(f"[FS Reader] No stale documents — clean start.")
        return result.deleted_count

    def next_burst(self, bearing_name: str) -> Optional[dict]:
        return self._col.find_one(
            {"bearing_name": bearing_name,
             "consumed":     False,
             "session_end":  {"$exists": False}},
            sort=[("burst_idx", 1)],
        )

    def mark_consumed(self, doc_id) -> None:
        self._col.update_one(
            {"_id": doc_id},
            {"$set": {"consumed": True,
                       "consumed_at": datetime.now(timezone.utc).isoformat()}},
        )

    def check_session_end(self, bearing_name: str) -> bool:
        return self._col.find_one(
            {"bearing_name": bearing_name, "session_end": True}
        ) is not None

    def pending_count(self, bearing_name: str) -> int:
        return self._col.count_documents({
            "bearing_name": bearing_name,
            "consumed":     False,
            "session_end":  {"$exists": False},
        })

    # ── Serving lock — Fix #5 ─────────────────────────────────────────────────

    def acquire_burst_lock(self, bearing_name: str, burst_idx: int) -> None:
        """Signal that serving is mid-burst. model_registry will wait before swapping."""
        try:
            self._lock_col.update_one(
                {"bearing_name": bearing_name},
                {"$set": {"active": True,
                           "burst_idx": burst_idx,
                           "locked_at": datetime.now(timezone.utc).isoformat()}},
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"Could not acquire serving lock: {e}")

    def release_burst_lock(self, bearing_name: str) -> None:
        """Signal that the burst is complete — model swap is now safe."""
        try:
            self._lock_col.update_one(
                {"bearing_name": bearing_name},
                {"$set": {"active": False,
                           "released_at": datetime.now(timezone.utc).isoformat()}},
            )
        except Exception as e:
            logger.warning(f"Could not release serving lock: {e}")

    def write_monitoring_metrics(
        self,
        bearing_name:   str,
        burst_idx:      int,
        run_id:         str,
        monitoring_out: dict,
        pm_out:         dict,
        inference_out:  dict,
    ) -> None:
        doc = {
            "doc_type":       "monitoring",          # distinguishes from burst docs
            "bearing_name":   bearing_name,
            "burst_idx":      burst_idx,
            "run_id":         run_id,
            "recorded_at":    datetime.now(timezone.utc).isoformat(),
            "drift_detected": monitoring_out.get("drift_detected", False),
            "drift_features": monitoring_out.get("drift_features", []),
            "anomaly_flag":   monitoring_out.get("anomaly_flag", False),
            "baseline_ready": monitoring_out.get("baseline_ready", False),
            "stats":          monitoring_out.get("stats", {}),
            "pm_status":      pm_out.get("status"),
            "rul_s":          pm_out.get("rul_s"),
            "rul_min":        pm_out.get("rul_min"),
            "alert":          pm_out.get("alert", False),
            "model_version":  inference_out.get("model_version"),
            "data_quality":   inference_out.get("data_quality"),
        }
        try:
            self._mon.insert_one(doc)
        except Exception as e:
            logger.warning(f"Could not write monitoring metrics: {e}")

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Main serving loop
# ─────────────────────────────────────────────────────────────────────────────

def run_serving(
    bearing_name:  str,
    mongo_uri:     str,
    db_name:       str,
    poll_interval: float,
    window_size:   int,
):
    """
    Main serving loop. Runs until Ctrl+C.

    Fix #2: NEVER stops on critical status — keeps predicting every burst.
    Fix #5: Acquires/releases serving_lock around each burst so model_registry
            can safely swap champion.json between bursts.
    """
    from serving_pipeline.serving_pipeline import ServingPipeline

    fs_reader = FeatureStoreReader(mongo_uri, db_name)

    # ── Startup: clear stale data ─────────────────────────────────────────────
    fs_reader.clear_stale_data(bearing_name)

    # ── Initialise serving pipeline ───────────────────────────────────────────
    pipeline = ServingPipeline(config={
        "mongo_uri":              mongo_uri,
        "db_name":                db_name,
        "window_size":            window_size,
        "critical_threshold_s":   3600,
        "warning_threshold_s":    14400,
        "baseline_path":          "model_registry/monitoring_baseline.json",
        "enable_serving_history": True,
    })

    champion_watcher = ChampionWatcher(CHAMPION_PATH)
    run_id           = f"serve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    first_burst_seen = False

    logger.info(f"Run ID: {run_id}")
    logger.info("Stale data cleared — waiting for fresh bursts from SCADA simulator...\n")

    burst_count = 0
    idle_logged = False

    try:
        while True:

            # ── Session-end sentinel — idle, but NEVER stop (Fix #2) ──────────
            if fs_reader.check_session_end(bearing_name):
                remaining = fs_reader.pending_count(bearing_name)
                if remaining == 0:
                    if not idle_logged:
                        logger.info(
                            f"\n{'=' * 60}\n"
                            f"  Session end detected for {bearing_name}.\n"
                            f"  Processed {burst_count} bursts total.\n"
                            f"  Idling — waiting for next SCADA session (Ctrl+C to exit).\n"
                            f"{'=' * 60}"
                        )
                        idle_logged = True
                    time.sleep(poll_interval * 5)
                    continue

            # ── Poll for next burst ───────────────────────────────────────────
            burst_doc = fs_reader.next_burst(bearing_name)

            if burst_doc is None:
                if not idle_logged:
                    logger.info(f"No bursts available — polling every {poll_interval}s...")
                    idle_logged = True
                time.sleep(poll_interval)
                continue

            idle_logged = False   # reset once we have data

            # ── First burst — set champion baseline ───────────────────────────
            if not first_burst_seen:
                champion_watcher.initialise_baseline()
                first_burst_seen = True
                logger.info(
                    "First burst received from SCADA — "
                    "champion baseline set, hot-swap monitoring active."
                )

            burst_idx   = burst_doc["burst_idx"]
            scada_stats = burst_doc["scada_stats"]

            # ── Acquire serving lock BEFORE processing (Fix #5) ──────────────
            fs_reader.acquire_burst_lock(bearing_name, burst_idx)

            try:
                # ── Hot-swap check (between bursts — never mid-prediction) ────
                # Checked AFTER lock acquired so we always finish the current burst
                # before accepting a new model.
                new_champion = champion_watcher.check_for_new_champion()
                if new_champion:
                    logger.info(
                        f"[HotSwap] New champion detected: "
                        f"{new_champion.get('model_id')} "
                        f"(promoted at {new_champion.get('promoted_at', 'unknown')})"
                    )
                    pipeline._inference.reload_model()
                    logger.info("[HotSwap] Model reloaded — continuing predictions ✓")

                # ── Derive full 18-feature dict ───────────────────────────────
                features = _derive_features(scada_stats)

                h_signal = np.array([features.get("h_rms", 0.0)], dtype=np.float32)
                v_signal = np.array([features.get("v_rms", 0.0)], dtype=np.float32)

                result = pipeline.run_burst(
                    run_id               = run_id,
                    bearing_name         = bearing_name,
                    burst_idx            = burst_idx,
                    h_signal             = h_signal,
                    v_signal             = v_signal,
                    precomputed_features = features,
                )

            finally:
                # ── Release lock AFTER burst completes (Fix #5) ───────────────
                fs_reader.release_burst_lock(bearing_name)

            # ── Mark burst consumed ───────────────────────────────────────────
            fs_reader.mark_consumed(burst_doc["_id"])

            # ── Write monitoring metrics ──────────────────────────────────────
            if result.get("ok") and result.get("ready"):
                fs_reader.write_monitoring_metrics(
                    bearing_name   = bearing_name,
                    burst_idx      = burst_idx,
                    run_id         = run_id,
                    monitoring_out = result.get("monitoring") or {},
                    pm_out         = result.get("pm") or {},
                    inference_out  = result.get("inference") or {},
                )
                burst_count += 1
                pm = result.get("pm") or {}

                rul_min  = pm.get("rul_min")
                rul_str  = f"{rul_min:>10.1f}" if rul_min is not None else "       N/A"
                pm_status = pm.get("status") or "unknown"
                log_fn = logger.warning if pm_status == "critical" else logger.info
                log_fn(
                    f"  Burst {burst_idx:>4} | "
                    f"RUL={rul_str} min | "
                    f"status={pm_status:<8} | "
                    f"drift={result.get('monitoring', {}).get('drift_detected', False)} | "
                    f"alert={pm.get('alert', False)}"
                )

                # Fix #2 — Log critical alert but CONTINUE PREDICTING.
                # Do NOT break or stop the loop.
                if pm.get("status") == "critical":
                    logger.warning(
                        f"\n{'!'*60}\n"
                        f"  CRITICAL RUL THRESHOLD REACHED for {bearing_name}!\n"
                        f"  RUL = {pm.get('rul_min', '?')} min\n"
                        f"  Predictions continue — maintenance worker must act via dashboard.\n"
                        f"{'!'*60}"
                    )

            elif not result.get("ready"):
                logger.debug(
                    f"  Burst {burst_idx}: pipeline warming up "
                    f"({burst_count + 1}/{window_size})"
                )
            else:
                logger.warning(
                    f"  Burst {burst_idx}: pipeline error — {result.get('error')}"
                )

    except KeyboardInterrupt:
        logger.info(f"\nServing stopped by user. Processed {burst_count} bursts.")
        # Release lock if we were interrupted mid-burst
        try:
            fs_reader.release_burst_lock(bearing_name)
        except Exception:
            pass

    logger.info("Serving pipeline exited.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Serving Pipeline — polls Feature Store and predicts RUL."
    )
    parser.add_argument(
        "--bearing", default=DEFAULT_BEARING,
        help=f"Bearing name to serve (default: {DEFAULT_BEARING})",
    )
    parser.add_argument(
        "--poll_interval", type=float, default=DEFAULT_POLL_S,
        help=f"Seconds between FS polls when idle (default: {DEFAULT_POLL_S})",
    )
    parser.add_argument(
        "--mongo_uri", default=DEFAULT_MONGO_URI,
        help=f"MongoDB URI (default: {DEFAULT_MONGO_URI})",
    )
    parser.add_argument(
        "--db_name", default=DEFAULT_DB_NAME,
        help=f"MongoDB database (default: {DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "--window_size", type=int, default=40,
        help="Rolling window size for inference (default: 40)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_serving(
        bearing_name  = args.bearing,
        mongo_uri     = args.mongo_uri,
        db_name       = args.db_name,
        poll_interval = args.poll_interval,
        window_size   = args.window_size,
    )