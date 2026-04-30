"""
run_serving.py
═══════════════════════════════════════════════════════════════════════════════
Serving Entry Point — runs independently of Pre-Production.

This script is the ONLY process that owns the Serving Pipeline. It:

1. Clears any stale bursts/sentinels left in live_features from previous
   sessions BEFORE starting predictions — ensures only fresh SCADA data
   is processed
2. Polls the Feature Store ('live_features' collection) for new bursts
   written by scada_simulator.py
3. Passes each burst through the 4-stage Serving Pipeline:
       Feature Engineering → Inference → Predictive Maintenance → Monitoring
4. Writes full prediction audit to RUL_predictions (one doc per burst)
5. Writes operational telemetry to serving_history (latency, throughput,
   CPU, model version, feedback metadata — one doc per burst)
6. Writes monitoring metrics back to Feature Store ('feature_store')
7. Checks model_registry/champion.json ONLY after receiving the first burst
   from the current SCADA session — prevents stale champion detections on
   startup before any data has arrived
8. Detects the session-end sentinel and idles until the next SCADA session

MongoDB collections written by this script
──────────────────────────────────────────
RUL_predictions   ← full prediction audit log (features, inference, PM, monitoring)
serving_history   ← operational telemetry (latency, CPU, throughput, model version)
feature_store     ← monitoring metrics written back per burst (Point 3)

Architecture:
    [scada_simulator.py] --> [FS: feature_store / live_features]
                                      |
                                 [run_serving.py]  <-- champion.json (hot-swap)
                                      |
                    ┌─────────────────┴──────────────────┐
                    ▼                                     ▼
            RUL_predictions                       serving_history
         (full prediction audit)              (operational telemetry)

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
# Both SCADA bursts (live) and monitoring metrics written-back live in
# the same 'feature_store' collection — matching COL_FEATURE_STORE in db_collections.
from utils.db_collections import COL_FEATURE_STORE
FS_LIVE_COLLECTION = COL_FEATURE_STORE   # "feature_store"
FS_MONITOR_COLL    = COL_FEATURE_STORE   # monitoring metrics written back to same collection
CHAMPION_PATH      = os.path.join("model_registry", "champion.json")
DEFAULT_POLL_S     = 2.0
DEFAULT_BEARING    = "Bearing1_1"


# ─────────────────────────────────────────────────────────────────────────────
# System-side feature derivation
#
# The SCADA simulator sends 5 simple stats per axis (10 values total).
# Your system derives the remaining 4 stats per axis (8 values) here,
# completing the full 18-feature vector before the pipeline runs.
# ─────────────────────────────────────────────────────────────────────────────

def _derive_features(scada_stats: dict) -> dict:
    """
    Derive the full 18-feature dict from the 10 SCADA-sent stats.

    SCADA sends (10):   {prefix}_max, min, mean, sd, rms   (per axis)
    System derives (8): {prefix}_skew, kurt, crest, form   (per axis)

    Crest and form are computed exactly from the received rms/mean values.
    Skew and kurt require the full signal distribution — they are set to 0.0
    as a neutral approximation (same as the training-time default for
    bearings where these values were unavailable).

    Returns the complete 18-feature dict ready for the pipeline.
    """
    features = dict(scada_stats)   # start with the 10 SCADA stats

    for prefix in ("h", "v"):
        rms  = scada_stats.get(f"{prefix}_rms",  0.0)
        mean = scada_stats.get(f"{prefix}_mean", 0.0)
        mx   = scada_stats.get(f"{prefix}_max",  0.0)

        features[f"{prefix}_skew"]  = 0.0
        features[f"{prefix}_kurt"]  = 0.0
        features[f"{prefix}_crest"] = mx / rms   if rms  > 1e-9 else 0.0
        features[f"{prefix}_form"]  = rms / mean if abs(mean) > 1e-9 else 0.0

    return features   # 18 values total


# ─────────────────────────────────────────────────────────────────────────────
# Champion pointer — atomic model hot-swap
# ─────────────────────────────────────────────────────────────────────────────

class ChampionWatcher:
    """
    Watches model_registry/champion.json for changes between bursts.

    When run_preprod.py promotes a new model it atomically writes champion.json.
    This watcher detects the change and signals the pipeline to reload.

    IMPORTANT: initialise_baseline() must be called after the first burst
    arrives from SCADA — not on startup. This prevents detecting the existing
    champion.json from a previous session as a "new" champion before any data
    has been received.
    """

    def __init__(self, champion_path: str = CHAMPION_PATH):
        self._path        = champion_path
        self._last_mtime  = None
        self._last_model  = None
        self._initialised = False

    def initialise_baseline(self):
        """
        Record the current champion.json state as the baseline.
        Call this once after the first burst arrives from SCADA.
        From this point on, check_for_new_champion() will only fire
        when the file actually changes during this session.
        """
        if self._initialised:
            return
        if os.path.exists(self._path):
            try:
                self._last_mtime = os.path.getmtime(self._path)
                self._last_model = self._read()
                logger.info(
                    f"[ChampionWatcher] Baseline set — current champion: "
                    f"{self._last_model.get('model_id') if self._last_model else 'none'}"
                )
            except OSError:
                pass
        self._initialised = True

    def _read(self) -> Optional[dict]:
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception:
            return None

    def check_for_new_champion(self) -> Optional[dict]:
        """
        Returns the new champion dict if champion.json changed since baseline,
        else None. Call this between bursts — never mid-prediction.
        Only works after initialise_baseline() has been called.
        """
        if not self._initialised:
            return None
        if not os.path.exists(self._path):
            return None
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return None

        if mtime != self._last_mtime:
            champion = self._read()
            if champion and champion != self._last_model:
                self._last_mtime = mtime
                self._last_model = champion
                return champion
        return None

    def current_champion_id(self) -> Optional[str]:
        c = self._read()
        return c.get("model_id") if c else None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Store reader
# ─────────────────────────────────────────────────────────────────────────────

class LiveFeatureStoreReader:
    """
    Reads unconsumed bursts from the live_features MongoDB collection.
    Marks each burst as consumed after the serving pipeline processes it.
    """

    def __init__(self, mongo_uri: str, db_name: str):
        from pymongo import MongoClient, ASCENDING
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._db     = self._client[db_name]
        self._col    = self._db[FS_LIVE_COLLECTION]
        self._mon    = self._db[FS_MONITOR_COLL]

        # Ensure monitoring metrics indexes
        self._mon.create_index(
            [("bearing_name", ASCENDING), ("burst_idx", ASCENDING)],
            name="idx_mon_bearing_burst",
        )
        logger.info(f"[FS Reader] Connected → {db_name}.{FS_LIVE_COLLECTION}")

    def clear_stale_data(self, bearing_name: str) -> int:
        """
        Delete ALL documents for this bearing from live_features — including
        consumed bursts, unconsumed bursts, and session-end sentinels.

        Called on startup before waiting for new SCADA data. This ensures:
        - No stale bursts from a previous run are accidentally consumed
        - No old session-end sentinel causes an immediate idle state
        - The serving pipeline only processes bursts from the current session

        Returns the number of documents deleted.
        """
        result = self._col.delete_many({"bearing_name": bearing_name})
        if result.deleted_count > 0:
            logger.info(
                f"[FS Reader] Cleared {result.deleted_count} stale document(s) "
                f"for {bearing_name} from previous session."
            )
        else:
            logger.info(
                f"[FS Reader] No stale documents found for {bearing_name} — "
                f"clean start."
            )
        return result.deleted_count

    def next_burst(self, bearing_name: str) -> Optional[dict]:
        """
        Return the next unconsumed burst for this bearing (lowest burst_idx),
        or None if nothing is available yet. Excludes sentinel documents.
        The returned doc contains 'scada_stats' (10 values) — caller calls
        _derive_features() to get the full 18-feature dict.
        """
        return self._col.find_one(
            {
                "bearing_name": bearing_name,
                "consumed":     False,
                "session_end":  {"$exists": False},
            },
            sort=[("burst_idx", 1)],
        )

    def mark_consumed(self, doc_id) -> None:
        """Mark a burst document as consumed by the serving pipeline."""
        self._col.update_one(
            {"_id": doc_id},
            {"$set": {
                "consumed":    True,
                "consumed_at": datetime.now(timezone.utc).isoformat(),
            }},
        )

    def check_session_end(self, bearing_name: str) -> bool:
        """Return True if the SCADA simulator has sent the session-end sentinel."""
        return self._col.find_one({
            "bearing_name": bearing_name,
            "session_end":  True,
        }) is not None

    def write_monitoring_metrics(
        self,
        bearing_name:   str,
        burst_idx:      int,
        run_id:         str,
        monitoring_out: dict,
        pm_out:         dict,
        inference_out:  dict,
    ) -> None:
        """
        Write monitoring metrics back to the Feature Store (Point 3).
        Collection: feature_store
        """
        doc = {
            "bearing_name":   bearing_name,
            "burst_idx":      burst_idx,
            "run_id":         run_id,
            "recorded_at":    datetime.now(timezone.utc).isoformat(),
            "type":           "monitoring_metrics",
            # Monitoring stage output
            "drift_detected": monitoring_out.get("drift_detected", False),
            "drift_features": monitoring_out.get("drift_features", []),
            "anomaly_flag":   monitoring_out.get("anomaly_flag", False),
            "baseline_ready": monitoring_out.get("baseline_ready", False),
            "stats":          monitoring_out.get("stats", {}),
            # PM stage output
            "pm_status":      pm_out.get("status"),
            "rul_s":          pm_out.get("rul_s"),
            "rul_min":        pm_out.get("rul_min"),
            "alert":          pm_out.get("alert", False),
            # Inference
            "model_version":  inference_out.get("model_version"),
            "data_quality":   inference_out.get("data_quality"),
        }
        try:
            self._mon.insert_one(doc)
        except Exception as e:
            logger.warning(f"Could not write monitoring metrics: {e}")

    def pending_count(self, bearing_name: str) -> int:
        return self._col.count_documents({
            "bearing_name": bearing_name,
            "consumed":     False,
            "session_end":  {"$exists": False},
        })

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
    Main serving loop. Runs until session-end sentinel is received or Ctrl+C.
    """
    logger.info("=" * 60)
    logger.info("  PHM MLOps — Serving Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Bearing       : {bearing_name}")
    logger.info(f"  MongoDB       : {mongo_uri} / {db_name}")
    logger.info(f"  Poll interval : {poll_interval}s")
    logger.info(f"  Window size   : {window_size}")
    logger.info(f"  Champion file : {CHAMPION_PATH}")
    logger.info("=" * 60)

    # ── Connect to FS ─────────────────────────────────────────────────────────
    fs_reader = LiveFeatureStoreReader(mongo_uri=mongo_uri, db_name=db_name)
    if not fs_reader.ping():
        logger.error("Cannot reach MongoDB. Is it running?")
        sys.exit(1)

    # ── Clear stale data from previous sessions ───────────────────────────────
    logger.info(f"Clearing stale data for {bearing_name} from previous sessions...")
    fs_reader.clear_stale_data(bearing_name)

    # ── Initialise Serving Pipeline ───────────────────────────────────────────
    try:
        from serving_pipeline.serving_pipeline import ServingPipeline
    except ImportError as e:
        logger.error(f"Cannot import ServingPipeline: {e}")
        sys.exit(1)

    pipeline = ServingPipeline(config={
        "mongo_uri":   mongo_uri,
        "db_name":     db_name,
        "window_size": window_size,
    })

    # ── Telemetry writer (serving_history collection) ─────────────────────────
    from utils.serving_history import ServingTelemetry
    telemetry = ServingTelemetry(mongo_uri=mongo_uri, db_name=db_name)

    # ── Champion watcher ──────────────────────────────────────────────────────
    # Baseline is NOT set here — set after the first burst arrives.
    # This prevents the existing champion.json triggering a false hot-swap
    # on startup before any SCADA data has been received.
    champion_watcher = ChampionWatcher(CHAMPION_PATH)
    run_id           = f"serve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    first_burst_seen = False

    logger.info(f"Run ID: {run_id}")
    logger.info(
        f"Stale data cleared — waiting for fresh bursts from SCADA simulator...\n"
    )

    burst_count = 0
    idle_logged = False

    try:
        while True:

            # ── Check for session end ─────────────────────────────────────────
            # Only stop when sentinel exists AND all bursts have been consumed
            if fs_reader.check_session_end(bearing_name):
                remaining = fs_reader.pending_count(bearing_name)
                if remaining == 0:
                    logger.info(
                        f"\n{'=' * 60}\n"
                        f"  Session end detected for {bearing_name}.\n"
                        f"  Processed {burst_count} bursts total.\n"
                        f"  Waiting for next SCADA session (Ctrl+C to exit)...\n"
                        f"{'=' * 60}"
                    )
                    while True:
                        time.sleep(poll_interval * 5)
                # else: sentinel exists but bursts remain — keep processing

            # ── Poll for next burst ───────────────────────────────────────────
            burst_doc = fs_reader.next_burst(bearing_name)

            if burst_doc is None:
                if not idle_logged:
                    logger.info(
                        f"No bursts available — polling every {poll_interval}s..."
                    )
                    idle_logged = True
                time.sleep(poll_interval)
                continue

            # ── First burst received — set champion baseline now ──────────────
            if not first_burst_seen:
                champion_watcher.initialise_baseline()
                first_burst_seen = True
                logger.info(
                    "First burst received from SCADA — "
                    "champion baseline set, hot-swap monitoring active."
                )

            # ── Hot-swap check (between bursts — never mid-prediction) ────────
            new_champion = champion_watcher.check_for_new_champion()
            if new_champion:
                logger.info(
                    f"[HotSwap] New champion detected: "
                    f"{new_champion.get('model_id')} "
                    f"(promoted at {new_champion.get('promoted_at', 'unknown')})"
                )
                pipeline._inference.reload_model()
                logger.info("[HotSwap] Model reloaded — continuing predictions ✓")

            idle_logged = False
            burst_idx   = burst_doc["burst_idx"]
            scada_stats = burst_doc["scada_stats"]

            # ── Derive full 18-feature dict ───────────────────────────────────
            features = _derive_features(scada_stats)

            # ── Run the full 4-stage pipeline (timed for latency) ─────────────
            h_signal = np.array([features.get("h_rms", 0.0)], dtype=np.float32)
            v_signal = np.array([features.get("v_rms", 0.0)], dtype=np.float32)

            t_start = time.time()
            result  = pipeline.run_burst(
                run_id               = run_id,
                bearing_name         = bearing_name,
                burst_idx            = burst_idx,
                h_signal             = h_signal,
                v_signal             = v_signal,
                precomputed_features = features,
            )
            latency_ms = (time.time() - t_start) * 1000.0

            # ── Mark burst as consumed in FS ──────────────────────────────────
            fs_reader.mark_consumed(burst_doc["_id"])

            pm         = result.get("pm")         or {}
            monitoring = result.get("monitoring") or {}
            inference  = result.get("inference")  or {}
            ok         = result.get("ok",    False)
            ready      = result.get("ready", False)

            # ── Write monitoring metrics back to feature_store (Point 3) ──────
            if ok and ready:
                fs_reader.write_monitoring_metrics(
                    bearing_name   = bearing_name,
                    burst_idx      = burst_idx,
                    run_id         = run_id,
                    monitoring_out = monitoring,
                    pm_out         = pm,
                    inference_out  = inference,
                )
                burst_count += 1

            # ── Write operational telemetry to serving_history ────────────────
            try:
                import psutil
                proc    = psutil.Process()
                cpu_pct = proc.cpu_percent(interval=None)
                mem_mb  = proc.memory_info().rss / 1024 / 1024
            except Exception:
                cpu_pct = None
                mem_mb  = None

            telemetry.record(
                run_id              = run_id,
                bearing_name        = bearing_name,
                burst_idx           = burst_idx,
                model_version       = champion_watcher.current_champion_id() or "unknown",
                latency_ms          = latency_ms,
                pipeline_ok         = ok,
                pm_status           = pm.get("status", "unknown"),
                rul_s               = pm.get("rul_s"),
                rul_min             = pm.get("rul_min"),
                drift_detected      = monitoring.get("drift_detected", False),
                anomaly_flag        = monitoring.get("anomaly_flag", False),
                bursts_this_session = burst_count,
                cpu_percent         = cpu_pct,
                memory_mb           = mem_mb,
            )

            # ── Log and check for critical threshold ──────────────────────────
            if ok and ready:
                logger.info(
                    f"  Burst {burst_idx:>4} | "
                    f"RUL={pm.get('rul_min', 0.0):>10.1f} min | "
                    f"status={pm.get('status', '—'):<8} | "
                    f"latency={latency_ms:.1f}ms | "
                    f"drift={monitoring.get('drift_detected', False)} | "
                    f"alert={pm.get('alert', False)}"
                )

                if pm.get("status") == "critical":
                    logger.warning(
                        f"\n{'!'*60}\n"
                        f"  CRITICAL RUL THRESHOLD REACHED for {bearing_name}!\n"
                        f"  RUL = {pm.get('rul_min', '?')} min\n"
                        f"  Stopping serving — awaiting maintenance worker action.\n"
                        f"{'!'*60}"
                    )
                    break

            elif not ready:
                logger.debug(
                    f"  Burst {burst_idx}: pipeline not ready yet "
                    f"(window warming up — {burst_count + 1}/{window_size})"
                )
            else:
                logger.warning(
                    f"  Burst {burst_idx}: pipeline error — {result.get('error')}"
                )

    except KeyboardInterrupt:
        logger.info(
            f"\nServing stopped by user. Processed {burst_count} bursts."
        )

    logger.info("Serving pipeline exited.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Serving Pipeline — polls Feature Store and predicts RUL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bearing",       type=str,   default=DEFAULT_BEARING,
                        help=f"Bearing name to serve (default: {DEFAULT_BEARING})")
    parser.add_argument("--mongo_uri",     type=str,   default=DEFAULT_MONGO_URI)
    parser.add_argument("--db_name",       type=str,   default=DEFAULT_DB_NAME)
    parser.add_argument("--poll_interval", type=float, default=DEFAULT_POLL_S,
                        help=f"Seconds between FS polls when idle (default: {DEFAULT_POLL_S})")
    parser.add_argument("--window_size",   type=int,   default=40,
                        help="Feature engineering window size (default: 40)")
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