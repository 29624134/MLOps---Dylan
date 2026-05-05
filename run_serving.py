"""
run_serving.py
═══════════════════════════════════════════════════════════════════════════════
Serving Entry Point — group-aware, runs independently of Pre-Production.

Each bearing group has its own champion file:
    model_registry/champion_bearing1.json  ← Group 1
    model_registry/champion_bearing2.json  ← Group 2
    model_registry/champion_bearing3.json  ← Group 3

This script is launched once per bearing by ProcessManager (API.py), with
the --champion flag pointing at the correct group file. That way:
    - Bearing1_x → loads champion_bearing1.json
    - Bearing2_x → loads champion_bearing2.json
    - Bearing3_x → loads champion_bearing3.json

When run_preprod.py promotes a new model for Group 1, only the
run_serving.py instances watching champion_bearing1.json hot-swap.
Group 2 and Group 3 are completely unaffected.

If --champion is not specified, the group is inferred from the bearing name
(e.g. "Bearing2_4" → group "2" → champion_bearing2.json).

This script:
1. Clears stale bursts/sentinels from previous sessions BEFORE predicting
2. Polls the shared Feature Store for new bursts (filtered by bearing_name)
3. Passes each burst through the 4-stage Serving Pipeline:
       Feature Engineering → Inference → Predictive Maintenance → Monitoring
4. Writes full prediction audit to RUL_predictions (one doc per burst)
5. Writes operational telemetry to serving_history
6. Writes monitoring metrics back to Feature Store
7. Watches the group champion file for changes → hot-swaps model between bursts
8. Detects session-end sentinel and idles until next SCADA session

Behaviour on CRITICAL status
─────────────────────────────
When RUL drops below the critical threshold the pipeline logs a prominent
warning and continues predicting every subsequent burst. Serving only stops
on a session-end sentinel (all SCADA data consumed) or Ctrl+C. The
maintenance decision (confirm / deny fault) is made by the human operator
via the Fault Review dashboard — NOT by this script.

Usage
─────
    # Champion inferred from bearing name (preferred):
    python run_serving.py --bearing Bearing1_5

    # Champion specified explicitly:
    python run_serving.py --bearing Bearing2_4 --champion model_registry/champion_bearing2.json

    # With options:
    python run_serving.py --bearing Bearing3_1 --poll_interval 2.0 --window_size 40

Stop with Ctrl+C.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import re
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
DEFAULT_MONGO_URI   = "mongodb://localhost:27017"
DEFAULT_DB_NAME     = "phm_mlops"
MODEL_REGISTRY_DIR  = "model_registry"
DEFAULT_POLL_S      = 2.0
DEFAULT_BEARING     = "Bearing1_1"

from utils.db_collections import COL_FEATURE_STORE
FS_LIVE_COLLECTION = COL_FEATURE_STORE
FS_MONITOR_COLL    = COL_FEATURE_STORE


# ─────────────────────────────────────────────────────────────────────────────
# Group / champion resolution
# ─────────────────────────────────────────────────────────────────────────────

def _group_from_bearing(bearing_name: str) -> Optional[str]:
    """
    Infer group ID from bearing name.
    e.g. "Bearing1_5" → "1", "Bearing2_4" → "2", "Bearing3_1" → "3"
    Returns None if the name doesn't match the expected pattern.
    """
    m = re.match(r"Bearing(\d+)_\d+", bearing_name, re.IGNORECASE)
    return m.group(1) if m else None


def _default_champion_path(bearing_name: str) -> str:
    """
    Return the default group-specific champion path for a bearing.
    Falls back to generic champion.json if group cannot be inferred.
    """
    group = _group_from_bearing(bearing_name)
    if group:
        return os.path.join(MODEL_REGISTRY_DIR, f"champion_bearing{group}.json")
    return os.path.join(MODEL_REGISTRY_DIR, "champion.json")


# ─────────────────────────────────────────────────────────────────────────────
# Feature derivation — system adds 8 stats to SCADA's 10
# ─────────────────────────────────────────────────────────────────────────────

def _derive_features(scada_stats: dict) -> dict:
    """
    Derive the full 18-feature dict from the 10 SCADA-sent stats.

    SCADA sends (10):   {prefix}_max, min, mean, sd, rms   (per axis)
    System derives (8): {prefix}_skew, kurt, crest, form   (per axis)

    Correct formulae:
        crest = max / rms   (peak-to-RMS ratio)
        form  = rms / mean  (form factor)
    Skew and kurt cannot be derived from simple stats — set to 0.0.
    """
    features = dict(scada_stats)   # copy the 10 SCADA stats
    for prefix in ("h", "v"):
        mx   = features.get(f"{prefix}_max", 0.0)
        rms  = features.get(f"{prefix}_rms",  0.0)
        mean = features.get(f"{prefix}_mean", 0.0)
        features[f"{prefix}_crest"] = mx  / rms  if rms  != 0 else 0.0
        features[f"{prefix}_form"]  = rms / mean if mean != 0 else 0.0
        features[f"{prefix}_skew"]  = 0.0
        features[f"{prefix}_kurt"]  = 0.0
    return features   # 18 values


# ─────────────────────────────────────────────────────────────────────────────
# Champion watcher — detects model hot-swap between bursts
# ─────────────────────────────────────────────────────────────────────────────

class ChampionWatcher:
    """
    Watches the group-specific champion JSON file for changes.
    When a new champion is promoted (mtime changes), returns the new record
    so the serving loop can trigger a model reload.

    Baseline is set after the first burst arrives — not at startup — to
    avoid false positives from files written by a concurrent training thread.
    """

    def __init__(self, champion_path: str):
        self._path        = champion_path
        self._last_mtime  = None
        self._last_model  = None
        self._initialised = False

    def initialise_baseline(self):
        """Call after the first burst arrives to arm the hot-swap detector."""
        if os.path.exists(self._path):
            self._last_mtime = os.path.getmtime(self._path)
            self._last_model = self._read()
        self._initialised = True

    def _read(self) -> Optional[dict]:
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception:
            return None

    def check_for_new_champion(self) -> Optional[dict]:
        """
        Return the new champion record if the file has changed since last check.
        Call this between bursts — never mid-prediction.
        Only active after initialise_baseline() has been called.
        """
        if not self._initialised or not os.path.exists(self._path):
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
    Reads unconsumed bursts for this bearing from the shared feature_store
    collection. Marks each burst as consumed after the serving pipeline
    processes it.

    Multiple run_serving.py instances share the same collection — each only
    sees its own bearing's documents (filtered by bearing_name).
    """

    def __init__(self, mongo_uri: str, db_name: str):
        from pymongo import MongoClient, ASCENDING
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._db     = self._client[db_name]
        self._col    = self._db[FS_LIVE_COLLECTION]
        self._mon    = self._db[FS_MONITOR_COLL]

        self._mon.create_index(
            [("bearing_name", ASCENDING), ("burst_idx", ASCENDING)],
            name="idx_mon_bearing_burst",
        )
        logger.info(f"[FS Reader] Connected → {db_name}.{FS_LIVE_COLLECTION}")

    def clear_stale_data(self, bearing_name: str) -> int:
        """
        Delete ALL documents for this bearing from the feature_store collection —
        including consumed bursts, unconsumed bursts, and session-end sentinels.

        Called on startup before waiting for new SCADA data. Ensures no stale
        bursts from a previous run are accidentally consumed.
        """
        result = self._col.delete_many({"bearing_name": bearing_name})
        if result.deleted_count > 0:
            logger.info(
                f"[FS Reader] Cleared {result.deleted_count} stale document(s) "
                f"for {bearing_name} from previous session."
            )
        else:
            logger.info(
                f"[FS Reader] No stale documents for {bearing_name} — clean start."
            )
        return result.deleted_count

    def next_burst(self, bearing_name: str) -> Optional[dict]:
        """
        Return the next unconsumed burst for this bearing (lowest burst_idx),
        or None if nothing is available. Excludes session-end sentinels.
        """
        from pymongo import ASCENDING
        doc = self._col.find_one(
            {
                "bearing_name": bearing_name,
                "consumed":     False,
                "session_end":  {"$exists": False},
            },
            sort=[("burst_idx", ASCENDING)],
        )
        return doc

    def mark_consumed(self, doc_id, derived_features: dict = None):
        """Mark a burst document as consumed. Optionally store derived features."""
        update = {"$set": {"consumed": True, "consumed_at": datetime.now(timezone.utc).isoformat()}}
        if derived_features:
            update["$set"]["derived_features"] = derived_features
        self._col.update_one({"_id": doc_id}, update)

    def check_session_end(self, bearing_name: str) -> bool:
        """Return True if a session-end sentinel exists for this bearing."""
        return self._col.count_documents(
            {"bearing_name": bearing_name, "session_end": True}
        ) > 0

    def write_monitoring_metrics(
        self,
        bearing_name:   str,
        burst_idx:      int,
        run_id:         str,
        monitoring_out: dict,
        pm_out:         dict,
        inference_out:  dict,
    ):
        """Write monitoring metrics back to the Feature Store for this burst."""
        metrics = {
            "run_id":         run_id,
            "recorded_at":    datetime.now(timezone.utc).isoformat(),
            "type":           "monitoring_metrics",
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
            self._mon.update_one(
                {"bearing_name": bearing_name, "burst_idx": burst_idx},
                {"$set": metrics},
                upsert=True,
            )
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
    champion_path: str,
    mongo_uri:     str,
    db_name:       str,
    poll_interval: float,
    window_size:   int,
):
    """
    Main serving loop for one bearing. Runs until session-end sentinel or Ctrl+C.

    Parameters
    ----------
    bearing_name  : str   — bearing being served (e.g. "Bearing1_5")
    champion_path : str   — path to this group's champion JSON file
    mongo_uri     : str   — MongoDB connection string
    db_name       : str   — database name
    poll_interval : float — seconds between Feature Store polls when idle
    window_size   : int   — FE rolling window size (must match training)
    """
    group = _group_from_bearing(bearing_name) or "?"

    logger.info("=" * 60)
    logger.info("  PHM MLOps — Serving Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Bearing       : {bearing_name}")
    logger.info(f"  Group         : {group}")
    logger.info(f"  Champion file : {champion_path}")
    logger.info(f"  MongoDB       : {mongo_uri} / {db_name}")
    logger.info(f"  Poll interval : {poll_interval}s")
    logger.info(f"  Window size   : {window_size}")
    logger.info("=" * 60)

    # ── Connect to Feature Store ──────────────────────────────────────────────
    fs_reader = LiveFeatureStoreReader(mongo_uri=mongo_uri, db_name=db_name)
    if not fs_reader.ping():
        logger.error("Cannot reach MongoDB. Is it running?")
        sys.exit(1)

    # ── Clear stale data from previous sessions ───────────────────────────────
    logger.info(f"Clearing stale data for {bearing_name}...")
    fs_reader.clear_stale_data(bearing_name)

    # ── Initialise Serving Pipeline ───────────────────────────────────────────
    try:
        from serving_pipeline.serving_pipeline import ServingPipeline
    except ImportError as e:
        logger.error(f"Cannot import ServingPipeline: {e}")
        sys.exit(1)

    pipeline = ServingPipeline(config={
        "mongo_uri":     mongo_uri,
        "db_name":       db_name,
        "window_size":   window_size,
        "champion_path": champion_path,
    })

    # ── Telemetry writer ──────────────────────────────────────────────────────
    from utils.serving_history import ServingTelemetry
    telemetry = ServingTelemetry(mongo_uri=mongo_uri, db_name=db_name)

    # ── Champion watcher ──────────────────────────────────────────────────────
    # Baseline is NOT set here — set after the first burst arrives.
    # This prevents detecting the existing file from a previous session.
    champion_watcher = ChampionWatcher(champion_path)
    run_id           = f"serve_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{bearing_name}"
    first_burst_seen = False

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Stale data cleared — waiting for fresh bursts from SCADA...\n")

    burst_count = 0
    idle_logged = False

    try:
        while True:

            # ── Check for session end ─────────────────────────────────────────
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
                # sentinel exists but bursts remain — keep processing

            # ── Poll for next burst ───────────────────────────────────────────
            burst_doc = fs_reader.next_burst(bearing_name)

            if burst_doc is None:
                if not idle_logged:
                    logger.info(
                        f"[{bearing_name}] No bursts available — "
                        f"polling every {poll_interval}s..."
                    )
                    idle_logged = True
                time.sleep(poll_interval)
                continue

            idle_logged = False

            # ── First burst received — set champion baseline now ──────────────
            if not first_burst_seen:
                champion_watcher.initialise_baseline()
                first_burst_seen = True
                logger.info(
                    f"[{bearing_name}] First burst received — "
                    f"champion baseline set ({champion_path}), hot-swap active."
                )

            # ── Hot-swap check (between bursts, never mid-prediction) ─────────
            new_champion = champion_watcher.check_for_new_champion()
            if new_champion:
                logger.info(
                    f"[HotSwap] [{bearing_name}] New Group {group} champion: "
                    f"{new_champion.get('model_id')} "
                    f"(promoted {new_champion.get('promoted_at', '?')})"
                )
                pipeline.reload_model()

            # ── Extract burst data ────────────────────────────────────────────
            # SCADA writes the 10 stats flat on the document (not nested).
            burst_idx   = burst_doc.get("burst_idx", 0)
            _SCADA_KEYS = (
                "h_max", "h_min", "h_mean", "h_sd", "h_rms",
                "v_max", "v_min", "v_mean", "v_sd", "v_rms",
            )
            scada_stats = {k: burst_doc[k] for k in _SCADA_KEYS if k in burst_doc}
            features    = _derive_features(scada_stats)

            # ── Run 4-stage serving pipeline ──────────────────────────────────
            # h_signal/v_signal are required by the pipeline signature but ignored
            # when precomputed_features is provided — SCADA already extracted them.
            t_start    = time.perf_counter()
            result     = pipeline.run_burst(
                run_id               = run_id,
                bearing_name         = bearing_name,
                burst_idx            = burst_idx,
                h_signal             = np.array([0.0], dtype=np.float32),
                v_signal             = np.array([0.0], dtype=np.float32),
                precomputed_features = features,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000

            # ── Mark burst consumed — store the 18 base features ────────────
            # The Feature Store holds raw sensor data only (18 base features).
            # The trainer's _add_rolling_features() computes the rolling stats
            # (mean/std/slope) to produce the full 76-dim vector at training
            # time — exactly as it does for train/val bearings.
            fs_reader.mark_consumed(burst_doc["_id"], derived_features=features)

            pm         = result.get("pm")         or {}
            monitoring = result.get("monitoring") or {}
            inference  = result.get("inference")  or {}
            ok         = result.get("ok",    False)
            ready      = result.get("ready", False)

            # ── Write monitoring metrics back to Feature Store ─────────────────
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

            # ── Operational telemetry → serving_history ───────────────────────
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

            # ── Log result ────────────────────────────────────────────────────
            if ok and ready:
                logger.info(
                    f"  [{bearing_name}] Burst {burst_idx:>4} | "
                    f"RUL={pm.get('rul_min', 0.0):>10.1f} min | "
                    f"status={pm.get('status', '—'):<8} | "
                    f"latency={latency_ms:.1f}ms | "
                    f"drift={monitoring.get('drift_detected', False)} | "
                    f"alert={pm.get('alert', False)}"
                )

                if pm.get("status") == "critical":
                    # Log a prominent warning but DO NOT stop serving.
                    # The maintenance decision is made by the human operator
                    # via the Fault Review dashboard. Serving continues until
                    # the SCADA session ends or the operator intervenes.
                    logger.warning(
                        f"\n{'!'*60}\n"
                        f"  CRITICAL RUL ALERT for {bearing_name}!\n"
                        f"  RUL = {pm.get('rul_min', '?'):.3f} min\n"
                        f"  Continuing predictions — awaiting operator action.\n"
                        f"{'!'*60}"
                    )

            elif not ready:
                logger.debug(
                    f"  [{bearing_name}] Burst {burst_idx}: window warming up "
                    f"({burst_count + 1}/{window_size})"
                )
            else:
                logger.warning(
                    f"  [{bearing_name}] Burst {burst_idx}: pipeline error — "
                    f"{result.get('error')}"
                )

    except KeyboardInterrupt:
        logger.info(
            f"\n[{bearing_name}] Serving stopped by user. "
            f"Processed {burst_count} bursts."
        )

    logger.info(f"[{bearing_name}] Serving pipeline exited.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Serving Pipeline — polls the shared Feature Store and predicts RUL "
            "using the group-specific champion model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Group inferred from bearing name (preferred):
  python run_serving.py --bearing Bearing1_5
  python run_serving.py --bearing Bearing2_4
  python run_serving.py --bearing Bearing3_1

  # Champion specified explicitly:
  python run_serving.py --bearing Bearing1_5 --champion model_registry/champion_bearing1.json

  # With options:
  python run_serving.py --bearing Bearing1_5 --poll_interval 1.0 --window_size 40
        """,
    )
    parser.add_argument(
        "--bearing",
        type=str,
        default=DEFAULT_BEARING,
        help=f"Bearing name to serve (default: {DEFAULT_BEARING})",
    )
    parser.add_argument(
        "--champion",
        type=str,
        default=None,
        help=(
            "Path to group champion JSON file. "
            "If not specified, inferred from bearing name "
            "(e.g. Bearing2_4 → model_registry/champion_bearing2.json)."
        ),
    )
    parser.add_argument("--mongo_uri",     type=str,   default=DEFAULT_MONGO_URI)
    parser.add_argument("--db_name",       type=str,   default=DEFAULT_DB_NAME)
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=DEFAULT_POLL_S,
        help=f"Seconds between FS polls when idle (default: {DEFAULT_POLL_S})",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=40,
        help="Feature engineering window size (default: 40)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve champion path — explicit flag wins, otherwise infer from bearing name
    champion_path = args.champion or _default_champion_path(args.bearing)

    run_serving(
        bearing_name  = args.bearing,
        champion_path = champion_path,
        mongo_uri     = args.mongo_uri,
        db_name       = args.db_name,
        poll_interval = args.poll_interval,
        window_size   = args.window_size,
    )