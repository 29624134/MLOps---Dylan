"""
utils/serving_history.py
═══════════════════════════════════════════════════════════════════════════════
Two MongoDB-backed stores written by run_serving.py after each burst:

RUL_predictions  (COL_RUL_PREDICTIONS)
────────────────────────────────────────
Full prediction audit log — one document per burst.
Stores: features, raw inference output, PM status, monitoring flags,
model version, pipeline version, pipeline health.
Written by: ServingHistory.save_pipeline_output()
Read by:    Dashboard (RUL Monitor), Audit Service, Export Service

serving_history  (COL_SERVING_HISTORY)
────────────────────────────────────────
Operational telemetry — one document per burst.
Stores: latency (ms), throughput (bursts processed), resource usage
(CPU %, memory MB), model version, pipeline version, correction/feedback metadata.
Written by: ServingTelemetry.record()
Read by:    Dashboard (Serving Telemetry page), Audit Service

Pipeline version
────────────────
Both stores record `pipeline_version` (e.g. "V1") on every record so the
audit trail can answer "which workflow definition produced this output"
without having to cross-reference timestamps against the WorkflowRegistry.
Default is "V1"; run_serving.py looks up the active workflow version from
WorkflowRegistry and passes it through, falling back to "V1" if the registry
is unavailable.

Diagram connections:
    ServPipeline ──(step 9)──► RUL_predictions    [ServingHistory]
    ServPipeline ──(step 9)──► serving_history     [ServingTelemetry]
    Both         ──(step 10)─► AuditSvc
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Collection names — always imported from db_collections, never hardcoded
# ─────────────────────────────────────────────────────────────────────────────

from utils.db_collections import COL_RUL_PREDICTIONS, COL_SERVING_HISTORY


# Default pipeline version label, used when callers don't pass one explicitly.
DEFAULT_PIPELINE_VERSION = "V1"


# ─────────────────────────────────────────────────────────────────────────────
# ServingHistory — RUL_predictions collection
# Full prediction audit log, one document per burst
# ─────────────────────────────────────────────────────────────────────────────

class ServingHistory:
    """
    Persists the full pipeline output for every burst to RUL_predictions.

    One document per (run_id, bearing_name, burst_idx) triple.

    Document schema
    ───────────────
    {
      run_id           : str       — workflow run identifier
      bearing_name     : str       — e.g. "Bearing1_5"
      burst_idx        : int       — burst index within the bearing stream
      timestamp        : datetime  — UTC wall-clock time of pipeline run
      model_version    : str       — model_id from ModelRegistry
      pipeline_version : str       — active workflow version (e.g. "V1")
      features         : dict      — quality-labelled feature vector
      inference        : { rul_s, rul_min, horizon_preds, ... }
      pm_status        : str       — "healthy" | "warning" | "critical"
      pm_out           : dict
      monitoring       : dict
      pipeline_ok      : bool
      error            : str|None
    }
    """

    def __init__(self, mongo_uri: str, db_name: str = "phm_mlops"):
        self._uri     = mongo_uri
        self._db_name = db_name
        self._client  = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._col     = self._client[db_name][COL_RUL_PREDICTIONS]
        self._ensure_indexes()
        logger.info(
            f"ServingHistory connected → {db_name}.{COL_RUL_PREDICTIONS}"
        )

    def _ensure_indexes(self):
        try:
            self._col.create_index("run_id",           name="idx_run_id")
            self._col.create_index("bearing_name",     name="idx_bearing_name")
            self._col.create_index("pipeline_version", name="idx_pipeline_version")
            self._col.create_index(
                [("timestamp", DESCENDING)], name="idx_timestamp_desc"
            )
            self._col.create_index(
                [("run_id", ASCENDING), ("bearing_name", ASCENDING)],
                name="idx_run_bearing",
            )
            self._col.create_index("pm_status", name="idx_pm_status")
        except Exception as e:
            logger.warning(f"[ServingHistory] Index creation warning: {e}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_pipeline_output(
        self,
        run_id:           str,
        bearing_name:     str,
        burst_idx:        int,
        model_version:    str,
        features:         Dict[str, Any],
        inference_out:    Dict[str, Any],
        pm_out:           Dict[str, Any],
        monitoring_out:   Dict[str, Any],
        pipeline_ok:      bool = True,
        error:            Optional[str] = None,
        pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    ) -> str:
        """
        Persist one full pipeline output record to RUL_predictions.
        Returns the MongoDB inserted_id as a string.
        """
        doc = {
            "run_id":           run_id,
            "bearing_name":     bearing_name,
            "burst_idx":        burst_idx,
            "timestamp":        datetime.now(timezone.utc),
            "model_version":    model_version,
            "pipeline_version": pipeline_version or DEFAULT_PIPELINE_VERSION,
            "features":         features,
            "inference":        inference_out,
            "pm_status":        pm_out.get("status", "unknown"),
            "pm_out":           pm_out,
            "monitoring":       monitoring_out,
            "pipeline_ok":      pipeline_ok,
            "error":            error,
        }

        try:
            result    = self._col.insert_one(doc)
            record_id = str(result.inserted_id)
            logger.info(
                f"[RUL_predictions] Saved → run={run_id}  "
                f"bearing={bearing_name}  burst={burst_idx}  "
                f"pipeline={doc['pipeline_version']}  "
                f"status={pm_out.get('status', '?')}  id={record_id}"
            )
            return record_id
        except PyMongoError as exc:
            logger.error(f"[ServingHistory] Failed to save record: {exc}")
            raise

    def save_error_record(
        self,
        run_id:           str,
        bearing_name:     str,
        burst_idx:        int,
        error:            str,
        pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    ) -> str:
        """Save a minimal error record when the pipeline fails mid-burst."""
        return self.save_pipeline_output(
            run_id           = run_id,
            bearing_name     = bearing_name,
            burst_idx        = burst_idx,
            model_version    = "unknown",
            features         = {},
            inference_out    = {},
            pm_out           = {"status": "error"},
            monitoring_out   = {},
            pipeline_ok      = False,
            error            = error,
            pipeline_version = pipeline_version,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_serving_metadata(
        self,
        run_id:       Optional[str] = None,
        bearing_name: Optional[str] = None,
        limit:        int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve prediction records for the Audit Service (step 10)."""
        query: Dict[str, Any] = {}
        if run_id:
            query["run_id"] = run_id
        if bearing_name:
            query["bearing_name"] = bearing_name

        cursor = (
            self._col
            .find(query)
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        records = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            records.append(doc)
        logger.info(
            f"[RUL_predictions] Queried {len(records)} record(s)"
            + (f" run={run_id}" if run_id else "")
            + (f" bearing={bearing_name}" if bearing_name else "")
        )
        return records

    def get_latest_for_bearing(
        self,
        bearing_name: str,
        n: int = 1,
    ) -> List[Dict[str, Any]]:
        """Return the n most-recent pipeline outputs for a given bearing."""
        cursor = (
            self._col
            .find({"bearing_name": bearing_name})
            .sort("timestamp", DESCENDING)
            .limit(n)
        )
        docs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)
        return docs

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """High-level summary of a pipeline run."""
        total     = self._col.count_documents({"run_id": run_id})
        ok_count  = self._col.count_documents({"run_id": run_id, "pipeline_ok": True})
        alerts    = self._col.count_documents({"run_id": run_id, "pm_status": "critical"})
        warnings  = self._col.count_documents({"run_id": run_id, "pm_status": "warning"})
        pipeline_versions = self._col.distinct("pipeline_version", {"run_id": run_id})
        return {
            "run_id":            run_id,
            "total_bursts":      total,
            "ok_count":          ok_count,
            "error_count":       total - ok_count,
            "critical_count":    alerts,
            "warning_count":     warnings,
            "pipeline_versions": pipeline_versions,
        }

    def get_drift_flags(
        self,
        bearing_name: str,
        limit:        int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent drift-flagged records for a bearing."""
        cursor = (
            self._col
            .find({
                "bearing_name": bearing_name,
                "monitoring.drift_detected": True,
            })
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        docs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)
        return docs


# ─────────────────────────────────────────────────────────────────────────────
# ServingTelemetry — serving_history collection
# Operational metadata per burst: latency, throughput, resource usage,
# model version, pipeline version, correction/feedback metadata
# ─────────────────────────────────────────────────────────────────────────────

class ServingTelemetry:
    """
    Records operational telemetry for every processed burst to serving_history.

    One document per (run_id, bearing_name, burst_idx) triple.

    Document schema
    ───────────────
    {
      run_id              : str
      bearing_name        : str
      burst_idx           : int
      timestamp           : datetime
      model_version       : str        — model_id used for this burst
      pipeline_version    : str        — active workflow version (e.g. "V1")
      latency_ms          : float
      pipeline_ok         : bool
      pm_status           : str
      rul_s               : float
      rul_min             : float
      drift_detected      : bool
      anomaly_flag        : bool
      correction          : dict
      cpu_percent         : float|None
      memory_mb           : float|None
      bursts_this_session : int
    }
    """

    def __init__(self, mongo_uri: str, db_name: str = "phm_mlops"):
        self._uri = mongo_uri
        self._db_name = db_name
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._col = self._client[db_name][COL_SERVING_HISTORY]
        self._ensure_indexes()
        logger.info(
            f"ServingTelemetry connected → {db_name}.{COL_SERVING_HISTORY}"
        )

    def _ensure_indexes(self):
        try:
            self._col.create_index("run_id", name="idx_run_id")
            self._col.create_index("bearing_name", name="idx_bearing_name")
            self._col.create_index(
                [("timestamp", DESCENDING)], name="idx_timestamp_desc"
            )
            self._col.create_index("model_version", name="idx_model_version")
        except Exception as e:
            logger.warning(f"[ServingTelemetry] Index creation warning: {e}")

    def record(
            self,
            run_id: str,
            bearing_name: str,
            burst_idx: int,
            model_version: str,
            latency_ms: float,
            pipeline_ok: bool,
            pm_status: str,
            rul_s: Optional[float],
            rul_min: Optional[float],
            drift_detected: bool,
            anomaly_flag: bool,
            bursts_this_session: int,
            correction: Optional[Dict[str, Any]] = None,
            cpu_percent: Optional[float] = None,
            memory_mb: Optional[float] = None,
            # ── NEW thesis-instrumentation fields (all optional / back-compat) ──
            pipeline_ms: Optional[float] = None,
            ingestion_lag_ms: Optional[float] = None,
            e2e_ms: Optional[float] = None,
            stage_timings_ms: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Write one telemetry record to serving_history.

        Call this after every processed burst in run_serving.py.
        Failures are also recorded (pipeline_ok=False) so nothing is invisible.

        Backwards-compat: if pipeline_ms is not provided, it defaults to
        latency_ms (preserving old callers).
        """
        if pipeline_ms is None:
            pipeline_ms = latency_ms

        doc = {
            "run_id": run_id,
            "bearing_name": bearing_name,
            "burst_idx": burst_idx,
            "timestamp": datetime.now(timezone.utc),
            "model_version": model_version,

            # Latency: legacy + new
            "latency_ms": round(float(latency_ms), 3),  # back-compat alias
            "pipeline_ms": round(float(pipeline_ms), 3),
            "ingestion_lag_ms": (round(float(ingestion_lag_ms), 3)
                                 if ingestion_lag_ms is not None else None),
            "e2e_ms": (round(float(e2e_ms), 3)
                       if e2e_ms is not None else None),
            "stage_timings_ms": {k: round(float(v), 3)
                                 for k, v in (stage_timings_ms or {}).items()},

            "pipeline_ok": pipeline_ok,
            "pm_status": pm_status,
            "rul_s": rul_s,
            "rul_min": rul_min,
            "drift_detected": drift_detected,
            "anomaly_flag": anomaly_flag,
            "bursts_this_session": bursts_this_session,
            "correction": correction or {},
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
        }
        try:
            self._col.insert_one(doc)
        except PyMongoError as exc:
            logger.warning(f"[ServingTelemetry] Failed to write telemetry: {exc}")

    def get_session_stats(self, run_id: str) -> Dict[str, Any]:
        """Return aggregated telemetry for a run — used by the Dashboard."""
        docs = list(self._col.find({"run_id": run_id}, {"_id": 0}))
        if not docs:
            return {}

        pipeline_lats = [d.get("pipeline_ms") or d.get("latency_ms")
                         for d in docs
                         if (d.get("pipeline_ms") or d.get("latency_ms")) is not None]
        e2e_lats = [d["e2e_ms"] for d in docs if d.get("e2e_ms") is not None]
        ingest_lags = [d["ingestion_lag_ms"] for d in docs if d.get("ingestion_lag_ms") is not None]

        def _avg(xs):  return round(sum(xs) / len(xs), 3) if xs else None

        def _maxn(xs): return round(max(xs), 3) if xs else None

        def _minn(xs): return round(min(xs), 3) if xs else None

        return {
            "run_id": run_id,
            "total_bursts": len(docs),

            # Pipeline (old API name kept for back-compat)
            "avg_latency_ms": _avg(pipeline_lats),
            "max_latency_ms": _maxn(pipeline_lats),
            "min_latency_ms": _minn(pipeline_lats),

            # New thesis breakdown
            "avg_pipeline_ms": _avg(pipeline_lats),
            "avg_e2e_ms": _avg(e2e_lats),
            "max_e2e_ms": _maxn(e2e_lats),
            "avg_ingestion_lag_ms": _avg(ingest_lags),
            "max_ingestion_lag_ms": _maxn(ingest_lags),

            "ok_count": sum(1 for d in docs if d.get("pipeline_ok")),
            "error_count": sum(1 for d in docs if not d.get("pipeline_ok")),
            "critical_count": sum(1 for d in docs if d.get("pm_status") == "critical"),
            "warning_count": sum(1 for d in docs if d.get("pm_status") == "warning"),
            "drift_count": sum(1 for d in docs if d.get("drift_detected")),
            "anomaly_count": sum(1 for d in docs if d.get("anomaly_flag")),
            "model_versions": list({d["model_version"] for d in docs if d.get("model_version")}),
        }
