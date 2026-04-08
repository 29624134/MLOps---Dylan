"""
utils/serving_history.py
═══════════════════════════════════════════════════════════════════════════════
Serving History — MongoDB-backed store for pipeline run outputs.

Diagram connections:
    ServPipeline  ──(step 9)──►  ServHistory   [this module]
    ServHistory   ──(step 10)──► AuditSvc      [metadata query method]

Collection: phm_mlops.serving_history
Each document represents one complete pipeline run for one bearing burst,
storing features, predictions, maintenance status, monitoring flags, and
the model version that produced the result.

Usage
─────
    from utils.serving_history import ServingHistory

    sh = ServingHistory(mongo_uri="mongodb://localhost:27017", db_name="phm_mlops")

    record_id = sh.save_pipeline_output(
        run_id        = "serve_20260407_123456",
        bearing_name  = "Bearing1_5",
        burst_idx     = 42,
        model_version = "rul_model_v3",
        features      = {...},           # quality-labelled feature dict
        inference_out = {...},           # raw prediction values
        pm_out        = {...},           # RUL status, thresholds
        monitoring_out= {...},           # stats, drift flags
    )

    # Audit Service query (step 10)
    records = sh.get_serving_metadata(run_id="serve_20260407_123456")
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema constants
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "serving_history"

# Recommended indexes — created once via ensure_indexes()
_INDEXES = [
    [("run_id",       ASCENDING)],
    [("bearing_name", ASCENDING)],
    [("timestamp",    DESCENDING)],
    [("run_id",       ASCENDING), ("bearing_name", ASCENDING)],
    [("pm_status",    ASCENDING)],
]


# ─────────────────────────────────────────────────────────────────────────────
# ServingHistory
# ─────────────────────────────────────────────────────────────────────────────

class ServingHistory:
    """
    Persists the output of every Serving Pipeline run to MongoDB.

    One document per (run_id, bearing_name, burst_idx) triple.

    Document schema
    ───────────────
    {
      run_id         : str           — workflow run identifier
      bearing_name   : str           — e.g. "Bearing1_5"
      burst_idx      : int           — burst index within the bearing stream
      timestamp      : datetime      — UTC wall-clock time of pipeline run
      model_version  : str           — model_id from ModelRegistry
      features       : dict          — quality-labelled feature vector
      inference      : {             — raw model output
          rul_s          : float
          rul_min        : float
          horizon_preds  : list[float]
      }
      pm_status      : str           — "healthy" | "warning" | "critical"
      pm_out         : dict          — full predictive-maintenance output
      monitoring     : dict          — stats, drift flags, anomaly flags
      pipeline_ok    : bool          — True if all 4 stages completed cleanly
      error          : str | None    — set if pipeline_ok is False
    }
    """

    def __init__(self, mongo_uri: str, db_name: str = "phm_mlops"):
        self._uri     = mongo_uri
        self._db_name = db_name
        self._client  = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._col     = self._client[db_name][COLLECTION_NAME]
        logger.info(f"ServingHistory connected → {db_name}.{COLLECTION_NAME}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_pipeline_output(
        self,
        run_id:         str,
        bearing_name:   str,
        burst_idx:      int,
        model_version:  str,
        features:       Dict[str, Any],
        inference_out:  Dict[str, Any],
        pm_out:         Dict[str, Any],
        monitoring_out: Dict[str, Any],
        pipeline_ok:    bool = True,
        error:          Optional[str] = None,
    ) -> str:
        """
        Persist one pipeline run record.

        Returns the MongoDB inserted_id as a string.
        """
        doc = {
            "run_id":        run_id,
            "bearing_name":  bearing_name,
            "burst_idx":     burst_idx,
            "timestamp":     datetime.now(timezone.utc),
            "model_version": model_version,
            "features":      features,
            "inference":     inference_out,
            "pm_status":     pm_out.get("status", "unknown"),
            "pm_out":        pm_out,
            "monitoring":    monitoring_out,
            "pipeline_ok":   pipeline_ok,
            "error":         error,
        }

        try:
            result = self._col.insert_one(doc)
            record_id = str(result.inserted_id)
            logger.info(
                f"[ServingHistory] Saved → run={run_id}  bearing={bearing_name}"
                f"  burst={burst_idx}  status={pm_out.get('status', '?')}"
                f"  id={record_id}"
            )
            return record_id
        except PyMongoError as exc:
            logger.error(f"[ServingHistory] Failed to save record: {exc}")
            raise

    def save_error_record(
        self,
        run_id:       str,
        bearing_name: str,
        burst_idx:    int,
        error:        str,
    ) -> str:
        """
        Save a minimal error record when the pipeline fails mid-run.
        Ensures every initiated pipeline run has a traceable entry.
        """
        return self.save_pipeline_output(
            run_id        = run_id,
            bearing_name  = bearing_name,
            burst_idx     = burst_idx,
            model_version = "unknown",
            features      = {},
            inference_out = {},
            pm_out        = {"status": "error"},
            monitoring_out= {},
            pipeline_ok   = False,
            error         = error,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_serving_metadata(
        self,
        run_id:       Optional[str] = None,
        bearing_name: Optional[str] = None,
        limit:        int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve serving history records for the Audit Service (step 10).

        Filters by run_id and/or bearing_name.  Returns up to `limit` records
        sorted newest-first, with _id converted to string for serialisability.
        """
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
            f"[ServingHistory] Queried {len(records)} record(s)"
            + (f" for run_id={run_id}" if run_id else "")
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
        """
        High-level summary of a pipeline run — used by the Dashboard and
        Audit Service to quickly assess run health.
        """
        pipeline = [
            {"$match": {"run_id": run_id}},
            {"$group": {
                "_id":            "$run_id",
                "total_bursts":   {"$sum": 1},
                "ok_count":       {"$sum": {"$cond": ["$pipeline_ok", 1, 0]}},
                "error_count":    {"$sum": {"$cond": ["$pipeline_ok", 0, 1]}},
                "critical_count": {"$sum": {"$cond": [{"$eq": ["$pm_status", "critical"]}, 1, 0]}},
                "warning_count":  {"$sum": {"$cond": [{"$eq": ["$pm_status", "warning"]},  1, 0]}},
                "healthy_count":  {"$sum": {"$cond": [{"$eq": ["$pm_status", "healthy"]},  1, 0]}},
                "first_ts":       {"$min": "$timestamp"},
                "last_ts":        {"$max": "$timestamp"},
                "bearings":       {"$addToSet": "$bearing_name"},
            }},
        ]
        results = list(self._col.aggregate(pipeline))
        if not results:
            return {"run_id": run_id, "total_bursts": 0}

        summary = results[0]
        summary.pop("_id", None)
        summary["run_id"] = run_id
        return summary

    def get_drift_flags(
        self,
        bearing_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Return recent monitoring records where drift was detected.
        Useful for dashboard alerting.
        """
        cursor = (
            self._col
            .find({
                "bearing_name": bearing_name,
                "monitoring.drift_detected": True,
            })
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        return [
            {
                "burst_idx": d.get("burst_idx"),
                "timestamp": d.get("timestamp"),
                "drift_features": d.get("monitoring", {}).get("drift_features", []),
                "anomaly_flag": d.get("monitoring", {}).get("anomaly_flag", False),
            }
            for d in cursor
        ]

    # ── Admin ─────────────────────────────────────────────────────────────────

    def ensure_indexes(self) -> None:
        """Create recommended indexes. Safe to call multiple times (idempotent)."""
        for key_spec in _INDEXES:
            self._col.create_index(key_spec)
        logger.info(f"[ServingHistory] Indexes ensured on '{COLLECTION_NAME}'.")

    def count(self, run_id: Optional[str] = None) -> int:
        query = {"run_id": run_id} if run_id else {}
        return self._col.count_documents(query)

    def close(self) -> None:
        self._client.close()