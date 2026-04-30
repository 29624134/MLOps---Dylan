"""
utils/audit_service.py
═══════════════════════════════════════════════════════════════════════════════
Audit Service  (Diagram: ServHistory → AuditSvc → External Data Destination)

Responsibilities
────────────────
The Audit Service sits between Serving History and the External Data
Destination.  It:

1. Reads serving history records from MongoDB (via ServingHistory)
2. Validates / enriches them (adds audit metadata, severity classification)
3. Forwards them to the Export Service which writes to the external destination

This makes the audit trail fully observable outside the MLOps system — the
External destination (CSV / JSON on disk) can be picked up by any downstream
consumer (BI tool, maintenance system, SCADA historian, etc.).

Two modes of operation
──────────────────────
A. Push (real-time): called from serving_pipeline.py after each burst so
   every pipeline output is immediately audited and exported.

B. Pull (batch): called periodically (e.g. by a scheduler or the dashboard)
   to flush all un-audited records from Serving History to the external
   destination.

Usage
─────
    from utils.audit_service import AuditService

    auditor = AuditService(config={
        "mongo_uri":    "mongodb://localhost:27017",
        "db_name":      "phm_mlops",
        "output_dir":   "export_output",
        "enable_json":  False,
    })

    # Push mode — called per burst from serving_pipeline.py
    auditor.audit_record(serving_history_record_dict)

    # Pull / batch mode — flush all records for a bearing
    n_exported = auditor.flush_bearing(bearing_name="Bearing1_5", limit=500)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditService:
    """
    Audit Service — bridges Serving History and the External Data Destination.

    Parameters
    ──────────
    config : dict
        mongo_uri   : str   MongoDB connection string (default "mongodb://localhost:27017")
        db_name     : str   MongoDB database name    (default "phm_mlops")
        output_dir  : str   export output directory  (default "export_output")
        enable_json : bool  also write JSON snapshots (default False)
        enable_rul_csv   : bool  (default True)
        enable_audit_csv : bool  (default True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self._mongo_uri  = cfg.get("mongo_uri", "mongodb://localhost:27017")
        self._db_name    = cfg.get("db_name",   "phm_mlops")

        # Lazy-import to avoid circular dependencies
        from utils.export_service import ExportService
        self._exporter = ExportService({
            "output_dir":       cfg.get("output_dir",       "export_output"),
            "enable_rul_csv":   cfg.get("enable_rul_csv",   True),
            "enable_audit_csv": cfg.get("enable_audit_csv", True),
            "enable_json":      cfg.get("enable_json",      False),
        })

        logger.info(
            f"[AuditService] Initialised — "
            f"mongo={self._db_name}  export={self._exporter.get_export_paths()['output_dir']}"
        )

    # ── Push mode ─────────────────────────────────────────────────────────────

    def audit_record(self, record: Dict[str, Any]) -> bool:
        """
        Audit and export a single Serving History record immediately.

        Called from serving_pipeline.py right after each burst is persisted
        to Serving History (real-time push path).

        Parameters
        ──────────
        record : dict — a serving history document (as returned by
                        ServingHistory.save_pipeline_output or
                        ServingHistory.get_serving_metadata)

        Returns True on success, False on failure (never raises).
        """
        enriched = self._enrich(record)
        ok = self._exporter.export_audit_record(enriched)
        if ok:
            logger.debug(
                f"[AuditService] Audited burst — "
                f"bearing={record.get('bearing_name')}  "
                f"burst={record.get('burst_idx')}"
            )
        return ok

    # ── Pull / batch mode ─────────────────────────────────────────────────────

    def flush_bearing(
        self,
        bearing_name: str,
        limit: int = 500,
    ) -> int:
        """
        Pull the most recent `limit` records for `bearing_name` from Serving
        History and export each one via the Export Service.

        Returns the number of records successfully exported.
        """
        try:
            from utils.serving_history import ServingHistory
            sh = ServingHistory(
                mongo_uri=self._mongo_uri,
                db_name=self._db_name,
            )
            records = sh.get_serving_metadata(bearing_name=bearing_name, limit=limit)
        except Exception as exc:
            logger.error(f"[AuditService] Could not read Serving History: {exc}")
            return 0

        exported = 0
        for record in records:
            if self._exporter.export_audit_record(self._enrich(record)):
                exported += 1

        logger.info(
            f"[AuditService] Flushed {exported}/{len(records)} records "
            f"for bearing={bearing_name}."
        )
        return exported

    def flush_run(self, run_id: str, limit: int = 5000) -> int:
        """
        Pull all records for a specific run_id and export them.
        Useful for post-run batch auditing from the dashboard.
        """
        try:
            from utils.serving_history import ServingHistory
            sh = ServingHistory(
                mongo_uri=self._mongo_uri,
                db_name=self._db_name,
            )
            records = sh.get_serving_metadata(run_id=run_id, limit=limit)
        except Exception as exc:
            logger.error(f"[AuditService] Could not read Serving History: {exc}")
            return 0

        exported = 0
        for record in records:
            if self._exporter.export_audit_record(self._enrich(record)):
                exported += 1

        logger.info(
            f"[AuditService] Flushed {exported}/{len(records)} records "
            f"for run_id={run_id}."
        )
        return exported

    def get_export_paths(self) -> Dict[str, str]:
        """Return the export destination paths (for API / dashboard)."""
        return self._exporter.get_export_paths()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _enrich(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add audit-level metadata to a Serving History record before export.
        Adds: audited_at timestamp and severity classification.
        """
        enriched = dict(record)

        enriched["audited_at"] = datetime.now(timezone.utc).isoformat()

        # Severity: derived from PM status for downstream filtering
        pm     = (record.get("pm") or {})
        status = pm.get("status", "unknown")
        enriched["severity"] = {
            "critical": "HIGH",
            "warning":  "MEDIUM",
            "healthy":  "LOW",
        }.get(status, "UNKNOWN")

        return enriched