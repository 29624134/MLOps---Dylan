"""
export_service/export_service.py
═══════════════════════════════════════════════════════════════════════════════
Export Service  (Diagram: ServPipeline → ExportSvc → External Data Destination
                          AuditSvc    → ExportSvc → External Data Destination)

Responsibilities
────────────────
Receives pipeline output (RUL predictions, PM status, monitoring stats) from
the Serving Pipeline and confirmed audit metadata from the Audit Service, then
writes them to an external "Data Destination" in one or more configurable
formats:

    1. Rolling CSV   — appends one row per burst to  export_output/rul_export.csv
    2. Audit CSV     — appends one row per burst to  export_output/audit_export.csv
    3. JSON snapshots — writes per-burst JSON to     export_output/snapshots/<bearing>/<burst>.json

Both the CSV and JSON destinations are local disk by default (suitable for a
shared network drive, S3 mount, or any path the OS can write to).  The output
directory is configurable.

Usage
─────
    # Direct (called from serving_pipeline.py instead of the old stub):
    from export_service.export_service import ExportService

    exporter = ExportService(config={
        "output_dir":        "export_output",
        "enable_rul_csv":    True,
        "enable_audit_csv":  True,
        "enable_json":       True,
    })

    # From serving pipeline (step 8 — RUL export):
    exporter.export_pipeline_output(pipeline_result)

    # From audit service (AuditSvc → External):
    exporter.export_audit_record(serving_history_record)

Integration points
──────────────────
serving_pipeline/serving_pipeline.py  — replaces _stub_export_service()
utils/audit_service.py                — new file, calls export_audit_record()

Thread safety
─────────────
All file writes use exclusive file-level locking via a threading.Lock so the
export service is safe to call from concurrent burst threads.
"""

import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / column definitions
# ─────────────────────────────────────────────────────────────────────────────

_RUL_CSV_FIELDNAMES = [
    "exported_at", "run_id", "bearing", "burst_idx",
    "rul_s", "rul_min", "pm_status", "alert",
    "data_quality", "drift_detected", "anomaly_flag",
    "model_version",
]

_AUDIT_CSV_FIELDNAMES = [
    "exported_at", "run_id", "bearing", "burst_idx",
    "timestamp", "pipeline_ok", "rul_s", "rul_min",
    "pm_status", "alert", "data_quality",
    "drift_detected", "anomaly_flag", "error",
]


class ExportService:
    """
    Functional Export Service — writes RUL and audit data to an external
    data destination (CSV files + optional JSON snapshots).

    Parameters
    ──────────
    config : dict
        output_dir       : str   directory for all export files  (default "export_output")
        enable_rul_csv   : bool  write rolling RUL CSV           (default True)
        enable_audit_csv : bool  write rolling audit CSV         (default True)
        enable_json      : bool  write per-burst JSON snapshots  (default False)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self._output_dir      = cfg.get("output_dir", "export_output")
        self._enable_rul_csv  = bool(cfg.get("enable_rul_csv",  True))
        self._enable_audit_csv= bool(cfg.get("enable_audit_csv", True))
        self._enable_json     = bool(cfg.get("enable_json", False))

        self._rul_csv_path   = os.path.join(self._output_dir, "rul_export.csv")
        self._audit_csv_path = os.path.join(self._output_dir, "audit_export.csv")
        self._snap_dir       = os.path.join(self._output_dir, "snapshots")

        self._lock = threading.Lock()
        self._ensure_dirs()

        logger.info(
            f"[ExportService] Initialised — output_dir={self._output_dir}  "
            f"rul_csv={self._enable_rul_csv}  "
            f"audit_csv={self._enable_audit_csv}  "
            f"json={self._enable_json}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def export_pipeline_output(self, pipeline_result: Dict[str, Any]) -> bool:
        """
        Step 8 export: receive the full pipeline result dict from ServingPipeline
        and write it to the external data destination.

        Parameters
        ──────────
        pipeline_result : dict returned by ServingPipeline.run_burst()

        Returns True on success, False on failure (never raises).
        """
        if not pipeline_result.get("ok") or not pipeline_result.get("ready"):
            return True  # nothing to export for warm-up / error bursts

        try:
            run_id      = pipeline_result.get("run_id", "unknown")
            bearing     = pipeline_result.get("bearing", "unknown")
            burst_idx   = pipeline_result.get("burst_idx", -1)
            infer_out   = pipeline_result.get("inference") or {}
            pm_out      = pipeline_result.get("pm")        or {}
            mon_out     = pipeline_result.get("monitoring") or {}

            exported_at = datetime.now(timezone.utc).isoformat()

            row = {
                "exported_at":    exported_at,
                "run_id":         run_id,
                "bearing":        bearing,
                "burst_idx":      burst_idx,
                "rul_s":          infer_out.get("rul_s"),
                "rul_min":        infer_out.get("rul_min"),
                "pm_status":      pm_out.get("status", "unknown"),
                "alert":          pm_out.get("alert", False),
                "data_quality":   infer_out.get("data_quality", "clean"),
                "drift_detected": mon_out.get("drift_detected", False),
                "anomaly_flag":   mon_out.get("anomaly_flag", False),
                "model_version":  infer_out.get("model_version", "unknown"),
            }

            with self._lock:
                if self._enable_rul_csv:
                    self._append_csv(self._rul_csv_path, row, _RUL_CSV_FIELDNAMES)

                if self._enable_json:
                    self._write_json_snapshot(
                        bearing, burst_idx,
                        {"type": "rul", "exported_at": exported_at, **pipeline_result}
                    )

            logger.debug(
                f"[ExportService] Exported burst — "
                f"bearing={bearing}  burst={burst_idx}  "
                f"rul_min={row['rul_min']}  status={row['pm_status']}"
            )
            return True

        except Exception as exc:
            logger.error(f"[ExportService] export_pipeline_output failed: {exc}", exc_info=True)
            return False

    def export_audit_record(self, record: Dict[str, Any]) -> bool:
        """
        AuditSvc → External: write a Serving History record to the audit CSV.

        Parameters
        ──────────
        record : dict — one document from the serving_history MongoDB collection,
                        as returned by ServingHistory.get_serving_metadata()

        Returns True on success, False on failure (never raises).
        """
        try:
            infer      = record.get("inference", {}) or {}
            pm         = record.get("pm", {}) or {}
            mon        = record.get("monitoring", {}) or {}

            row = {
                "exported_at":    datetime.now(timezone.utc).isoformat(),
                "run_id":         record.get("run_id", "unknown"),
                "bearing":        record.get("bearing_name", "unknown"),
                "burst_idx":      record.get("burst_idx", -1),
                "timestamp":      record.get("timestamp", ""),
                "pipeline_ok":    record.get("pipeline_ok", False),
                "rul_s":          infer.get("rul_s"),
                "rul_min":        infer.get("rul_min"),
                "pm_status":      pm.get("status", "unknown"),
                "alert":          pm.get("alert", False),
                "data_quality":   infer.get("data_quality", "clean"),
                "drift_detected": mon.get("drift_detected", False),
                "anomaly_flag":   mon.get("anomaly_flag", False),
                "error":          record.get("error", ""),
            }

            with self._lock:
                if self._enable_audit_csv:
                    self._append_csv(self._audit_csv_path, row, _AUDIT_CSV_FIELDNAMES)

                if self._enable_json:
                    self._write_json_snapshot(
                        row["bearing"], row["burst_idx"],
                        {"type": "audit", **row}
                    )

            logger.debug(
                f"[ExportService] Audit exported — "
                f"bearing={row['bearing']}  burst={row['burst_idx']}"
            )
            return True

        except Exception as exc:
            logger.error(f"[ExportService] export_audit_record failed: {exc}", exc_info=True)
            return False

    def get_export_paths(self) -> Dict[str, str]:
        """Return the configured output paths (for API / dashboard display)."""
        return {
            "output_dir":       self._output_dir,
            "rul_csv":          self._rul_csv_path   if self._enable_rul_csv   else None,
            "audit_csv":        self._audit_csv_path if self._enable_audit_csv else None,
            "json_snapshots":   self._snap_dir        if self._enable_json      else None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        if self._enable_json:
            os.makedirs(self._snap_dir, exist_ok=True)

    def _append_csv(
        self,
        path: str,
        row: Dict[str, Any],
        fieldnames: list,
    ) -> None:
        """Append a single row to a CSV, writing the header if the file is new."""
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _write_json_snapshot(
        self,
        bearing: str,
        burst_idx: int,
        data: Dict[str, Any],
    ) -> None:
        """Write a per-burst JSON file to the snapshots directory."""
        bearing_dir = os.path.join(self._snap_dir, bearing)
        os.makedirs(bearing_dir, exist_ok=True)
        path = os.path.join(bearing_dir, f"burst_{burst_idx:06d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (shared across serving_pipeline + audit_service)
# ─────────────────────────────────────────────────────────────────────────────

_default_exporter: Optional[ExportService] = None


def get_exporter(config: Optional[Dict[str, Any]] = None) -> ExportService:
    """
    Return (or lazily create) the module-level ExportService singleton.
    Pass config only on first call or when reconfiguring.
    """
    global _default_exporter
    if _default_exporter is None or config is not None:
        _default_exporter = ExportService(config)
    return _default_exporter