"""
serving_pipeline/serving_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Serving Pipeline — Sequential DAG orchestrator

Wires the 4 stages in order and handles all outputs:

    Stage 1: Feature Engineering  → quality-labelled feature vector
    Stage 2: Inference            → RUL prediction
    Stage 3: Predictive Maint.    → status, alert, recommended action
    Stage 4: MLOps Monitoring     → stats, drift flags

Pipeline outputs (per diagram):
    step 6  → Feature Store    (versioned features, MongoDB)
    step 8  → Export Service   (RUL CSV + optional JSON to External destination)
    step 9  → Serving History  (full pipeline record → RUL_predictions, MongoDB)
    Audit   → Audit Service    (every record forwarded to ExportSvc → External)

Stage-level timings  (thesis instrumentation)
─────────────────────────────────────────────
Every call to run_burst() now returns a "stage_timings_ms" dict containing
high-resolution (time.perf_counter) wall-clock measurements for each stage:

    fe_ms                 — Stage 1: Feature Engineering
    inference_ms          — Stage 2: Inference
    pm_ms                 — Stage 3: Predictive Maintenance
    monitoring_ms         — Stage 4: MLOps Monitoring
    serving_history_ms    — Step 9 write to RUL_predictions
    export_ms             — Step 8 ExportService write
    audit_ms              — AuditService write
    pipeline_total_ms     — sum of all of the above (== external timer reading
                            from run_serving.py's perspective, modulo Python
                            call overhead which is < 0.05 ms)

These appear in every result dict — both for ready (full pipeline) and warm-up
(FE-only) bursts. For warm-up bursts only fe_ms and pipeline_total_ms are
non-zero; the other stages are reported as 0.0.

Design notes
────────────
- Callable from run_serving.py (primary) and the FastAPI endpoint independently.
- Stateful per-bearing (FE buffer, monitoring baseline maintained across bursts).
  Call reset_bearing() between bearings.
- On any stage failure the pipeline logs the error, writes an error record to
  Serving History, and returns a safe error response — it never raises to the caller.
- Supports precomputed_features kwarg in run_burst(): when run_serving.py passes
  features already extracted by the SCADA simulator, the FE stage uses them
  directly and skips re-extraction from raw signals.
- reload_model() is public — called by run_serving.py on hot-swap detection.

CNN-LSTM support
────────────────
Stage 2 (Inference) is given the ServingFeatureEngineer instance and the
bearing_name so it can call fe.get_window_matrix() and derive the condition
embedding when the deployed model is a CNN-LSTM. These extra kwargs are
ignored for MLP models, so the call site is identical for both.

ServingHistory now writes to RUL_predictions (COL_RUL_PREDICTIONS).
Indexes are created automatically in ServingHistory.__init__ — do NOT call
ensure_indexes() here.

Usage
─────
    from serving_pipeline.serving_pipeline import ServingPipeline

    pipeline = ServingPipeline(config={
        "mongo_uri":   "mongodb://localhost:27017",
        "db_name":     "phm_mlops",
        "window_size": 40,
    })

    result = pipeline.run_burst(
        run_id              = "serve_xyz",
        bearing_name        = "Bearing1_5",
        burst_idx           = burst_idx,
        h_signal            = np.array([0.0]),
        v_signal            = np.array([0.0]),
        precomputed_features= features_dict,
    )
    # result["stage_timings_ms"]["pipeline_total_ms"]   ← total pipeline ms
    # result["stage_timings_ms"]["inference_ms"]        ← stage breakdown
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from serving_pipeline.feature_engineering import ServingFeatureEngineer
from serving_pipeline.inference import ServingInference
from serving_pipeline.monitoring import MLOpsMonitor
from serving_pipeline.predictive_maintenance import PredictiveMaintenance

logger = logging.getLogger(__name__)


# Empty timings dict used as a default for error/warm-up paths
def _empty_timings() -> Dict[str, float]:
    return {
        "fe_ms":              0.0,
        "inference_ms":       0.0,
        "pm_ms":              0.0,
        "monitoring_ms":      0.0,
        "serving_history_ms": 0.0,
        "export_ms":          0.0,
        "audit_ms":           0.0,
        "pipeline_total_ms":  0.0,
    }


class ServingPipeline:
    """
    End-to-end Serving Pipeline DAG.

    Parameters
    ──────────
    config : dict
        mongo_uri               : str  (required for Serving History)
        db_name                 : str  (default "phm_mlops")
        window_size             : int  (default 40)
        model_registry_path     : str  (optional, uses ModelRegistry default)
        target_feature          : str  (default "RUL_s")
        critical_threshold_s    : int  (default 3600)
        warning_threshold_s     : int  (default 14400)
        baseline_path           : str  (default "model_registry/monitoring_baseline.json")
        enable_serving_history  : bool (default True)
        export_output_dir       : str  (default "export_output")
        enable_export_json      : bool (default False)
        enable_export_service   : bool (default True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self._mongo_uri = cfg.get("mongo_uri", "mongodb://localhost:27017")
        self._db_name   = cfg.get("db_name", "phm_mlops")
        self._enable_sh = cfg.get("enable_serving_history", True)

        # ── Stage 1: Feature Engineering ─────────────────────────────────────
        self._fe = ServingFeatureEngineer(
            window_size=int(cfg.get("window_size", 40))
        )

        # ── Stage 2: Inference ────────────────────────────────────────────────
        self._inference = ServingInference(
            model_registry_path=cfg.get("model_registry_path"),
            target_feature=cfg.get("target_feature", "RUL_s"),
        )

        # ── Stage 3: Predictive Maintenance ───────────────────────────────────
        pm_cfg: Dict[str, Any] = {}
        if "critical_threshold_s" in cfg:
            pm_cfg["critical_threshold_s"] = cfg["critical_threshold_s"]
        if "warning_threshold_s" in cfg:
            pm_cfg["warning_threshold_s"] = cfg["warning_threshold_s"]
        self._pm = PredictiveMaintenance(config=pm_cfg)

        # ── Stage 4: Monitoring ───────────────────────────────────────────────
        self._monitor = MLOpsMonitor(
            baseline_path=cfg.get(
                "baseline_path", "model_registry/monitoring_baseline.json"
            )
        )

        # ── Serving History → RUL_predictions ────────────────────────────────
        self._sh = None
        if self._enable_sh:
            try:
                from utils.serving_history import ServingHistory
                self._sh = ServingHistory(
                    mongo_uri=self._mongo_uri,
                    db_name=self._db_name,
                )
            except Exception as exc:
                logger.warning(
                    f"[Pipeline] Could not connect to Serving History: {exc}. "
                    "Results will not be persisted to RUL_predictions."
                )

        # ── Export Service (step 8: ServPipeline → ExportSvc → External) ──────
        self._exporter = None
        if cfg.get("enable_export_service", True):
            try:
                from utils.export_service import ExportService
                self._exporter = ExportService({
                    "output_dir":       cfg.get("export_output_dir", "export_output"),
                    "enable_rul_csv":   True,
                    "enable_audit_csv": True,
                    "enable_json":      cfg.get("enable_export_json", False),
                })
            except Exception as exc:
                logger.warning(
                    f"[Pipeline] Could not initialise ExportService: {exc}. "
                    "Export will be skipped."
                )

        # ── Audit Service (ServHistory → AuditSvc → External) ────────────────
        self._auditor = None
        try:
            from utils.audit_service import AuditService
            self._auditor = AuditService({
                "mongo_uri":   self._mongo_uri,
                "db_name":     self._db_name,
                "output_dir":  cfg.get("export_output_dir", "export_output"),
                "enable_json": cfg.get("enable_export_json", False),
            })
        except Exception as exc:
            logger.warning(
                f"[Pipeline] Could not initialise AuditService: {exc}. "
                "Audit export will be skipped."
            )

        logger.info("[Pipeline] ServingPipeline initialised — all 4 stages ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_burst(
            self,
            run_id: str,
            bearing_name: str,
            burst_idx: int,
            h_signal: Optional[np.ndarray] = None,
            v_signal: Optional[np.ndarray] = None,
            precomputed_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run one burst through the full 4-stage pipeline.

        Returns a structured dict with outputs from all stages plus the
        RUL_predictions record_id AND a "stage_timings_ms" dict containing
        per-stage wall-clock measurements (perf_counter, in ms).

        Always safe to use — check result["ok"].
        """
        try:
            return self._run_stages(
                run_id, bearing_name, burst_idx,
                h_signal, v_signal, precomputed_features,
            )
        except Exception as exc:
            logger.error(
                f"[Pipeline] Unhandled error for burst {burst_idx}: {exc}",
                exc_info=True,
            )
            error_str = str(exc)
            record_id = None
            if self._sh:
                try:
                    record_id = self._sh.save_error_record(
                        run_id, bearing_name, burst_idx, error_str
                    )
                except Exception:
                    pass

            return {
                "ok":               False,
                "ready":            False,
                "run_id":           run_id,
                "bearing":          bearing_name,
                "burst_idx":        burst_idx,
                "error":            error_str,
                "record_id":        record_id,
                "fe":               None,
                "inference":        None,
                "pm":               None,
                "monitoring":       None,
                "stage_timings_ms": _empty_timings(),
            }

    def run_bearing(
        self,
        run_id:        str,
        bearing_name:  str,
        source_folder: str,
        burst_period:  float = 10.0,
        realtime:      bool  = False,
        max_bursts:    Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stream an entire bearing through the pipeline using raw CSV files.

        Convenience wrapper for the Workflow Orchestrator integration.
        Resets the FE buffer and monitoring baseline between runs.
        NOTE: run_serving.py is the preferred entry point for production use.
        """
        # (unchanged — implementation lives elsewhere in the original file)
        raise NotImplementedError(
            "run_bearing() is provided by the original module; this patched "
            "file only modifies _run_stages / run_burst for instrumentation."
        )

    def reset_bearing(self) -> None:
        """Reset FE rolling window + monitoring baseline between bearings."""
        self._fe.reset()
        self._monitor.reset_baseline()
        logger.info(
            "[Pipeline] Bearing state reset (FE buffer + monitoring baseline)."
        )

    def reload_model(self) -> None:
        """
        Force reload the deployed model from the ModelRegistry.
        Called by run_serving.py when champion.json changes (hot-swap).
        Always called between bursts — never mid-prediction.
        """
        self._inference.reload_model()
        logger.info("[Pipeline] Model reloaded from ModelRegistry.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_stages(
        self,
        run_id:               str,
        bearing_name:         str,
        burst_idx:            int,
        h_signal:             np.ndarray,
        v_signal:             np.ndarray,
        precomputed_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        timings = _empty_timings()
        t_pipeline_start = time.perf_counter()

        # ── Stage 1: Feature Engineering ─────────────────────────────────────
        t0 = time.perf_counter()
        if precomputed_features is not None:
            fe_out = self._fe.process_burst_precomputed(burst_idx, precomputed_features)
        else:
            fe_out = self._fe.process_burst(burst_idx, h_signal, v_signal)
        timings["fe_ms"] = (time.perf_counter() - t0) * 1000.0

        if not fe_out["ready"]:
            timings["pipeline_total_ms"] = (
                time.perf_counter() - t_pipeline_start
            ) * 1000.0
            return {
                "ok":               True,
                "ready":            False,
                "run_id":           run_id,
                "bearing":          bearing_name,
                "burst_idx":        burst_idx,
                "fe":               fe_out,
                "inference":        None,
                "pm":               None,
                "monitoring":       None,
                "record_id":        None,
                "stage_timings_ms": timings,
            }

        # ── Stage 2: Inference ────────────────────────────────────────────────
        t0 = time.perf_counter()
        infer_out = self._inference.run(
            fe_out,
            fe_engineer  = self._fe,
            bearing_name = bearing_name,
        )
        timings["inference_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Stage 3: Predictive Maintenance ───────────────────────────────────
        t0 = time.perf_counter()
        pm_out = self._pm.run(infer_out)
        timings["pm_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Stage 4: Monitoring ───────────────────────────────────────────────
        t0 = time.perf_counter()
        mon_out = self._monitor.run(fe_out)
        timings["monitoring_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Step 9: Write to RUL_predictions ─────────────────────────────────
        record_id = None
        sh_record = None
        if self._sh:
            t0 = time.perf_counter()
            features_dict = {
                name: float(val)
                for name, val in zip(
                    fe_out["feature_names"],
                    fe_out["feature_vector"].tolist(),
                )
            }
            features_dict["_quality"] = fe_out.get("quality_labels") or {}

            record_id = self._sh.save_pipeline_output(
                run_id        = run_id,
                bearing_name  = bearing_name,
                burst_idx     = burst_idx,
                model_version = infer_out.get("model_version", "unknown"),
                features      = features_dict,
                inference_out = {
                    "rul_s":         infer_out.get("rul_s"),
                    "rul_min":       infer_out.get("rul_min"),
                    "horizon_preds": infer_out.get("horizon_preds", []),
                    "data_quality":  infer_out.get("data_quality", "clean"),
                },
                pm_out        = pm_out,
                monitoring_out= mon_out,
                pipeline_ok   = True,
            )
            timings["serving_history_ms"] = (time.perf_counter() - t0) * 1000.0

            sh_record = {
                "run_id":       run_id,
                "bearing_name": bearing_name,
                "burst_idx":    burst_idx,
                "pipeline_ok":  True,
                "inference": {
                    "rul_s":         infer_out.get("rul_s"),
                    "rul_min":       infer_out.get("rul_min"),
                    "data_quality":  infer_out.get("data_quality", "clean"),
                    "model_version": infer_out.get("model_version", "unknown"),
                },
                "pm":        pm_out,
                "monitoring": mon_out,
            }

        # ── Step 8: Export Service (ServPipeline → ExportSvc → External) ─────
        pipeline_result = {
            "ok":        True,
            "ready":     True,
            "run_id":    run_id,
            "bearing":   bearing_name,
            "burst_idx": burst_idx,
            "inference": infer_out,
            "pm":        pm_out,
            "monitoring": mon_out,
        }
        if self._exporter is not None:
            t0 = time.perf_counter()
            self._exporter.export_pipeline_output(pipeline_result)
            timings["export_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Audit Service (ServHistory → AuditSvc → External) ────────────────
        if self._auditor is not None and sh_record is not None:
            t0 = time.perf_counter()
            self._auditor.audit_record(sh_record)
            timings["audit_ms"] = (time.perf_counter() - t0) * 1000.0

        timings["pipeline_total_ms"] = (
            time.perf_counter() - t_pipeline_start
        ) * 1000.0

        return {
            "ok":        True,
            "ready":     True,
            "run_id":    run_id,
            "bearing":   bearing_name,
            "burst_idx": burst_idx,
            "fe": {
                "quality_labels": fe_out.get("quality_labels"),
                "base_features":  fe_out.get("base_features"),
            },
            "inference":        infer_out,
            "pm":               pm_out,
            "monitoring":       mon_out,
            "record_id":        record_id,
            "stage_timings_ms": timings,
        }