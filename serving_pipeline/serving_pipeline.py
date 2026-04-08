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
    step 7  → Feature Store    (versioned features)
    step 9  → Serving History  (full pipeline record)
    step 8  → Dashboard        (via monitoring output — RUL metrics)
             Export Service    (stubbed — structural connection only)

Design notes
────────────
- The pipeline is callable from the Workflow Orchestrator and from the FastAPI
  endpoint independently.
- It is stateful per-bearing (FE buffer, monitoring baseline are maintained
  across bursts).  Call reset_bearing() between bearings.
- On any stage failure the pipeline logs the error, writes an error record to
  Serving History, and returns a safe error response — it never raises to the
  caller.

Usage
─────
    from serving_pipeline.serving_pipeline import ServingPipeline

    pipeline = ServingPipeline(config={
        "mongo_uri":   "mongodb://localhost:27017",
        "db_name":     "phm_mlops",
        "window_size": 40,
    })

    for burst in ingestor.stream_bursts(source_folder):
        result = pipeline.run_burst(
            run_id       = "serve_20260407_abc123",
            bearing_name = "Bearing1_5",
            burst_idx    = burst["burst_idx"],
            h_signal     = burst["h_signal"],
            v_signal     = burst["v_signal"],
        )
        if result["ready"] and result["pm"]["alert"]:
            print(f"ALERT: {result['pm']['recommended_action']}")
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from serving_pipeline.feature_engineering import ServingFeatureEngineer
from serving_pipeline.inference import ServingInference
from serving_pipeline.monitoring import MLOpsMonitor
from serving_pipeline.predictive_maintenance import PredictiveMaintenance

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Export Service stub
# ─────────────────────────────────────────────────────────────────────────────

def _stub_export_service(pipeline_output: Dict[str, Any]) -> None:
    """
    Stub for the Export Service connection (diagram arrow ServPipeline → ExportSvc).
    Structural connection is present; no functional implementation per §2 instructions.
    """
    logger.debug("[ExportSvc] stub — export not implemented.")


# ─────────────────────────────────────────────────────────────────────────────
# ServingPipeline
# ─────────────────────────────────────────────────────────────────────────────

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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self._mongo_uri  = cfg.get("mongo_uri", "mongodb://localhost:27017")
        self._db_name    = cfg.get("db_name", "phm_mlops")
        self._enable_sh  = cfg.get("enable_serving_history", True)

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

        # ── Serving History ───────────────────────────────────────────────────
        self._sh = None
        if self._enable_sh:
            try:
                from utils.serving_history import ServingHistory
                self._sh = ServingHistory(
                    mongo_uri=self._mongo_uri,
                    db_name=self._db_name,
                )
                self._sh.ensure_indexes()
            except Exception as exc:
                logger.warning(
                    f"[Pipeline] Could not connect to Serving History: {exc}. "
                    "Results will not be persisted."
                )

        logger.info("[Pipeline] ServingPipeline initialised — all 4 stages ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_burst(
        self,
        run_id:       str,
        bearing_name: str,
        burst_idx:    int,
        h_signal:     np.ndarray,
        v_signal:     np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run one burst through the full 4-stage pipeline.

        Returns a structured dict with outputs from all stages plus
        the Serving History record_id.

        The return value is safe to use even on failure — check result["ok"].
        """
        try:
            return self._run_stages(run_id, bearing_name, burst_idx, h_signal, v_signal)
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
                "ok":          False,
                "ready":       False,
                "run_id":      run_id,
                "bearing":     bearing_name,
                "burst_idx":   burst_idx,
                "error":       error_str,
                "record_id":   record_id,
                "fe":          None,
                "inference":   None,
                "pm":          None,
                "monitoring":  None,
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
        Stream an entire bearing through the pipeline.

        Convenience wrapper for the Workflow Orchestrator integration.
        Resets the FE buffer and monitoring baseline between runs.
        """
        from scripts.data_ingestor import DataIngestorPHM

        self.reset_bearing()
        results: List[Dict[str, Any]] = []
        ingestor = DataIngestorPHM(config={
            "input_location":  source_folder,
            "output_location": source_folder,
        })

        for burst in ingestor.stream_bursts(
            source_folder,
            burst_period=burst_period,
            realtime=realtime,
        ):
            if max_bursts is not None and burst["burst_idx"] >= max_bursts:
                break

            result = self.run_burst(
                run_id       = run_id,
                bearing_name = bearing_name,
                burst_idx    = burst["burst_idx"],
                h_signal     = burst["h_signal"],
                v_signal     = burst["v_signal"],
            )
            results.append(result)

        logger.info(
            f"[Pipeline] run_bearing complete — bearing={bearing_name}  "
            f"bursts={len(results)}"
        )
        return results

    def reset_bearing(self) -> None:
        """Reset per-bearing state (FE window + monitoring baseline)."""
        self._fe.reset()
        self._monitor.reset_baseline()
        logger.info("[Pipeline] Bearing state reset (FE buffer + monitoring baseline).")

    def reload_model(self) -> None:
        """Force reload the deployed model from the ModelRegistry."""
        self._inference.reload_model()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_stages(
        self,
        run_id:       str,
        bearing_name: str,
        burst_idx:    int,
        h_signal:     np.ndarray,
        v_signal:     np.ndarray,
    ) -> Dict[str, Any]:

        # ── Stage 1: Feature Engineering ─────────────────────────────────────
        fe_out = self._fe.process_burst(burst_idx, h_signal, v_signal)

        if not fe_out["ready"]:
            return {
                "ok":         True,
                "ready":      False,
                "run_id":     run_id,
                "bearing":    bearing_name,
                "burst_idx":  burst_idx,
                "fe":         fe_out,
                "inference":  None,
                "pm":         None,
                "monitoring": None,
                "record_id":  None,
            }

        # ── Stage 2: Inference ────────────────────────────────────────────────
        infer_out = self._inference.run(fe_out)

        # ── Stage 3: Predictive Maintenance ───────────────────────────────────
        pm_out = self._pm.run(infer_out)

        # ── Stage 4: Monitoring ───────────────────────────────────────────────
        mon_out = self._monitor.run(fe_out)

        # ── Step 9: Write to Serving History ─────────────────────────────────
        record_id = None
        if self._sh:
            features_dict = {
                name: float(val)
                for name, val in zip(
                    fe_out["feature_names"],
                    fe_out["feature_vector"].tolist(),
                )
            }
            # Attach quality labels to the features dict
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

        # ── Export Service stub (structural connection) ────────────────────────
        _stub_export_service({
            "run_id": run_id, "bearing": bearing_name, "burst_idx": burst_idx,
            "pm": pm_out, "monitoring": mon_out,
        })

        return {
            "ok":         True,
            "ready":      True,
            "run_id":     run_id,
            "bearing":    bearing_name,
            "burst_idx":  burst_idx,
            "fe":         {
                "quality_labels": fe_out.get("quality_labels"),
                "base_features":  fe_out.get("base_features"),
            },
            "inference":  infer_out,
            "pm":         pm_out,
            "monitoring": mon_out,
            "record_id":  record_id,
        }