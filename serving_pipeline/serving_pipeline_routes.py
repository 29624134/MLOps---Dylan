"""
api/serving_pipeline_routes.py
═══════════════════════════════════════════════════════════════════════════════
FastAPI routes for the Serving Pipeline.

Endpoints
─────────
POST  /serve/pipeline          — run the full 4-stage pipeline for one burst
POST  /serve/pipeline/bearing  — run pipeline over an entire bearing (background)
GET   /serving-history         — query Serving History (step 10 / Audit)
GET   /serving-history/run/{run_id}/summary  — aggregated run summary
GET   /serving-history/bearing/{name}/latest — latest n records for a bearing

Mount into your existing FastAPI app:
    from api.serving_pipeline_routes import router as serving_router
    app.include_router(serving_router)
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Serving Pipeline"])

# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline instance (lazy-initialised on first request)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_instance = None
_pipeline_config: Dict[str, Any] = {
    "mongo_uri":            "mongodb://localhost:27017",
    "db_name":              "phm_mlops",
    "window_size":          40,
    "critical_threshold_s": 3600,
    "warning_threshold_s":  14400,
    "enable_serving_history": True,
}


def _get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        from serving_pipeline.serving_pipeline import ServingPipeline
        _pipeline_instance = ServingPipeline(config=_pipeline_config)
        logger.info("[API] ServingPipeline initialised.")
    return _pipeline_instance


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class BurstRequest(BaseModel):
    """Single-burst inference request."""
    bearing_name: str             = Field(..., example="Bearing1_5")
    burst_idx:    int             = Field(..., ge=0)
    h_signal:     List[float]     = Field(..., min_items=1, description="Horizontal vibration samples")
    v_signal:     List[float]     = Field(..., min_items=1, description="Vertical vibration samples")
    run_id:       Optional[str]   = Field(None, description="Workflow run ID (auto-generated if omitted)")


class BearingPipelineRequest(BaseModel):
    """Start pipeline over an entire bearing (background task)."""
    bearing_name:  str            = Field(..., example="Bearing1_5")
    source_folder: str            = Field(..., description="Path to folder with acc_*.csv files")
    burst_period:  float          = Field(10.0, gt=0)
    realtime:      bool           = Field(False)
    max_bursts:    Optional[int]  = Field(None, gt=0)


class PMResult(BaseModel):
    status:             str
    rul_s:              Optional[float]
    rul_min:            Optional[float]
    rul_hours:          Optional[float]
    alert:              bool
    recommended_action: str
    low_confidence:     bool


class MonitorResult(BaseModel):
    drift_detected: bool
    drift_features: List[str]
    anomaly_flag:   bool
    baseline_ready: bool


class BurstResponse(BaseModel):
    ok:            bool
    ready:         bool
    run_id:        str
    bearing:       str
    burst_idx:     int
    pm:            Optional[PMResult]
    monitoring:    Optional[MonitorResult]
    record_id:     Optional[str]
    error:         Optional[str]
    timestamp:     str


# ─────────────────────────────────────────────────────────────────────────────
# POST /serve/pipeline  — single burst
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/serve/pipeline", response_model=BurstResponse)
async def run_serving_pipeline_burst(request: BurstRequest):
    """
    Run the 4-stage Serving Pipeline for a single sensor burst.

    - Stage 1: Feature Engineering (quality labels)
    - Stage 2: Inference (RUL prediction)
    - Stage 3: Predictive Maintenance (status + alert)
    - Stage 4: MLOps Monitoring (drift / anomaly detection)

    Result is persisted to MongoDB Serving History.
    """
    run_id = request.run_id or f"api_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    h = np.array(request.h_signal, dtype=np.float32)
    v = np.array(request.v_signal, dtype=np.float32)

    if len(h) != len(v):
        raise HTTPException(
            status_code=422,
            detail=f"h_signal and v_signal must have the same length ({len(h)} vs {len(v)}).",
        )

    pipeline = _get_pipeline()
    result   = pipeline.run_burst(
        run_id       = run_id,
        bearing_name = request.bearing_name,
        burst_idx    = request.burst_idx,
        h_signal     = h,
        v_signal     = v,
    )

    pm_out  = result.get("pm")  or {}
    mon_out = result.get("monitoring") or {}

    return BurstResponse(
        ok          = result["ok"],
        ready       = result["ready"],
        run_id      = run_id,
        bearing     = request.bearing_name,
        burst_idx   = request.burst_idx,
        pm          = PMResult(**{k: pm_out.get(k) for k in PMResult.__fields__}) if result["ready"] else None,
        monitoring  = MonitorResult(**{k: mon_out.get(k) for k in MonitorResult.__fields__}) if result["ready"] else None,
        record_id   = result.get("record_id"),
        error       = result.get("error"),
        timestamp   = datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /serve/pipeline/bearing  — full bearing (background)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/serve/pipeline/bearing", response_model=Dict[str, str])
async def run_pipeline_for_bearing(
    request: BearingPipelineRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start pipeline over an entire bearing folder in the background.

    Returns immediately with a run_id.
    Poll /serving-history/run/{run_id}/summary for progress.
    """
    run_id = f"api_bearing_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(_bg_run_bearing, run_id, request)
    return {
        "run_id":  run_id,
        "status":  "started",
        "bearing": request.bearing_name,
    }


async def _bg_run_bearing(run_id: str, request: BearingPipelineRequest):
    try:
        pipeline = _get_pipeline()
        pipeline.reset_bearing()
        results = pipeline.run_bearing(
            run_id        = run_id,
            bearing_name  = request.bearing_name,
            source_folder = request.source_folder,
            burst_period  = request.burst_period,
            realtime      = request.realtime,
            max_bursts    = request.max_bursts,
        )
        n_alerts = sum(1 for r in results if r.get("ready") and r.get("pm", {}).get("alert"))
        logger.info(
            f"[API] Bearing pipeline complete — run_id={run_id}  "
            f"bursts={len(results)}  alerts={n_alerts}"
        )
    except Exception as exc:
        logger.error(f"[API] Bearing pipeline error for run_id={run_id}: {exc}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# GET /serving-history  — query Serving History
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/serving-history", response_model=List[Dict[str, Any]])
async def get_serving_history(
    run_id:       Optional[str] = None,
    bearing_name: Optional[str] = None,
    limit:        int           = 50,
):
    """
    Query the Serving History (Audit Service step 10).

    Filter by run_id and/or bearing_name.  Returns up to `limit` records
    sorted newest-first.
    """
    try:
        from utils.serving_history import ServingHistory
        sh = ServingHistory(
            mongo_uri=_pipeline_config["mongo_uri"],
            db_name=_pipeline_config["db_name"],
        )
        return sh.get_serving_metadata(
            run_id=run_id,
            bearing_name=bearing_name,
            limit=min(limit, 500),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/serving-history/run/{run_id}/summary")
async def get_run_summary(run_id: str):
    """Aggregated summary for a pipeline run (for Dashboard and Audit Service)."""
    try:
        from utils.serving_history import ServingHistory
        sh = ServingHistory(
            mongo_uri=_pipeline_config["mongo_uri"],
            db_name=_pipeline_config["db_name"],
        )
        return sh.get_run_summary(run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/serving-history/bearing/{bearing_name}/latest")
async def get_latest_for_bearing(bearing_name: str, n: int = 10):
    """Latest n pipeline records for a specific bearing."""
    try:
        from utils.serving_history import ServingHistory
        sh = ServingHistory(
            mongo_uri=_pipeline_config["mongo_uri"],
            db_name=_pipeline_config["db_name"],
        )
        return sh.get_latest_for_bearing(bearing_name, n=min(n, 100))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/serving-history/bearing/{bearing_name}/drift")
async def get_drift_history(bearing_name: str, limit: int = 50):
    """Recent drift-detected records for a bearing — for Dashboard alerting."""
    try:
        from utils.serving_history import ServingHistory
        sh = ServingHistory(
            mongo_uri=_pipeline_config["mongo_uri"],
            db_name=_pipeline_config["db_name"],
        )
        return sh.get_drift_flags(bearing_name, limit=min(limit, 200))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))