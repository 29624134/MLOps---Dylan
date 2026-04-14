"""
API.py
═══════════════════════════════════════════════════════════════════════════════
PHM 2012 RUL Prediction API
"""

import io
import logging
import os
import uuid
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from utils.model_registry import ModelRegistry
from utils.workflow_registry import WorkflowRegistry

app = FastAPI(title="PHM 2012 RUL Prediction API", version="1.0.0")

# ── Serving Pipeline router ───────────────────────────────────────────────────
from serving_pipeline.serving_pipeline_routes import router as serving_router
app.include_router(serving_router)


# ====================================================================
# REQUEST / RESPONSE MODELS
# ====================================================================

class WorkflowTriggerRequest(BaseModel):
    workflow_name:    str                      = "rul_prediction"
    config_overrides: Optional[Dict[str, Any]] = None
    priority:         str                      = Field(default="normal", pattern="^(low|normal|high)$")

class WorkflowStatusResponse(BaseModel):
    run_id:     str
    status:     str
    start_time: str
    end_time:   Optional[str] = None
    steps:      Dict[str, Dict]

class RULPredictionRequest(BaseModel):
    features:      Dict[str, float]
    model_version: Optional[str] = "latest"

class RULPredictionResponse(BaseModel):
    predicted_rul_s:   float
    predicted_rul_min: float
    horizon:           int
    model_version:     str
    timestamp:         str

class LiveServingRequest(BaseModel):
    bearing_name: str
    realtime:     bool          = False
    max_bursts:   Optional[int] = None

class RegisterWorkflowRequest(BaseModel):
    workflow_name: str
    version:       str
    definition:    Dict[str, Any]
    trigger:       Optional[Dict[str, Any]] = None
    git_hash:      Optional[str]            = None
    environment:   Optional[Dict[str, str]] = None
    metadata:      Optional[Dict[str, Any]] = None

# ── Bearing lifecycle models ──────────────────────────────────────────────────

class FaultConfirmRequest(BaseModel):
    bearing_name:   str   = Field(..., json_schema_extra={"example": "Bearing1_5"})
    run_id:         str   = Field(..., description="Workflow run_id that produced the features")
    rul_at_failure: float = Field(0.0,       description="Confirmed RUL at failure (seconds)")
    worker_name:    str   = Field("Unknown", description="Maintenance tech name")
    note:           str   = Field("",        description="Optional note")

class FaultDenyRequest(BaseModel):
    bearing_name: str = Field(..., json_schema_extra={"example": "Bearing1_5"})
    worker_name:  str = Field("Unknown")
    note:         str = Field("")

class ContinueRequest(BaseModel):
    worker_name:     str  = Field("Unknown")
    trigger_new_run: bool = Field(True, description="Immediately start ingest+serve for next bearing")


# ====================================================================
# WORKFLOW TRIGGER ENDPOINTS
# ====================================================================

@app.post("/workflow/trigger", response_model=Dict[str, str])
async def trigger_workflow(request: WorkflowTriggerRequest,
                           background_tasks: BackgroundTasks):
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(execute_workflow_async, run_id, request)
    return {"run_id": run_id, "status": "queued"}


@app.get("/workflow/{run_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(run_id: str):
    from orchestrator import WorkflowStateManager
    manager = WorkflowStateManager()
    try:
        state = manager.load_state(run_id)
        if not state:
            raise FileNotFoundError
        return WorkflowStatusResponse(**state)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


@app.get("/workflow/{run_id}/artifacts")
async def get_workflow_artifacts(run_id: str):
    return {"artifacts": []}


async def execute_workflow_async(run_id: str, request: WorkflowTriggerRequest):
    """
    Full workflow background task.

    The orchestrator's start_workflow() will automatically detect whether a
    deployed model already exists and skip training/selection if it does.
    """
    try:
        from orchestrator import WorkflowExecutor
        executor = WorkflowExecutor()
        executor.start_workflow(
            workflow_name    = request.workflow_name,
            config_overrides = request.config_overrides,
        )
    except Exception as e:
        from orchestrator import WorkflowStateManager
        try:
            WorkflowStateManager().update_step_status(
                run_id, "workflow", "FAILED", str(e)
            )
        except Exception:
            pass
        raise


# ====================================================================
# RUL PREDICTION ENDPOINT  (single feature vector, synchronous)
# ====================================================================

@app.post("/predict/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """Predict RUL for a single burst feature vector."""
    registry    = ModelRegistry()
    model_entry = registry.get_deployed_model(target_feature="RUL_s")
    if not model_entry:
        raise HTTPException(status_code=404, detail="No deployed RUL model found")

    checkpoint = torch.load(model_entry["model_path"], map_location="cpu")
    scaler     = joblib.load(io.BytesIO(checkpoint["scaler_bytes"]))

    from models.rul_net_model import RULNetModel
    hp        = checkpoint.get("hyperparameters", {})
    rul_model = RULNetModel(**hp)
    state     = checkpoint["model_state_dict"]
    input_dim = next(iter(state.values())).shape[1]
    rul_model.model = rul_model._build_net(input_dim)
    rul_model.model.load_state_dict(state)
    rul_model.model.eval()
    rul_model.is_trained = True

    features        = np.array([list(request.features.values())], dtype=np.float32)
    features_scaled = scaler.transform(features)

    with torch.no_grad():
        x_in  = torch.tensor(features_scaled, dtype=torch.float32)
        preds = rul_model.model(x_in).detach().cpu().numpy().reshape(-1)

    rul_scale     = hp.get("rul_scale", 30000.0)
    predicted_rul = float(np.clip(preds[0] * rul_scale, 0.0, None))

    return RULPredictionResponse(
        predicted_rul_s   = predicted_rul,
        predicted_rul_min = predicted_rul / 60.0,
        horizon           = len(preds),
        model_version     = model_entry["model_id"],
        timestamp         = datetime.now().isoformat(),
    )


# ====================================================================
# LIVE SERVING ENDPOINT  (streams a bearing, runs in background)
# ====================================================================

@app.post("/serve/live", response_model=Dict[str, str])
async def start_live_serving(request: LiveServingRequest,
                              background_tasks: BackgroundTasks):
    """
    Start live inference for a named bearing.
    Returns immediately with a run_id.
    Poll /workflow/{run_id}/status for progress.
    """
    run_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(_run_live_serving_bg, run_id, request)
    return {"run_id": run_id, "status": "started", "bearing": request.bearing_name}


async def _run_live_serving_bg(run_id: str, request: LiveServingRequest):
    """Background task: stream a named bearing through the deployed model."""
    from orchestrator import WorkflowStateManager, BearingRegistry
    from Live_implementation.live_feature_buffer import LiveFeatureBuffer
    from Live_implementation.live_predictor import LivePredictor
    from scripts.data_ingestor import DataIngestorPHM

    state_manager = WorkflowStateManager()
    state_manager.update_step_status(run_id, "live_serving", "RUNNING")

    try:
        bearing_registry = BearingRegistry("config/bearings.json")
        matching = [b for b in bearing_registry.all_bearings()
                    if b["name"] == request.bearing_name]
        if not matching:
            raise ValueError(
                f"Bearing '{request.bearing_name}' not found in bearings.json."
            )

        bearing       = matching[0]
        source_folder = bearing_registry.source_path(bearing)

        model_entry = ModelRegistry().get_deployed_model("RUL_s")
        if not model_entry:
            raise RuntimeError(
                "No deployed model found. Trigger a training workflow first."
            )

        predictor = LivePredictor.from_path(model_entry["model_path"])
        buffer    = LiveFeatureBuffer(window_size=40)
        ingestor  = DataIngestorPHM(config={
            "input_location":  source_folder,
            "output_location": source_folder,
        })

        predictions = []
        for burst in ingestor.stream_bursts(source_folder,
                                            burst_period=10.0,
                                            realtime=request.realtime):
            if request.max_bursts and burst["burst_idx"] >= request.max_bursts:
                break
            vec = buffer.push_burst(burst["h_signal"], burst["v_signal"])
            if vec is None:
                continue
            rul_s = predictor.predict(vec)
            predictions.append({
                "bearing":   request.bearing_name,
                "burst_idx": burst["burst_idx"],
                "time_s":    burst["time_s"],
                "rul_s":     rul_s,
                "rul_min":   rul_s / 60.0,
                "h_max":     burst["h_max"],
                "v_max":     burst["v_max"],
            })

        out_path = f"workflow_data/{run_id}/live/{request.bearing_name}_predictions.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(predictions).to_csv(out_path, index=False)

        state_manager.mark_step_outputs(run_id, "live_serving",
                                        {"predictions_path": out_path,
                                         "n_predictions":    len(predictions)})
        state_manager.update_step_status(run_id, "live_serving", "COMPLETE")

    except Exception as e:
        state_manager.update_step_status(run_id, "live_serving", "FAILED", str(e))
        raise


# ====================================================================
# MODEL REGISTRY ENDPOINTS
# ====================================================================

@app.get("/models", operation_id="list_all_models")
async def list_models(
    status:         Optional[str] = None,
    run_id:         Optional[str] = None,
    target_feature: Optional[str] = None,
):
    registry = ModelRegistry()
    return registry.list_models(status=status, run_id=run_id,
                                 target_feature=target_feature)


@app.post("/models/{model_id}/approve", operation_id="approve_model_by_id")
async def approve_model(model_id: str, approved_by: str):
    registry = ModelRegistry()
    if not registry.approve_model(model_id, approved_by):
        raise HTTPException(status_code=400, detail="Failed to approve model")
    return {"status": "approved", "model_id": model_id}


@app.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    registry = ModelRegistry()
    if not registry.deploy_model(model_id):
        raise HTTPException(status_code=400, detail="Failed to deploy model")
    return {"status": "deployed", "model_id": model_id}


@app.get("/models/{target_feature}/deployed")
async def get_deployed_model(target_feature: str):
    registry = ModelRegistry()
    model = registry.get_deployed_model(target_feature)
    if not model:
        raise HTTPException(status_code=404, detail="No deployed model found")
    return model


# ====================================================================
# WORKFLOW REGISTRY ENDPOINTS
# ====================================================================

@app.post("/workflows", operation_id="register_workflow")
async def register_workflow(request: RegisterWorkflowRequest):
    registry    = WorkflowRegistry()
    workflow_id = registry.register_workflow(
        workflow_name = request.workflow_name,
        version       = request.version,
        definition    = request.definition,
        trigger       = request.trigger,
        git_hash      = request.git_hash,
        environment   = request.environment,
        metadata      = request.metadata,
    )
    return {"status": "registered", "workflow_id": workflow_id}


@app.get("/workflows", operation_id="list_workflows")
async def list_workflows(
    workflow_name: Optional[str] = None,
    status:        Optional[str] = None,
):
    registry = WorkflowRegistry()
    return registry.list_workflows(workflow_name=workflow_name, status=status)


@app.post("/workflows/{workflow_id}/approve", operation_id="approve_workflow")
async def approve_workflow(workflow_id: str, approved_by: str):
    registry = WorkflowRegistry()
    if not registry.approve_workflow(workflow_id, approved_by):
        raise HTTPException(status_code=400, detail="Failed to approve workflow")
    return {"status": "approved", "workflow_id": workflow_id}


@app.post("/workflows/{workflow_id}/reject", operation_id="reject_workflow")
async def reject_workflow(workflow_id: str, rejected_by: str, reason: str = ""):
    registry = WorkflowRegistry()
    if not registry.reject_workflow(workflow_id, rejected_by, reason=reason):
        raise HTTPException(status_code=400, detail="Failed to reject workflow")
    return {"status": "rejected", "workflow_id": workflow_id}


@app.get("/workflows/{workflow_name}/active", operation_id="get_active_workflow")
async def get_active_workflow(workflow_name: str):
    registry = WorkflowRegistry()
    wf = registry.get_active_workflow(workflow_name)
    if not wf:
        raise HTTPException(status_code=404, detail="No active workflow found")
    return wf


@app.get("/workflows/{workflow_id}", operation_id="get_workflow")
async def get_workflow(workflow_id: str):
    registry = WorkflowRegistry()
    wf = registry.get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf


# ====================================================================
# BEARING LIFECYCLE ENDPOINTS
# ====================================================================

@app.get("/bearing/current", operation_id="get_current_bearing",
         tags=["Bearing Lifecycle"])
def get_current_bearing():
    """Return the bearing currently at the head of the live queue."""
    from orchestrator import BearingRegistry
    reg     = BearingRegistry("config/bearings.json")
    bearing = reg.current_live_bearing()
    if not bearing:
        return {"current_bearing": None, "queue_exhausted": True}
    return {
        "current_bearing": bearing,
        "queue_index":     reg.current_live_index,
        "queue_length":    len(reg.live_queue),
        "remaining":       len(reg.live_queue) - reg.current_live_index,
    }


@app.get("/bearing/queue", operation_id="get_bearing_queue",
         tags=["Bearing Lifecycle"])
def get_bearing_queue():
    """Show the full live bearing queue and current position."""
    from orchestrator import BearingRegistry
    reg = BearingRegistry("config/bearings.json")
    queue_detail = []
    for i, name in enumerate(reg.live_queue):
        bearing = reg.get_bearing(name)
        queue_detail.append({
            "index":      i,
            "name":       name,
            "status":     bearing.get("status", "unknown") if bearing else "not found",
            "is_current": i == reg.current_live_index,
        })
    return {
        "queue":         queue_detail,
        "current_index": reg.current_live_index,
        "queue_length":  len(reg.live_queue),
    }


@app.post("/bearing/reset-queue", operation_id="reset_bearing_queue",
          tags=["Bearing Lifecycle"])
def reset_bearing_queue():
    """
    Reset the live bearing queue back to the first bearing.

    Sets current_live_index to 0 and resets all live bearings that are not
    already 'available' back to 'available', so they can be re-processed
    from scratch. Does not delete any files from disk.
    """
    from orchestrator import BearingRegistry
    reg = BearingRegistry("config/bearings.json")

    reset_bearings = []
    for b in reg.live_bearings():
        if b.get("status") != "available":
            reg.set_status(b["name"], "available")
            reset_bearings.append(b["name"])

    reg.current_live_index = 0
    reg._save()

    first = reg.current_live_bearing()
    logger.info(
        f"[reset-queue] Queue reset to index 0. "
        f"Bearings reset to available: {reset_bearings}"
    )
    return {
        "status":         "reset",
        "current_index":  0,
        "first_bearing":  first["name"] if first else None,
        "bearings_reset": reset_bearings,
    }


@app.post("/bearing/confirm-fault", operation_id="confirm_bearing_fault",
          tags=["Bearing Lifecycle"])
def confirm_fault(request: FaultConfirmRequest):
    """
    Maintenance tech confirms a fault.

    1. Re-labels features.csv with confirmed RUL values.
    2. Pushes labelled data to MongoDB Feature Store (confirmed_faults collection).
    3. Sets bearing status → 'confirmed'.

    Call POST /bearing/continue afterwards to advance to the next bearing.
    """
    from orchestrator import WorkflowExecutor
    try:
        executor = WorkflowExecutor()
        result   = executor.confirm_fault_and_push_to_store(
            bearing_name   = request.bearing_name,
            run_id         = request.run_id,
            rul_at_failure = request.rul_at_failure,
        )
        return {
            "status":       "confirmed",
            "bearing":      request.bearing_name,
            "worker":       request.worker_name,
            "confirmed_at": datetime.now().isoformat(),
            "store_result": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bearing/deny-fault", operation_id="deny_bearing_fault",
          tags=["Bearing Lifecycle"])
def deny_fault(request: FaultDenyRequest):
    """
    Maintenance tech denies the fault (false positive).
    Sets bearing status → 'denied'. Features NOT pushed to Feature Store.
    Call POST /bearing/continue to advance the queue.
    """
    from orchestrator import BearingRegistry
    reg     = BearingRegistry("config/bearings.json")
    bearing = reg.get_bearing(request.bearing_name)
    if not bearing:
        raise HTTPException(status_code=404,
                            detail=f"Bearing '{request.bearing_name}' not found.")
    reg.set_status(request.bearing_name, "denied")
    return {
        "status":    "denied",
        "bearing":   request.bearing_name,
        "worker":    request.worker_name,
        "denied_at": datetime.now().isoformat(),
    }


@app.post("/bearing/continue", operation_id="continue_to_next_bearing",
          tags=["Bearing Lifecycle"])
def continue_to_next_bearing(request: ContinueRequest,
                              background_tasks: BackgroundTasks):
    """
    Tech clicks 'Continue' after confirming or denying a fault.

    1. Advances the live bearing queue to the next bearing.
    2. If trigger_new_run=True (default), fires a background task that
       ingests, extracts, and serves the new bearing. Training is skipped
       because a deployed model already exists at this point.
    """
    from orchestrator import BearingRegistry
    reg    = BearingRegistry("config/bearings.json")
    next_b = reg.advance_live_bearing()

    if not next_b:
        return {
            "status":       "queue_exhausted",
            "next_bearing": None,
            "message":      "All live bearings have been processed.",
        }

    run_id = None
    if request.trigger_new_run:
        run_id = f"bearing_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        background_tasks.add_task(_run_bearing_workflow_bg, run_id)

    return {
        "status":       "advanced",
        "next_bearing": next_b,
        "run_id":       run_id,
        "message":      (f"Now serving {next_b['name']}. "
                         + (f"Background run {run_id} started." if run_id else "")),
    }


async def _run_bearing_workflow_bg(run_id: str):
    """
    Background workflow triggered by POST /bearing/continue after a fault is
    confirmed or denied.

    Full sequence
    ─────────────
    1. Ingest      — consolidate raw CSVs for the new live bearing
    2. Extract     — compute features.csv for the new live bearing
    3. Retrain     — train on all train-role files + any confirmed fault files
                     (new model registers as 'pending'; existing deployed model
                      stays live until a human approves the new one)
    4. Model sel.  — if no deployed model yet, auto-approve and deploy;
                     otherwise leave new model as pending
    5. Serve       — stream new bearing burst-by-burst through deployed model
    """
    from orchestrator import WorkflowExecutor, WorkflowStateManager

    state_mgr = WorkflowStateManager()
    state_mgr.init_state(run_id, "bearing_serve")

    try:
        executor = WorkflowExecutor()
        reg      = executor.registry
        bearing  = reg.current_live_bearing()

        if not bearing:
            state_mgr.mark_workflow_failed(run_id, "No live bearing available.")
            return

        # 1. Extract directly from acc_*.csv — no ingestion/parquet step needed
        executor._run_extraction(run_id, bearing)

        # 3. Retrain — includes any confirmed fault files from previous bearings
        confirmed_files = executor._confirmed_fault_files()
        if confirmed_files:
            logger.info(
                f"[{run_id}] Confirmed fault data found "
                f"({len(confirmed_files)} file(s)) — triggering retrain."
            )
            executor._run_training(run_id)
            executor._run_model_selection(run_id)
        else:
            logger.info(
                f"[{run_id}] No confirmed fault data — skipping retrain, "
                f"using existing deployed model."
            )

        # 4. Serve — always uses the currently deployed model
        executor._run_serving_pipeline(run_id)

        state_mgr.mark_workflow_complete(run_id)

    except Exception as e:
        state_mgr.mark_workflow_failed(run_id, traceback.format_exc())