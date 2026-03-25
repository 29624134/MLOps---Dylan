from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from utils.model_registry import ModelRegistry
import uuid
import torch
import joblib
import numpy as np
import io

app = FastAPI(title="PHM 2012 RUL Prediction API", version="1.0.0")


# ====================================================================
# REQUEST / RESPONSE MODELS
# ====================================================================

class WorkflowTriggerRequest(BaseModel):
    workflow_name: str = "rul_prediction"
    config_overrides: Optional[Dict[str, Any]] = None
    priority: str = Field(default="normal", pattern="^(low|normal|high)$")

class WorkflowStatusResponse(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    steps: Dict[str, Dict]

class RULPredictionRequest(BaseModel):
    features: Dict[str, float]
    model_version: Optional[str] = "latest"

class RULPredictionResponse(BaseModel):
    predicted_rul_s: float
    predicted_rul_min: float
    horizon: int
    model_version: str
    timestamp: str

class LiveServingRequest(BaseModel):
    bearing_name: str                    # must match a bearing with role='live' in bearings.json
    realtime: bool = False               # True = sleep burst_period between bursts
    max_bursts: Optional[int] = None     # stop early (useful for testing)


# ====================================================================
# WORKFLOW ENDPOINTS
# ====================================================================

@app.post("/workflow/trigger", response_model=Dict[str, str])
async def trigger_workflow(request: WorkflowTriggerRequest, background_tasks: BackgroundTasks):
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(execute_workflow_async, run_id, request)
    return {"run_id": run_id, "status": "queued"}

@app.get("/workflow/{run_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(run_id: str):
    from orchestrator import WorkflowStateManager
    manager = WorkflowStateManager()
    try:
        state = manager.load_state(run_id)
        return WorkflowStatusResponse(**state)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

@app.get("/workflow/{run_id}/artifacts")
async def get_workflow_artifacts(run_id: str):
    return {"artifacts": []}


# ====================================================================
# RUL PREDICTION ENDPOINT  (single feature vector, synchronous)
# ====================================================================

@app.post("/predict/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """
    Predict remaining useful life for a single burst's feature vector.
    Returns the step-0 prediction from the multi-step horizon.
    """
    registry = ModelRegistry(run_id="__deployed__")
    model_entry = registry.get_deployed_model(target_feature="RUL_s")
    if not model_entry:
        raise HTTPException(status_code=404, detail="No deployed RUL model found")

    checkpoint = torch.load(model_entry["model_path"], map_location="cpu")
    scaler = joblib.load(io.BytesIO(checkpoint["scaler_bytes"]))

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

    # Use direct forward pass (single row — sliding window loop needs horizon rows)
    with torch.no_grad():
        x_in = torch.tensor(features_scaled, dtype=torch.float32)
        preds = rul_model.model(x_in).detach().cpu().numpy().reshape(-1)

    rul_scale     = hp.get("rul_scale", 30000.0)
    predicted_rul = float(np.clip(preds[0] * rul_scale, 0.0, None))
    horizon       = len(preds)

    return RULPredictionResponse(
        predicted_rul_s   = predicted_rul,
        predicted_rul_min = predicted_rul / 60.0,
        horizon           = horizon,
        model_version     = model_entry["model_id"],
        timestamp         = datetime.now().isoformat(),
    )


# ====================================================================
# LIVE SERVING ENDPOINT  (streams a bearing, runs in background)
# ====================================================================

@app.post("/serve/live", response_model=Dict[str, str])
async def start_live_serving(request: LiveServingRequest, background_tasks: BackgroundTasks):
    """
    Start live inference for a bearing that has role='live' in bearings.json.
    Returns immediately with a run_id. Poll /workflow/{run_id}/status for progress.
    Predictions are written to workflow_data/{run_id}/live/{bearing}_predictions.csv
    """
    run_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(_run_live_serving_bg, run_id, request)
    return {"run_id": run_id, "status": "started", "bearing": request.bearing_name}


async def _run_live_serving_bg(run_id: str, request: LiveServingRequest):
    """Background task: stream a live bearing through the deployed model."""
    import os
    import pandas as pd
    from orchestrator import BearingRegistry, WorkflowStateManager
    from Live_implementation.live_feature_buffer import LiveFeatureBuffer
    from Live_implementation.live_predictor      import LivePredictor
    from scripts import DataIngestorPHM

    state_manager = WorkflowStateManager()
    state_manager.create_run_state(run_id, ["live_serving"])
    state_manager.update_step_status(run_id, "live_serving", "RUNNING")

    try:
        # Resolve the bearing
        bearing_registry = BearingRegistry("config/bearings.json")
        matching = [b for b in bearing_registry.live_bearings()
                    if b["name"] == request.bearing_name]
        if not matching:
            raise ValueError(
                f"Bearing '{request.bearing_name}' not found or does not have role='live'. "
                f"Check config/bearings.json."
            )
        bearing       = matching[0]
        source_folder = bearing_registry.source_path(bearing)

        # Load deployed model
        model_registry = ModelRegistry(run_id="__deployed__")
        model_entry    = model_registry.get_deployed_model("RUL_s")
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
        for burst in ingestor.stream_bursts(
            source_folder,
            burst_period=10.0,
            realtime=request.realtime,
        ):
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

        # Save predictions
        out_path = f"workflow_data/{run_id}/live/{request.bearing_name}_predictions.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(predictions).to_csv(out_path, index=False)

        state_manager.mark_step_outputs(run_id, "live_serving", {
            "predictions_path": out_path,
            "n_predictions":    len(predictions),
        })
        state_manager.update_step_status(run_id, "live_serving", "COMPLETE")

    except Exception as e:
        state_manager.update_step_status(run_id, "live_serving", "FAILED", str(e))
        raise


# ====================================================================
# MODEL REGISTRY ENDPOINTS
# ====================================================================

@app.get("/models", operation_id="list_all_models")
async def list_models(run_id: str, status: Optional[str] = None):
    registry = ModelRegistry(run_id=run_id)
    return registry.list_models(status=status)

@app.post("/models/{model_id}/approve", operation_id="approve_model_by_id")
async def approve_model(run_id: str, model_id: str, approved_by: str):
    registry = ModelRegistry(run_id=run_id)
    if not registry.approve_model(model_id, approved_by):
        raise HTTPException(status_code=400, detail="Failed to approve model")
    return {"status": "approved", "model_id": model_id}

@app.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str, run_id: str):
    registry = ModelRegistry(run_id=run_id)
    if not registry.deploy_model(model_id):
        raise HTTPException(status_code=400, detail="Failed to deploy model")
    return {"status": "deployed", "model_id": model_id}

@app.get("/models/{target_feature}/deployed")
async def get_deployed_model(target_feature: str, run_id: str):
    registry = ModelRegistry(run_id=run_id)
    model = registry.get_deployed_model(target_feature)
    if not model:
        raise HTTPException(status_code=404, detail="No deployed model found")
    return model


# ====================================================================
# HELPERS
# ====================================================================

async def execute_workflow_async(run_id: str, request: WorkflowTriggerRequest):
    from orchestrator import WorkflowExecutor
    executor = WorkflowExecutor("config/workflow.yaml")
    executor.start_workflow(run_id)


# ====================================================================
# WEBSOCKET
# ====================================================================

@app.websocket("/ws/workflow/{run_id}")
async def workflow_updates(websocket: WebSocket, run_id: str):
    await websocket.accept()
    while True:
        await websocket.send_json({"status": "running", "run_id": run_id})


# ====================================================================
# STARTUP / SHUTDOWN
# ====================================================================

@app.on_event("startup")
async def startup_event():
    pass

@app.on_event("shutdown")
async def shutdown_event():
    pass