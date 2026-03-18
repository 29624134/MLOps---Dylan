from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
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
    features: Dict[str, float]          # feature_name → value (one burst's worth)
    model_version: Optional[str] = "latest"

class RULPredictionResponse(BaseModel):
    predicted_rul_s: float              # seconds remaining
    predicted_rul_min: float            # minutes remaining (convenience)
    horizon: int                        # steps predicted ahead
    model_version: str
    timestamp: str


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
# RUL PREDICTION ENDPOINT
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

    # Restore scaler
    scaler = joblib.load(io.BytesIO(checkpoint["scaler_bytes"]))

    # Rebuild model
    from models.rul_net_model import RULNetModel
    hp = checkpoint.get("hyperparameters", {})
    rul_model = RULNetModel(**hp)
    state = checkpoint["model_state_dict"]
    input_dim = next(iter(state.values())).shape[1]
    rul_model.model = rul_model.create_model(input_dim=input_dim)
    rul_model.model.load_state_dict(state)
    rul_model.model.eval()
    rul_model.is_trained = True

    # Scale and predict
    features = np.array([list(request.features.values())], dtype=np.float32)
    features_scaled = scaler.transform(features)
    preds = rul_model.predict(features_scaled)  # (1, horizon)

    predicted_rul_s = float(np.clip(preds[0, 0], 0, None))
    horizon = preds.shape[1]

    return RULPredictionResponse(
        predicted_rul_s   = predicted_rul_s,
        predicted_rul_min = predicted_rul_s / 60.0,
        horizon           = horizon,
        model_version     = model_entry["model_id"],
        timestamp         = datetime.now().isoformat(),
    )


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