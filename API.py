# api/main_manual.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from utils.model_registry import ModelRegistry
import uuid
import joblib
import numpy as np

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

# ====================================================================
# REQUEST/RESPONSE MODELS
# ====================================================================

class WorkflowTriggerRequest(BaseModel):
    workflow_name: str = "predictive_maintenance"
    config_overrides: Optional[Dict[str, Any]] = None
    priority: str = Field(default="normal", pattern="^(low|normal|high)$")

class WorkflowStatusResponse(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    steps: Dict[str, Dict]

class FeatureIngestionRequest(BaseModel):
    source_path: str
    dataset_id: str
    version: str
    metadata: Optional[Dict] = None

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    timestamp: str

class ModelMetadata(BaseModel):
    model_id: str
    version: str
    accuracy: float
    created_at: str
    status: str

# ====================================================================
# WORKFLOW ENDPOINTS
# ====================================================================

@app.post("/workflow_data/trigger", response_model=Dict[str, str])
async def trigger_workflow(request: WorkflowTriggerRequest, background_tasks: BackgroundTasks):
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    # Add workflow execution to background task queue
    background_tasks.add_task(execute_workflow_async, run_id, request)
    return {"run_id": run_id, "status": "queued", "message": "Workflow execution started"}

@app.get("/workflow_data/{run_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(run_id: str):
    from orchestrator import WorkflowStateManager
    manager = WorkflowStateManager()
    try:
        state = manager.load_state(run_id)
        return WorkflowStatusResponse(**state)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

@app.get("/workflow_data/{run_id}/artifacts")
async def get_workflow_artifacts(run_id: str):
    # Placeholder for artifact retrieval
    return {"artifacts": []}

# ====================================================================
# FEATURE STORE ENDPOINTS
# ====================================================================

@app.post("/features/ingest")
async def ingest_features(request: FeatureIngestionRequest):
    from utils.MongoDB import FeatureStore
    config = {
        "mongo_uri": "mongodb://localhost:27017",
        "db_name": "feature_store",
        "dataset_id": request.dataset_id,
        "version": request.version,
        "df_path": request.source_path,
        "metadata": request.metadata
    }
    store = FeatureStore(config)
    result = store.run()
    return result

@app.get("/features/{dataset_id}/{version}")
async def get_features(dataset_id: str, version: str = "latest", feature_type: Optional[str] = None):
    from utils.MongoDB import FeatureStore
    store = FeatureStore({"mongo_uri": "mongodb://localhost:27017", "db_name": "feature_store"})
    df = store.get_features(dataset_id, version, feature_type)
    return {"dataset_id": dataset_id, "version": version, "rows": len(df), "columns": list(df.columns)}

# ====================================================================
# MODEL SERVING ENDPOINTS
# ====================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_fault(request: PredictionRequest):
    model_path = f"models/{request.model_version or 'latest'}/model.pkl"
    model = joblib.load(model_path)
    features = np.array([list(request.features.values())])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    return PredictionResponse(
        prediction=str(prediction),
        confidence=float(confidence),
        model_version=request.model_version or "latest",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/batch")
async def predict_batch(features: List[Dict[str, float]], background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(process_batch_predictions, job_id, features)
    return {"job_id": job_id, "status": "processing"}

# ====================================================================
# MODEL REGISTRY ENDPOINTS
# ====================================================================

@app.get("/models", operation_id="list_all_models")
async def list_models(run_id: str, status: Optional[str] = None):
    registry_path = f"workflow_data/{run_id}/models/model_registry/registry.json"
    registry = ModelRegistry(run_id=run_id)
    return registry.list_models(status=status)

@app.post("/models/{model_id}/approve", operation_id="approve_model_by_id")
async def approve_model(run_id: str, model_id: str, approved_by: str):
    registry_path = f"workflow_data/{run_id}/models/model_registry/registry.json"
    registry = ModelRegistry(run_id=run_id)
    success = registry.approve_model(model_id, approved_by)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to approve model")
    return {"status": "approved", "model_id": model_id}

@app.post("/models/{model_id}/deploy")
async def deploy_model(model_id: str, run_id: str):
    registry_path = f"workflow_data/{run_id}/models/model_registry/registry.json"
    registry = ModelRegistry(run_id=run_id)
    success = registry.deploy_model(model_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to deploy model")
    return {"status": "deployed", "model_id": model_id}

@app.get("/models/{target_feature}/deployed")
async def get_deployed_model(target_feature: str, run_id: str):
    registry_path = f"workflow_data/{run_id}/models/model_registry/registry.json"
    registry = ModelRegistry(run_id=run_id)
    model = registry.get_deployed_model(target_feature)
    if not model:
        raise HTTPException(status_code=404, detail="No deployed model found")
    return model

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

async def execute_workflow_async(run_id: str, request: WorkflowTriggerRequest):
    from orchestrator import WorkflowExecutor
    registry = ModelRegistry(run_id=run_id)
    executor = WorkflowExecutor("config/workflow.yaml")
    executor.start_workflow(run_id)

async def process_batch_predictions(job_id: str, features: List[Dict]):
    pass

# ====================================================================
# WEBSOCKET
# ====================================================================

@app.websocket("/ws/workflow_data/{run_id}")
async def workflow_updates(websocket: WebSocket, run_id: str):
    await websocket.accept()
    while True:
        await websocket.send_json({"status": "running", "step": "feature_extraction"})

# ====================================================================
# STARTUP/SHUTDOWN
# ====================================================================

@app.on_event("startup")
async def startup_event():
    pass

@app.on_event("shutdown")
async def shutdown_event():
    pass
