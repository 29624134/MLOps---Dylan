"""
API.py
═══════════════════════════════════════════════════════════════════════════════
PHM 2012 RUL Prediction API  —  v2.0.0

Fixes applied
─────────────
Fix #1  — Val bearings are NOT re-fetched from disk on retraining;
           orchestrator reads from MongoDB first.
Fix #2  — Serving is NEVER stopped on a critical PM status. The pipeline
           keeps predicting every burst. Only explicit user actions
           (/bearing/continue, /bearing/reset-queue) stop processes.
Fix #3  — ModelRegistry is MongoDB-backed (model_registry.py rewritten).
           All /models endpoints read/write to Mongo.
Fix #4  — Only two GUIs exist: gui_fault_review.py (8501) and
           gui_rul_monitor.py (8502). The old dashboard/app.py is gone.
           This file no longer references it.
Fix #5  — champion.json is only written when the serving lock is clear.
           ProcessManager.stop_bearing() waits for the serving_lock before
           terminating the serving process so we never kill it mid-burst.
Fix #6  — orchestrator.run_training_only() reads all data from MongoDB.
           The API triggers it unchanged; no local-disk reads in training.

Process lifecycle
─────────────────
  POST /workflow/trigger
      → trains model (first run) / extracts features (subsequent runs)
      → auto-starts scada_simulator.py + run_serving.py

  POST /bearing/confirm-fault
      → pushes confirmed features to MongoDB FS Mirrored
      → starts run_preprod.py retraining in background
      → serving continues uninterrupted (Fix #2)

  POST /bearing/continue
      → waits for serving lock to clear (Fix #5)
      → stops SCADA + Serving processes
      → advances bearing queue
      → starts extraction + fresh SCADA + Serving for new bearing

  POST /bearing/reset-queue
      → stops all processes, resets queue index

  GET  /bearing/processes
      → shows subprocess status (scada, serving, preprod)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
import sys
import uuid
import subprocess
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.model_registry import ModelRegistry
from utils.workflow_registry import WorkflowRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PHM 2012 RUL Prediction API",
    version="2.0.0",
    description=(
        "Predictive Maintenance — Remaining Useful Life pipeline. "
        "Model Registry backed by MongoDB. "
        "GUIs: gui_fault_review.py :8501 | gui_rul_monitor.py :8502"
    ),
)

# ── Serving Pipeline router ───────────────────────────────────────────────────
from serving_pipeline.serving_pipeline_routes import router as serving_router
app.include_router(serving_router)

# ── MongoDB defaults (shared with model_registry / orchestrator) ──────────────
DEFAULT_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_DB_NAME   = os.environ.get("MONGO_DB",  "phm_mlops")
LOCK_COLLECTION   = "serving_lock"   # Fix #5: written by run_serving.py per burst


# ====================================================================
# PROCESS MANAGER  — owns scada / serving / preprod subprocesses
# ====================================================================

class ProcessManager:
    """
    Singleton that owns SCADA, Serving, and Pre-Production subprocesses.

    Fix #2: stop_all() does NOT stop serving because of a critical PM status.
            Serving is stopped only on explicit user action (continue / reset).
    Fix #5: _wait_for_serving_idle() polls serving_lock in MongoDB before
            terminating the serving process so we never kill it mid-burst.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._scada_proc      = None
            cls._instance._serving_proc    = None
            cls._instance._preprod_proc    = None
            cls._instance._current_bearing = None
        return cls._instance

    # ── Serving-lock helper (Fix #5) ──────────────────────────────────────────

    def _wait_for_serving_idle(
        self,
        bearing_name:     str,
        max_wait_s:       float = 15.0,
        poll_interval_s:  float = 0.5,
    ) -> None:
        """
        Block until run_serving.py signals it is between bursts (lock released),
        or until max_wait_s elapses.  Uses the same serving_lock collection that
        model_registry.write_champion_pointer() watches.
        """
        try:
            from pymongo import MongoClient
            client    = MongoClient(DEFAULT_MONGO_URI, serverSelectionTimeoutMS=2000)
            lock_col  = client[DEFAULT_DB_NAME][LOCK_COLLECTION]
            waited    = 0.0
            while waited < max_wait_s:
                doc = lock_col.find_one({"bearing_name": bearing_name, "active": True})
                if doc is None:
                    return   # idle — safe to stop
                logger.info(
                    f"[ProcessManager] Serving is mid-burst for {bearing_name} "
                    f"— waiting {poll_interval_s}s before stopping "
                    f"({waited:.1f}/{max_wait_s:.0f}s)..."
                )
                time.sleep(poll_interval_s)
                waited += poll_interval_s
            logger.warning(
                f"[ProcessManager] Serving did not become idle within "
                f"{max_wait_s}s — proceeding with stop anyway."
            )
        except Exception as e:
            logger.warning(
                f"[ProcessManager] Could not check serving_lock: {e}. "
                "Proceeding with stop."
            )

    # ── Start ─────────────────────────────────────────────────────────────────

    def start_bearing(self, bearing_name: str, realtime: bool = False) -> Dict:
        """
        Stop any existing processes (waiting for burst boundary — Fix #5) then
        start fresh scada + serving for bearing_name.
        """
        # Wait for serving to finish current burst before killing it (Fix #5)
        if self._current_bearing:
            self._wait_for_serving_idle(self._current_bearing)

        self.stop_all()

        logger.info(f"[ProcessManager] Starting SCADA + Serving for {bearing_name}")

        scada_cmd = [sys.executable, "scada_simulator.py", "--bearing", bearing_name]
        if realtime:
            scada_cmd.append("--realtime")

        serving_cmd = [sys.executable, "run_serving.py", "--bearing", bearing_name]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            self._scada_proc = subprocess.Popen(
                scada_cmd, env=env, stdout=None, stderr=None
            )
            self._serving_proc = subprocess.Popen(
                serving_cmd, env=env, stdout=None, stderr=None
            )
            self._current_bearing = bearing_name
            logger.info(
                f"[ProcessManager] SCADA PID={self._scada_proc.pid}  "
                f"Serving PID={self._serving_proc.pid}"
            )
            return {
                "bearing":     bearing_name,
                "scada_pid":   self._scada_proc.pid,
                "serving_pid": self._serving_proc.pid,
            }
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to start processes: {e}")
            self.stop_all()
            raise RuntimeError(f"Could not start bearing processes: {e}")

    def start_preprod(self, run_id: str) -> Optional[int]:
        """
        Launch run_preprod.py as a background subprocess.
        Only one preprod process runs at a time — previous one is terminated first.
        """
        if self._preprod_proc and self._preprod_proc.poll() is None:
            logger.info(
                f"[ProcessManager] Terminating previous preprod "
                f"(PID={self._preprod_proc.pid})"
            )
            self._preprod_proc.terminate()

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            self._preprod_proc = subprocess.Popen(
                [sys.executable, "run_preprod.py", "--run_id", run_id],
                env=env, stdout=None, stderr=None,
            )
            logger.info(
                f"[ProcessManager] Pre-Production started "
                f"(PID={self._preprod_proc.pid}, run_id={run_id})"
            )
            return self._preprod_proc.pid
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to start run_preprod.py: {e}")
            return None

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop_all(self) -> None:
        """
        Gracefully terminate SCADA and Serving subprocesses.

        Fix #2: This is called only on explicit user actions (continue / reset).
                It is NEVER triggered automatically by a critical PM status.
        """
        for name, proc in [
            ("SCADA",   self._scada_proc),
            ("Serving", self._serving_proc),
        ]:
            if proc and proc.poll() is None:
                logger.info(f"[ProcessManager] Stopping {name} (PID={proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        self._scada_proc      = None
        self._serving_proc    = None
        self._current_bearing = None

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        def _ps(proc, name):
            if proc is None:
                return {"name": name, "running": False, "pid": None}
            return {"name": name, "running": proc.poll() is None, "pid": proc.pid}

        return {
            "current_bearing": self._current_bearing,
            "scada":           _ps(self._scada_proc,   "scada_simulator"),
            "serving":         _ps(self._serving_proc, "run_serving"),
            "preprod":         _ps(self._preprod_proc, "run_preprod"),
        }


# Module-level singleton
_process_manager = ProcessManager()


# ====================================================================
# REQUEST / RESPONSE MODELS
# ====================================================================

class WorkflowTriggerRequest(BaseModel):
    workflow_name:      str                      = "rul_prediction"
    config_overrides:   Optional[Dict[str, Any]] = None
    priority:           str                      = Field(
        default="normal", pattern="^(low|normal|high)$"
    )
    auto_start_serving: bool = Field(
        True,
        description="Auto-start SCADA simulator and Serving pipeline after workflow",
    )
    realtime: bool = Field(
        False,
        description="Run SCADA simulator in realtime mode (10 s between bursts)",
    )


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
    trigger_new_run: bool = Field(
        True, description="Immediately start extraction + serving for next bearing"
    )
    realtime: bool = Field(
        False, description="Run SCADA simulator in realtime mode for new bearing"
    )


# ====================================================================
# WORKFLOW TRIGGER ENDPOINTS
# ====================================================================

@app.post("/workflow/trigger", response_model=Dict[str, str],
          tags=["Workflow"])
async def trigger_workflow(
    request: WorkflowTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the full workflow.

    First run  : ingest → extract → train → deploy model
    Subsequent : extract only (deployed model already exists)

    After completion, auto-starts SCADA simulator + Serving Pipeline for
    the current live bearing (unless auto_start_serving=False).
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(execute_workflow_async, run_id, request)
    return {"run_id": run_id, "status": "queued"}


@app.get("/workflow/{run_id}/status", response_model=WorkflowStatusResponse,
         tags=["Workflow"])
async def get_workflow_status(run_id: str):
    from orchestrator import WorkflowStateManager
    manager = WorkflowStateManager()
    try:
        state = manager.get_state(run_id)
        if not state:
            raise FileNotFoundError
        return WorkflowStatusResponse(**state)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@app.get("/workflow/{run_id}/artifacts", tags=["Workflow"])
async def get_workflow_artifacts(run_id: str):
    return {"artifacts": []}


async def execute_workflow_async(run_id: str, request: WorkflowTriggerRequest):
    """
    Background task: runs start_workflow() then auto-starts SCADA + Serving.

    Fix #6: orchestrator.start_workflow() calls _run_training() which reads
            ALL data from MongoDB — no local-disk reads during training.
    """
    try:
        from orchestrator import WorkflowExecutor, BearingRegistry
        executor = WorkflowExecutor()
        executor.start_workflow(run_id=run_id)

        if request.auto_start_serving:
            reg     = BearingRegistry("config/bearings.json")
            bearing = reg.current_live_bearing()
            if bearing:
                logger.info(
                    f"[{run_id}] Workflow complete — auto-starting SCADA + "
                    f"Serving for {bearing['name']}"
                )
                _process_manager.start_bearing(
                    bearing_name=bearing["name"],
                    realtime=request.realtime,
                )
            else:
                logger.warning(
                    f"[{run_id}] Workflow complete — no live bearing in queue."
                )
    except Exception as e:
        from orchestrator import WorkflowStateManager
        try:
            WorkflowStateManager().mark_workflow_failed(run_id, traceback.format_exc())
        except Exception:
            pass
        logger.error(f"[{run_id}] Workflow failed: {e}", exc_info=True)
        raise


# ====================================================================
# PROCESS STATUS ENDPOINT
# ====================================================================

@app.get("/bearing/processes", tags=["Bearing Lifecycle"])
def get_process_status():
    """Return current status of SCADA, Serving, and Pre-Production subprocesses."""
    return _process_manager.status()


# ====================================================================
# RUL PREDICTION ENDPOINT  (single feature vector, synchronous)
# ====================================================================

@app.post("/predict/rul", response_model=RULPredictionResponse,
          tags=["Prediction"])
async def predict_rul(request: RULPredictionRequest):
    """
    Predict RUL for a single burst feature vector.

    Fix #3: Deployed model is looked up from MongoDB via ModelRegistry.
    """
    try:
        registry    = ModelRegistry()   # Fix #3: MongoDB-backed
        model_entry = registry.get_deployed_model("RUL_s")
        if not model_entry:
            raise HTTPException(status_code=404, detail="No deployed model found.")

        from Live_implementation.live_predictor import LivePredictor
        predictor = LivePredictor.from_path(model_entry["model_path"])

        feature_vec = np.array(list(request.features.values()), dtype=np.float32)
        rul_s       = float(predictor.predict(feature_vec))
        rul_min     = rul_s / 60.0

        return RULPredictionResponse(
            predicted_rul_s   = rul_s,
            predicted_rul_min = rul_min,
            horizon           = 10,
            model_version     = model_entry["model_id"],
            timestamp         = datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# MODEL REGISTRY ENDPOINTS  (Fix #3 — MongoDB-backed)
# ====================================================================

@app.get("/models", operation_id="list_all_models", tags=["Model Registry"])
async def list_models(
    status:         Optional[str] = None,
    run_id:         Optional[str] = None,
    target_feature: Optional[str] = None,
):
    """List models from MongoDB model registry."""
    registry = ModelRegistry()
    return registry.list_models(
        status=status, run_id=run_id, target_feature=target_feature
    )


@app.post("/models/{model_id}/approve", operation_id="approve_model_by_id",
          tags=["Model Registry"])
async def approve_model(model_id: str, approved_by: str):
    registry = ModelRegistry()
    if not registry.approve_model(model_id, approved_by):
        raise HTTPException(status_code=400, detail="Failed to approve model")
    return {"status": "approved", "model_id": model_id}


@app.post("/models/{model_id}/deploy", tags=["Model Registry"])
async def deploy_model(model_id: str):
    """
    Deploy a model.

    Fix #5: ModelRegistry.deploy_model() → write_champion_pointer() waits
            for serving_lock to clear before writing champion.json.
    """
    registry = ModelRegistry()
    if not registry.deploy_model(model_id):
        raise HTTPException(status_code=400, detail="Failed to deploy model")
    return {"status": "deployed", "model_id": model_id}


@app.post("/models/{model_id}/archive", tags=["Model Registry"])
async def archive_model(model_id: str):
    registry = ModelRegistry()
    if not registry.archive_model(model_id):
        raise HTTPException(status_code=400, detail="Failed to archive model")
    return {"status": "archived", "model_id": model_id}


@app.get("/models/{target_feature}/deployed", tags=["Model Registry"])
async def get_deployed_model(target_feature: str):
    """Return the currently deployed model for a given target feature."""
    registry = ModelRegistry()
    model    = registry.get_deployed_model(target_feature)
    if not model:
        raise HTTPException(status_code=404, detail="No deployed model found")
    return model


@app.get("/models/champion", tags=["Model Registry"])
async def get_champion():
    """Return the current champion.json pointer (used for hot-swap)."""
    registry  = ModelRegistry()
    champion  = registry.read_champion_pointer()
    if not champion:
        raise HTTPException(status_code=404, detail="No champion model found")
    return champion


# ====================================================================
# WORKFLOW REGISTRY ENDPOINTS
# ====================================================================

@app.post("/workflows", operation_id="register_workflow",
          tags=["Workflow Registry"])
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


@app.get("/workflows", operation_id="list_workflows",
         tags=["Workflow Registry"])
async def list_workflows(
    workflow_name: Optional[str] = None,
    status:        Optional[str] = None,
):
    registry = WorkflowRegistry()
    return registry.list_workflows(workflow_name=workflow_name, status=status)


@app.post("/workflows/{workflow_id}/approve",
          operation_id="approve_workflow", tags=["Workflow Registry"])
async def approve_workflow(workflow_id: str, approved_by: str):
    registry = WorkflowRegistry()
    if not registry.approve_workflow(workflow_id, approved_by):
        raise HTTPException(status_code=400, detail="Failed to approve workflow")
    return {"status": "approved", "workflow_id": workflow_id}


@app.post("/workflows/{workflow_id}/reject",
          operation_id="reject_workflow", tags=["Workflow Registry"])
async def reject_workflow(workflow_id: str, rejected_by: str, reason: str = ""):
    registry = WorkflowRegistry()
    if not registry.reject_workflow(workflow_id, rejected_by, reason=reason):
        raise HTTPException(status_code=400, detail="Failed to reject workflow")
    return {"status": "rejected", "workflow_id": workflow_id}


@app.get("/workflows/{workflow_name}/active",
         operation_id="get_active_workflow", tags=["Workflow Registry"])
async def get_active_workflow(workflow_name: str):
    registry = WorkflowRegistry()
    wf = registry.get_active_workflow(workflow_name)
    if not wf:
        raise HTTPException(status_code=404, detail="No active workflow found")
    return wf


@app.get("/workflows/{workflow_id}",
         operation_id="get_workflow", tags=["Workflow Registry"])
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
    reg          = BearingRegistry("config/bearings.json")
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

    Waits for the serving lock to clear (Fix #5) then stops all processes.
    Sets current_live_index to 0 and resets all live bearing statuses to
    'available'. Does not delete any files from disk.
    """
    from orchestrator import BearingRegistry

    # Wait for burst boundary before stopping (Fix #5)
    if _process_manager._current_bearing:
        _process_manager._wait_for_serving_idle(_process_manager._current_bearing)

    _process_manager.stop_all()

    reg            = BearingRegistry("config/bearings.json")
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
    Maintenance worker confirms a fault.

    1. Re-labels features with confirmed RUL values.
    2. Pushes labelled data to MongoDB 'confirmed_faults' (FS Mirrored).
    3. Sets bearing status → 'confirmed'.
    4. Starts run_preprod.py retraining in background.

    Fix #2: Serving pipeline is NOT stopped — it continues predicting.
    Fix #5: Retraining writes champion.json only when serving is idle.
    Fix #6: Retraining reads all data from MongoDB (via orchestrator).

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

        preprod_run_id = (
            f"preprod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
        preprod_pid = _process_manager.start_preprod(preprod_run_id)
        logger.info(
            f"[confirm-fault] Fault confirmed for {request.bearing_name} — "
            f"Pre-Production retraining started (PID={preprod_pid}, "
            f"run_id={preprod_run_id}). Serving continues uninterrupted."
        )

        return {
            "status":         "confirmed",
            "bearing":        request.bearing_name,
            "worker":         request.worker_name,
            "confirmed_at":   datetime.now().isoformat(),
            "store_result":   result,
            "preprod_run_id": preprod_run_id,
            "preprod_pid":    preprod_pid,
            "message": (
                f"Fault confirmed. Data pushed to FS Mirrored. "
                f"Retraining started in background (run_id={preprod_run_id}). "
                f"Serving pipeline continues — model will hot-swap if new "
                f"model is better."
            ),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bearing/deny-fault", operation_id="deny_bearing_fault",
          tags=["Bearing Lifecycle"])
def deny_fault(request: FaultDenyRequest):
    """
    Maintenance worker denies the fault (false positive).

    Sets bearing status → 'denied'. Features NOT pushed to Feature Store.
    No retraining triggered.
    Fix #2: Serving pipeline continues uninterrupted.

    Call POST /bearing/continue to advance the queue.
    """
    from orchestrator import BearingRegistry
    reg     = BearingRegistry("config/bearings.json")
    bearing = reg.get_bearing(request.bearing_name)
    if not bearing:
        raise HTTPException(
            status_code=404,
            detail=f"Bearing '{request.bearing_name}' not found.",
        )
    reg.set_status(request.bearing_name, "denied")
    logger.info(
        f"[deny-fault] Fault denied for {request.bearing_name} "
        f"by {request.worker_name}. Serving continues."
    )
    return {
        "status":    "denied",
        "bearing":   request.bearing_name,
        "worker":    request.worker_name,
        "denied_at": datetime.now().isoformat(),
        "message":   "Fault denied. Serving pipeline continues uninterrupted.",
    }


@app.post("/bearing/continue", operation_id="continue_to_next_bearing",
          tags=["Bearing Lifecycle"])
def continue_to_next_bearing(
    request: ContinueRequest,
    background_tasks: BackgroundTasks,
):
    """
    Worker clicks 'Continue' after confirming or denying a fault.

    Fix #5: Waits for the serving_lock to clear before stopping processes.
    Fix #2: Serving is stopped only here — not on critical PM status.

    Steps:
    1. Wait for current burst to finish (serving_lock — Fix #5).
    2. Stop SCADA + Serving processes.
    3. Advance the live bearing queue.
    4. Background: extract features + start SCADA + Serving for new bearing.
    """
    from orchestrator import BearingRegistry

    # Wait for burst boundary (Fix #5) then stop
    if _process_manager._current_bearing:
        _process_manager._wait_for_serving_idle(_process_manager._current_bearing)
    _process_manager.stop_all()

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
        run_id = (
            f"bearing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
        background_tasks.add_task(
            _run_bearing_workflow_bg,
            run_id,
            next_b["name"],
            request.realtime,
        )

    return {
        "status":       "advanced",
        "next_bearing": next_b,
        "run_id":       run_id,
        "message": (
            f"Now serving {next_b['name']}. "
            + (f"Background run {run_id} started." if run_id else "")
        ),
    }


async def _run_bearing_workflow_bg(
    run_id:       str,
    bearing_name: str,
    realtime:     bool,
):
    """
    Background task triggered by POST /bearing/continue.

    Starts fresh SCADA + Serving processes for the new bearing.
    There is NO feature extraction here — SCADA handles all of that.
    """
    from orchestrator import WorkflowStateManager

    state_mgr = WorkflowStateManager()
    state_mgr.init_state(run_id, "bearing_continue")

    try:
        logger.info(f"[{run_id}] Starting SCADA + Serving for {bearing_name}...")
        pids = _process_manager.start_bearing(
            bearing_name=bearing_name,
            realtime=realtime,
        )
        logger.info(
            f"[{run_id}] Processes started — "
            f"SCADA PID={pids['scada_pid']}  "
            f"Serving PID={pids['serving_pid']}"
        )
        state_mgr.mark_workflow_complete(run_id)

    except Exception as e:
        state_mgr.mark_workflow_failed(run_id, traceback.format_exc())
        logger.error(f"[{run_id}] Bearing continuation FAILED: {e}", exc_info=True)
        raise


# ====================================================================
# LIVE SERVING  (manual / legacy endpoint — backward compat)
# ====================================================================

@app.post("/bearing/serve", tags=["Bearing Lifecycle"])
async def start_live_serving(
    request: LiveServingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Manually start SCADA + Serving for a named bearing.
    Useful for testing without going through the full workflow trigger.
    """
    run_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(
        _start_serving_bg, run_id, request.bearing_name, request.realtime
    )
    return {
        "run_id":  run_id,
        "status":  "started",
        "bearing": request.bearing_name,
        "message": (
            f"SCADA + Serving starting for {request.bearing_name}. "
            "Check /bearing/processes for status."
        ),
    }


async def _start_serving_bg(run_id: str, bearing_name: str, realtime: bool):
    try:
        _process_manager.start_bearing(bearing_name=bearing_name, realtime=realtime)
        logger.info(f"[{run_id}] SCADA + Serving started for {bearing_name}")
    except Exception as e:
        logger.error(f"[{run_id}] Failed to start serving: {e}", exc_info=True)