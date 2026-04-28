"""
API.py
═══════════════════════════════════════════════════════════════════════════════
PHM 2012 RUL Prediction API

Process lifecycle
─────────────────
The API owns the SCADA simulator and Serving Pipeline subprocesses via the
ProcessManager singleton. This means:

  POST /workflow/trigger
      → trains model (first run) / extracts features (subsequent runs)
      → automatically starts scada_simulator.py + run_serving.py for the
        current live bearing from bearings.json queue

  POST /bearing/continue  (after fault confirm/deny)
      → stops old scada + serving processes
      → advances bearing queue
      → extracts features for new bearing
      → starts scada_simulator.py + run_serving.py for new bearing
      → if confirmed: also starts run_preprod.py in background

  GET /bearing/processes
      → shows current subprocess status
"""

import io
import logging
import os
import sys
import uuid
import subprocess
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
# PROCESS MANAGER  — owns scada + serving subprocesses
# ====================================================================

class ProcessManager:
    """
    Singleton that owns the SCADA simulator and Serving Pipeline subprocesses.

    Ensures only one pair of processes runs at a time. When a new bearing
    is started (via /bearing/continue), it stops the old processes first,
    then starts fresh ones for the new bearing.

    Processes are started with Popen so they run independently of the API
    but are tracked so they can be stopped cleanly.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._scada_proc   = None
            cls._instance._serving_proc = None
            cls._instance._preprod_proc = None
            cls._instance._current_bearing = None
        return cls._instance

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start_bearing(self, bearing_name: str, realtime: bool = False) -> Dict:
        """
        Stop any existing processes and start fresh scada + serving for bearing_name.

        Parameters
        ----------
        bearing_name : str  — e.g. "Bearing1_5"
        realtime     : bool — if True, scada sleeps 10 s between bursts

        Returns dict with process PIDs.
        """
        # Stop existing processes first
        self.stop_all()

        logger.info(f"[ProcessManager] Starting SCADA + Serving for {bearing_name}")

        # Build command for scada_simulator.py
        scada_cmd = [
            sys.executable, "scada_simulator.py",
            "--bearing", bearing_name,
        ]
        if realtime:
            scada_cmd.append("--realtime")

        # Build command for run_serving.py
        serving_cmd = [
            sys.executable, "run_serving.py",
            "--bearing", bearing_name,
        ]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        try:
            self._scada_proc = subprocess.Popen(
                scada_cmd,
                env=env,
                stdout=None,   # prints directly to API terminal
                stderr=None,
            )
            self._serving_proc = subprocess.Popen(
                serving_cmd,
                env=env,
                stdout=None,   # prints directly to API terminal
                stderr=None,
            )
            self._current_bearing = bearing_name

            logger.info(
                f"[ProcessManager] SCADA PID={self._scada_proc.pid}  "
                f"Serving PID={self._serving_proc.pid}"
            )

            return {
                "bearing":      bearing_name,
                "scada_pid":    self._scada_proc.pid,
                "serving_pid":  self._serving_proc.pid,
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
                env=env,
                stdout=None,   # prints directly to API terminal
                stderr=None,
            )
            logger.info(
                f"[ProcessManager] Pre-Production started "
                f"(PID={self._preprod_proc.pid}, run_id={run_id})"
            )
            return self._preprod_proc.pid
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to start run_preprod.py: {e}")
            return None

    def stop_all(self):
        """Gracefully terminate scada and serving processes."""
        for name, proc in [
            ("SCADA",   self._scada_proc),
            ("Serving", self._serving_proc),
        ]:
            if proc and proc.poll() is None:
                logger.info(
                    f"[ProcessManager] Stopping {name} (PID={proc.pid})"
                )
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
        """Return current process status."""
        def _proc_status(proc, name):
            if proc is None:
                return {"name": name, "running": False, "pid": None}
            running = proc.poll() is None
            return {"name": name, "running": running, "pid": proc.pid}

        return {
            "current_bearing": self._current_bearing,
            "scada":           _proc_status(self._scada_proc,   "scada_simulator"),
            "serving":         _proc_status(self._serving_proc, "run_serving"),
            "preprod":         _proc_status(self._preprod_proc, "run_preprod"),
        }


# Module-level singleton
_process_manager = ProcessManager()


# ====================================================================
# REQUEST / RESPONSE MODELS
# ====================================================================

class WorkflowTriggerRequest(BaseModel):
    workflow_name:    str                      = "rul_prediction"
    config_overrides: Optional[Dict[str, Any]] = None
    priority:         str                      = Field(
        default="normal", pattern="^(low|normal|high)$"
    )
    # If True, automatically start SCADA + Serving after workflow completes
    auto_start_serving: bool = Field(
        True,
        description="Auto-start SCADA simulator and Serving pipeline after workflow"
    )
    realtime: bool = Field(
        False,
        description="Run SCADA simulator in realtime mode (10 s between bursts)"
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

# ── Bearing lifecycle models ──────────────────────────────────────────────────

# ── Bearing lifecycle models ──────────────────────────────────────────────────

class FaultConfirmRequest(BaseModel):
    bearing_name:   str   = Field(..., json_schema_extra={"example": "Bearing1_5"})
    run_id:         str   = Field(..., description="Workflow run_id that produced the features")
    rul_at_failure: float = Field(0.0, description="Confirmed RUL at failure (seconds)")
    worker_name:    str   = Field("Unknown", description="Maintenance tech name")
    note:           str   = Field("", description="Optional note")

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


# ── New models for pre-prod trigger and audit flush ───────────────────────────

class PreprodTriggerRequest(BaseModel):
    workflow_name: str = Field("rul_prediction")


class AuditFlushRequest(BaseModel):
    bearing_name: str = Field(..., description="Bearing to flush audit records for")
    limit:        int = Field(500, ge=1, le=10000)


# ====================================================================
# WORKFLOW TRIGGER ENDPOINTS
# ====================================================================

@app.post("/workflow/trigger", response_model=Dict[str, str])
async def trigger_workflow(
    request: WorkflowTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the full workflow.

    First run:  ingest → extract → train → deploy model
    Subsequent: extract only (deployed model already exists)

    After the workflow completes, automatically starts the SCADA simulator
    and Serving Pipeline for the current live bearing (unless
    auto_start_serving=False).
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(
        execute_workflow_async, run_id, request
    )
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

    Runs start_workflow() which automatically detects whether a deployed
    model exists and skips training if so.

    After completion, starts SCADA simulator + Serving Pipeline for the
    current live bearing automatically (if auto_start_serving=True).
    """
    try:
        from orchestrator import WorkflowExecutor, BearingRegistry
        executor = WorkflowExecutor()
        executor.start_workflow(
            workflow_name    = request.workflow_name,
            config_overrides = request.config_overrides,
        )

        # ── Auto-start SCADA + Serving after workflow completes ───────────────
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
                    f"[{run_id}] Workflow complete — no live bearing in queue, "
                    f"SCADA + Serving not started."
                )

    except Exception as e:
        from orchestrator import WorkflowStateManager
        try:
            WorkflowStateManager().update_step_status(
                run_id, "workflow", "FAILED", str(e)
            )
        except Exception:
            pass
        logger.error(f"[{run_id}] Workflow failed: {e}", exc_info=True)
        raise


# ====================================================================
# PRE-PRODUCTION RETRAINING  (Dashboard → AutoTrain dashed arrow)
# ====================================================================

@app.post("/preprod/trigger", tags=["Pre-Production"])
async def trigger_preprod_retraining():
    """
    Dashboard → AutoTrain: manually trigger Pre-Production retraining.

    Starts run_preprod.py in the background. The Serving Pipeline continues
    uninterrupted — model hot-swap happens automatically if the new model
    outperforms the current champion.

    This implements the dashed arrow:
        Dashboard -.Retraining Triggered.-> AutoTrain  (V9 diagram)
    """
    preprod_run_id = (
        f"preprod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:6]}"
    )
    try:
        preprod_pid = _process_manager.start_preprod(preprod_run_id)
        logger.info(
            f"[preprod/trigger] Manual retraining triggered from Dashboard — "
            f"run_id={preprod_run_id}  PID={preprod_pid}"
        )
        return {
            "status":         "started",
            "preprod_run_id": preprod_run_id,
            "preprod_pid":    preprod_pid,
            "message": (
                "Pre-Production retraining started in background. "
                "Serving Pipeline continues uninterrupted. "
                f"Poll /workflow/{preprod_run_id}/status for progress."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# PROCESS STATUS ENDPOINT
# ====================================================================

@app.get("/bearing/processes", tags=["Bearing Lifecycle"])
def get_process_status():
    """
    Return the current status of the SCADA simulator, Serving Pipeline,
    and Pre-Production subprocesses managed by the API.
    """
    return _process_manager.status()


# ====================================================================
# RUL PREDICTION ENDPOINT  (single feature vector, synchronous)
# ====================================================================

@app.post("/predict/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """Predict RUL for a single burst feature vector."""
    try:
        registry    = ModelRegistry()
        model_entry = registry.get_deployed_model("RUL_s")
        if not model_entry:
            raise HTTPException(status_code=404, detail="No deployed model found.")

        from Live_implementation.live_predictor import LivePredictor
        predictor = LivePredictor.from_path(model_entry["model_path"])

        feature_vec = np.array(
            list(request.features.values()), dtype=np.float32
        )
        rul_s   = float(predictor.predict(feature_vec))
        rul_min = rul_s / 60.0

        return RULPredictionResponse(
            predicted_rul_s=rul_s,
            predicted_rul_min=rul_min,
            horizon=10,
            model_version=model_entry["model_id"],
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    return registry.list_models(
        status=status, run_id=run_id, target_feature=target_feature
    )


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
    model    = registry.get_deployed_model(target_feature)
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
    Sets current_live_index to 0 and resets all live bearing statuses
    back to 'available'. Also stops any running processes.
    Does not delete any files from disk.
    """
    from orchestrator import BearingRegistry

    # Stop running processes before reset
    _process_manager.stop_all()

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
    2. Pushes labelled data to MongoDB 'confirmed_faults' (Feature Store Mirrored).
    3. Sets bearing status → 'confirmed'.
    4. Immediately starts run_preprod.py retraining in the background.
       Serving pipeline continues uninterrupted — hot-swap happens automatically
       when retraining completes and new model is better.

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

        # Start retraining immediately — no need to wait for Continue
        preprod_run_id = f"preprod_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        preprod_pid    = _process_manager.start_preprod(preprod_run_id)
        logger.info(
            f"[confirm-fault] Fault confirmed for {request.bearing_name} — "
            f"Pre-Production retraining started immediately "
            f"(PID={preprod_pid}, run_id={preprod_run_id})"
        )

        return {
            "status":          "confirmed",
            "bearing":         request.bearing_name,
            "worker":          request.worker_name,
            "confirmed_at":    datetime.now().isoformat(),
            "store_result":    result,
            "preprod_run_id":  preprod_run_id,
            "preprod_pid":     preprod_pid,
            "message":         (
                f"Fault confirmed. Confirmed fault data pushed to FS Mirrored. "
                f"Retraining started in background (run_id={preprod_run_id}). "
                f"Serving pipeline continues — model will hot-swap if new model is better."
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
    Maintenance tech denies the fault (false positive).
    Sets bearing status → 'denied'. Features NOT pushed to Feature Store.
    Call POST /bearing/continue to advance the queue.
    """
    from orchestrator import BearingRegistry
    reg     = BearingRegistry("config/bearings.json")
    bearing = reg.get_bearing(request.bearing_name)
    if not bearing:
        raise HTTPException(
            status_code=404,
            detail=f"Bearing '{request.bearing_name}' not found."
        )
    reg.set_status(request.bearing_name, "denied")
    return {
        "status":    "denied",
        "bearing":   request.bearing_name,
        "worker":    request.worker_name,
        "denied_at": datetime.now().isoformat(),
    }


@app.post("/bearing/continue", operation_id="continue_to_next_bearing",
          tags=["Bearing Lifecycle"])
def continue_to_next_bearing(
    request: ContinueRequest,
    background_tasks: BackgroundTasks,
):
    """
    Tech clicks 'Continue' after confirming or denying a fault.

    1. Stops current SCADA simulator + Serving Pipeline processes.
    2. Advances the live bearing queue to the next bearing.
    3. Fires a background task that extracts features for the new bearing
       then starts fresh SCADA + Serving processes for it.

    Note: retraining is triggered immediately on fault confirmation
    (POST /bearing/confirm-fault), not here.
    """
    from orchestrator import BearingRegistry

    # Stop current processes before advancing
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

    1. Extracts features for the new bearing
    2. Starts SCADA simulator + Serving Pipeline subprocesses

    Note: retraining (run_preprod.py) is triggered immediately on
    POST /bearing/confirm-fault, not here.
    """
    from orchestrator import WorkflowExecutor, WorkflowStateManager, BearingRegistry

    state_mgr = WorkflowStateManager()
    state_mgr.init_state(run_id, "bearing_continue")

    try:
        executor = WorkflowExecutor()
        reg      = BearingRegistry("config/bearings.json")
        bearing  = reg.get_bearing(bearing_name)

        if not bearing:
            state_mgr.mark_workflow_failed(run_id, f"Bearing '{bearing_name}' not found.")
            return

        # 1. Extract features for new bearing
        logger.info(f"[{run_id}] Extracting features for {bearing_name}...")
        executor._run_extraction(run_id, bearing)

        # 2. Start SCADA simulator + Serving Pipeline
        logger.info(f"[{run_id}] Starting SCADA + Serving for {bearing_name}...")
        pids = _process_manager.start_bearing(
            bearing_name=bearing_name,
            realtime=realtime,
        )
        logger.info(
            f"[{run_id}] Processes started: "
            f"SCADA PID={pids['scada_pid']}  "
            f"Serving PID={pids['serving_pid']}"
        )

        state_mgr.mark_workflow_complete(run_id)
        logger.info(f"[{run_id}] Bearing continuation complete.")

    except Exception as e:
        state_mgr.mark_workflow_failed(run_id, traceback.format_exc())
        logger.error(f"[{run_id}] Bearing continuation FAILED: {e}", exc_info=True)
        raise


# ====================================================================
# LIVE SERVING (legacy direct endpoint — kept for backward compat)
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
            f"SCADA simulator and Serving Pipeline starting for "
            f"{request.bearing_name}. Check /bearing/processes for status."
        ),
    }


async def _start_serving_bg(run_id: str, bearing_name: str, realtime: bool):
    try:
        _process_manager.start_bearing(
            bearing_name=bearing_name,
            realtime=realtime,
        )
        logger.info(f"[{run_id}] SCADA + Serving started for {bearing_name}")
    except Exception as e:
        logger.error(f"[{run_id}] Failed to start serving: {e}", exc_info=True)


# ====================================================================
# AUDIT SERVICE — batch flush (AuditSvc → External Data Destination)
# ====================================================================

@app.post("/audit/flush", tags=["Audit & Export"])
def flush_audit_records(request: AuditFlushRequest):
    """
    AuditSvc → External: batch-flush Serving History records to the
    external CSV data destination for a specific bearing.

    Useful when the pipeline was running without Export Service enabled,
    or to re-export historical data from the dashboard.
    """
    try:
        from utils.audit_service import AuditService
        auditor  = AuditService()
        exported = auditor.flush_bearing(
            bearing_name=request.bearing_name,
            limit=request.limit,
        )
        return {
            "status":       "ok",
            "bearing":      request.bearing_name,
            "exported":     exported,
            "export_paths": auditor.get_export_paths(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# EXPORT SERVICE — path info (ExportSvc → External Data Destination)
# ====================================================================

@app.get("/export/paths", tags=["Audit & Export"])
def get_export_paths():
    """
    Return the configured external data destination paths for the Export Service.
    Shown in the Dashboard → Retraining & Export Control page.
    """
    try:
        from export_service.export_service import get_exporter
        exporter = get_exporter()
        return exporter.get_export_paths()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))