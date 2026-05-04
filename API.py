"""
API.py
═══════════════════════════════════════════════════════════════════════════════
PHM 2012 RUL Prediction API — group-aware edition.

Process lifecycle
─────────────────
The API owns the SCADA simulator and Serving Pipeline subprocesses via the
ProcessManager singleton.

  POST /workflow/trigger
      → trains all 3 group models in parallel (first run)
      → auto-starts scada_simulator.py + run_serving.py for each group's
        current live bearing simultaneously

  POST /bearing/confirm-fault
      → pushes confirmed fault data to FS Mirrored (tagged with group)
      → starts run_preprod.py --group N for ONLY the affected group
      → other groups keep serving uninterrupted

  POST /bearing/continue
      → stops all current processes
      → restarts SCADA + Serving for all groups with updated bearings

  GET /bearing/processes
      → shows all subprocess statuses (SCADA + one serving per bearing)

Champion files (one per group):
    model_registry/champion_bearing1.json
    model_registry/champion_bearing2.json
    model_registry/champion_bearing3.json
═══════════════════════════════════════════════════════════════════════════════
"""

import io
import logging
import os
import re
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

from serving_pipeline.serving_pipeline_routes import router as serving_router
app.include_router(serving_router)

MODEL_REGISTRY_DIR = "model_registry"


def _group_from_bearing(bearing_name: str) -> Optional[str]:
    """Infer group ID from bearing name. e.g. 'Bearing2_4' → '2'."""
    m = re.match(r"Bearing(\d+)_\d+", bearing_name, re.IGNORECASE)
    return m.group(1) if m else None


def _champion_path(group: str) -> str:
    return os.path.join(MODEL_REGISTRY_DIR, f"champion_bearing{group}.json")


# ====================================================================
# PROCESS MANAGER  — owns scada + serving subprocesses
# ====================================================================

class ProcessManager:
    """
    Singleton that owns the SCADA simulator and Serving Pipeline subprocesses.

    Group-aware design
    ──────────────────
    - One scada_simulator.py handles ALL groups simultaneously (--bearing B1 B2 B3)
    - One run_serving.py per bearing, each using its group's champion file
    - One run_preprod.py per fault confirmation (group-scoped, --group N)

    start_bearing(bearing_names) accepts the current live bearing from each group.
    stop_all() terminates SCADA + all serving processes.
    start_preprod(run_id, group) launches group-scoped retraining only.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._scada_proc       = None
            cls._instance._serving_procs    = []
            cls._instance._preprod_procs    = {}   # group → proc (one per group)
            cls._instance._current_bearings = []
        return cls._instance

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start_bearing(
        self,
        bearing_names: List[str],
        realtime: bool = False,
    ) -> Dict:
        """
        Stop existing processes and start fresh SCADA + Serving for the
        given bearings (one per group, 1–3 total).

        Each bearing gets its own run_serving.py instance. The champion file
        is inferred from the bearing's group (Bearing1_x → champion_bearing1.json).

        Parameters
        ----------
        bearing_names : list of str — one live bearing per group (1–3)
        realtime      : bool — if True, SCADA sleeps burst_period between bursts
        """
        if isinstance(bearing_names, str):
            bearing_names = [bearing_names]

        self.stop_all()

        logger.info(f"[ProcessManager] Starting SCADA + Serving for {bearing_names}")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        # ── One SCADA simulator for all bearings (lock-step streaming) ────────
        scada_cmd = [sys.executable, "scada_simulator.py", "--bearing", *bearing_names]
        if realtime:
            scada_cmd.append("--realtime")

        try:
            self._scada_proc = subprocess.Popen(scada_cmd, env=env)
            logger.info(
                f"[ProcessManager] SCADA PID={self._scada_proc.pid}  "
                f"bearings={bearing_names}"
            )
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to start SCADA: {e}")
            self.stop_all()
            raise RuntimeError(f"Could not start SCADA process: {e}")

        # ── One run_serving.py per bearing, each pointing at its group champion ─
        self._serving_procs = []
        for bearing_name in bearing_names:
            group = _group_from_bearing(bearing_name)
            champ = _champion_path(group) if group else os.path.join(
                MODEL_REGISTRY_DIR, "champion.json"
            )
            serving_cmd = [
                sys.executable, "run_serving.py",
                "--bearing",  bearing_name,
                "--champion", champ,
            ]
            try:
                proc = subprocess.Popen(serving_cmd, env=env)
                self._serving_procs.append((bearing_name, proc))
                logger.info(
                    f"[ProcessManager] Serving PID={proc.pid}  "
                    f"bearing={bearing_name}  champion={champ}"
                )
            except Exception as e:
                logger.error(
                    f"[ProcessManager] Failed to start serving for {bearing_name}: {e}"
                )
                self.stop_all()
                raise RuntimeError(
                    f"Could not start serving for {bearing_name}: {e}"
                )

        self._current_bearings = list(bearing_names)

        return {
            "bearings":     bearing_names,
            "scada_pid":    self._scada_proc.pid,
            "serving_pids": {
                name: proc.pid for name, proc in self._serving_procs
            },
        }

    def start_preprod(self, run_id: str, group: str) -> Optional[int]:
        """
        Launch run_preprod.py --group N as a background subprocess.

        Each group has its own independent preprod process slot — confirming
        Group 2 never interrupts Group 1's retraining, and vice versa.
        If a previous preprod for the SAME group is still running it is
        terminated first (e.g. a second fault confirmation on the same group).

        Parameters
        ----------
        run_id : str — unique run identifier
        group  : str — "1", "2", or "3"
        """
        # Only terminate the previous process for THIS group
        existing = self._preprod_procs.get(group)
        if existing and existing.poll() is None:
            logger.info(
                f"[ProcessManager] Terminating previous Group {group} preprod "
                f"(PID={existing.pid})"
            )
            existing.terminate()

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.Popen(
                [
                    sys.executable, "run_preprod.py",
                    "--run_id", run_id,
                    "--group",  group,
                ],
                env=env,
            )
            self._preprod_procs[group] = proc
            logger.info(
                f"[ProcessManager] Pre-Production started — "
                f"Group {group}  PID={proc.pid}  run_id={run_id}"
            )
            return proc.pid
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to start run_preprod.py: {e}")
            return None

    def restart_group(
        self,
        group:        str,
        new_bearing:  str,
        all_bearings: list,
        realtime:     bool = False,
    ) -> Dict:
        """
        Restart ONLY the serving process for one group's new bearing,
        while keeping all other groups' serving processes running.

        Since SCADA streams all bearings in one process, we must:
          1. Stop the SCADA process (affects all groups temporarily)
          2. Stop only the old serving process for this group
          3. Restart SCADA with the full updated bearing list
          4. Start a new serving process for the new bearing only

        Other groups' serving processes are NOT touched — they idle briefly
        while SCADA restarts (their Feature Store poll loop just waits),
        then resume normally once SCADA is back up.

        Parameters
        ----------
        group        : str  — group ID being advanced ("1", "2", or "3")
        new_bearing  : str  — the new bearing name for this group
        all_bearings : list — full list of current live bearings across ALL groups
        realtime     : bool — realtime mode for SCADA
        """
        logger.info(
            f"[ProcessManager] Restarting Group {group} → {new_bearing}. "
            f"Other groups keep serving."
        )

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        # ── Stop SCADA (needed to update bearing list) ────────────────────────
        if self._scada_proc and self._scada_proc.poll() is None:
            logger.info(f"[ProcessManager] Stopping SCADA (PID={self._scada_proc.pid})")
            self._scada_proc.terminate()
            try:
                self._scada_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._scada_proc.kill()
        self._scada_proc = None

        # ── Stop only the serving process for this group ──────────────────────
        remaining = []
        for bearing_name, proc in self._serving_procs:
            grp = _group_from_bearing(bearing_name)
            if grp == group:
                if proc and proc.poll() is None:
                    logger.info(
                        f"[ProcessManager] Stopping Serving [{bearing_name}] "
                        f"(PID={proc.pid})"
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                # Do not re-add this group's old bearing
            else:
                remaining.append((bearing_name, proc))  # keep other groups

        self._serving_procs    = remaining
        self._current_bearings = [b for b in self._current_bearings
                                  if _group_from_bearing(b) != group]

        # ── Restart SCADA with the full updated bearing list ──────────────────
        scada_cmd = [sys.executable, "scada_simulator.py",
                     "--bearing", *all_bearings]
        if realtime:
            scada_cmd.append("--realtime")

        try:
            self._scada_proc = subprocess.Popen(scada_cmd, env=env)
            logger.info(
                f"[ProcessManager] SCADA restarted PID={self._scada_proc.pid}  "
                f"bearings={all_bearings}"
            )
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to restart SCADA: {e}")
            raise RuntimeError(f"Could not restart SCADA: {e}")

        # ── Start new serving process for the new bearing only ────────────────
        champ       = _champion_path(group)
        serving_cmd = [
            sys.executable, "run_serving.py",
            "--bearing",  new_bearing,
            "--champion", champ,
        ]
        try:
            proc = subprocess.Popen(serving_cmd, env=env)
            self._serving_procs.append((new_bearing, proc))
            self._current_bearings.append(new_bearing)
            logger.info(
                f"[ProcessManager] Serving started PID={proc.pid}  "
                f"bearing={new_bearing}  champion={champ}"
            )
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to start serving for {new_bearing}: {e}")
            raise RuntimeError(f"Could not start serving for {new_bearing}: {e}")

        return {
            "group":         group,
            "new_bearing":   new_bearing,
            "scada_pid":     self._scada_proc.pid,
            "serving_pid":   proc.pid,
            "all_bearings":  all_bearings,
        }
        """Gracefully terminate SCADA and all serving processes."""
        if self._scada_proc and self._scada_proc.poll() is None:
            logger.info(f"[ProcessManager] Stopping SCADA (PID={self._scada_proc.pid})")
            self._scada_proc.terminate()
            try:
                self._scada_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._scada_proc.kill()

        for bearing_name, proc in self._serving_procs:
            if proc and proc.poll() is None:
                logger.info(
                    f"[ProcessManager] Stopping Serving [{bearing_name}] "
                    f"(PID={proc.pid})"
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        self._scada_proc       = None
        self._serving_procs    = []
        self._current_bearings = []
        # Note: preprod processes are NOT stopped here — retraining continues
        # uninterrupted even when SCADA + Serving restarts for a new bearing.

    def stop_all(self):
        """Gracefully terminate SCADA and all serving processes."""
        if self._scada_proc and self._scada_proc.poll() is None:
            logger.info(f"[ProcessManager] Stopping SCADA (PID={self._scada_proc.pid})")
            self._scada_proc.terminate()
            try:
                self._scada_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._scada_proc.kill()

        for bearing_name, proc in self._serving_procs:
            if proc and proc.poll() is None:
                logger.info(
                    f"[ProcessManager] Stopping Serving [{bearing_name}] "
                    f"(PID={proc.pid})"
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        self._scada_proc       = None
        self._serving_procs    = []
        self._current_bearings = []
        # Note: preprod processes are NOT stopped — retraining continues
        # uninterrupted even when SCADA + Serving restarts.

    def status(self) -> Dict:
        def _ps(proc, name):
            if proc is None:
                return {"name": name, "running": False, "pid": None}
            return {"name": name, "running": proc.poll() is None, "pid": proc.pid}

        return {
            "current_bearings": self._current_bearings,
            "scada":            _ps(self._scada_proc, "scada_simulator"),
            "serving":          {
                name: _ps(proc, f"run_serving [{name}]")
                for name, proc in self._serving_procs
            },
            "preprod": {
                group: _ps(proc, f"run_preprod [group {group}]")
                for group, proc in self._preprod_procs.items()
            },
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
        description="Auto-start SCADA + Serving for all group live bearings after training"
    )
    realtime: bool = Field(False, description="Run SCADA simulator in realtime mode")

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
    bearing_names: List[str] = Field(
        ...,
        description="One live bearing per group (1–3 total)",
        min_items=1,
        max_items=3,
    )
    realtime: bool = False

class RegisterWorkflowRequest(BaseModel):
    workflow_name: str
    version:       str
    definition:    Dict[str, Any]
    trigger:       Optional[Dict[str, Any]] = None
    git_hash:      Optional[str]            = None
    environment:   Optional[Dict[str, str]] = None
    metadata:      Optional[Dict[str, Any]] = None

class FaultConfirmRequest(BaseModel):
    bearing_name:   str           = Field(..., description="Bearing being confirmed")
    run_id:         Optional[str] = Field(None)
    rul_at_failure: float         = Field(0.0, description="Confirmed RUL at failure (s)")
    worker_name:    str           = Field("Unknown")
    note:           str           = Field("")

class FaultDenyRequest(BaseModel):
    bearing_name: str = Field(..., json_schema_extra={"example": "Bearing1_5"})
    worker_name:  str = Field("Unknown")
    note:         str = Field("")

class ContinueRequest(BaseModel):
    bearing_name:    str  = Field(..., description="The bearing that was just confirmed or denied")
    worker_name:     str  = Field("Unknown")
    trigger_new_run: bool = Field(True)
    realtime:        bool = Field(False)

class AuditFlushRequest(BaseModel):
    bearing_name: str = Field(...)
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

    First run: trains all 3 group models in parallel, then auto-starts
    SCADA + Serving for each group's current live bearing.

    Subsequent runs: all training skipped (champions exist), serving
    is started directly.
    """
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

    Trains all 3 group models in parallel (first run only), then
    auto-starts SCADA + Serving for all groups' current live bearings.
    """
    try:
        from orchestrator import WorkflowExecutor, BearingRegistry
        executor = WorkflowExecutor()
        executor.start_workflow(
            workflow_name    = request.workflow_name,
            config_overrides = request.config_overrides,
        )

        if request.auto_start_serving:
            reg = BearingRegistry("config/bearings.json")
            live_bearings = []
            for group in reg.all_groups():
                bearing = reg.current_live_bearing(group)
                if bearing:
                    live_bearings.append(bearing["name"])
                    logger.info(
                        f"[{run_id}] Group {group}: serving {bearing['name']}"
                    )
                else:
                    logger.warning(
                        f"[{run_id}] Group {group}: queue exhausted — skipping."
                    )

            if live_bearings:
                _process_manager.start_bearing(
                    bearing_names=live_bearings,
                    realtime=request.realtime,
                )
            else:
                logger.warning(f"[{run_id}] No live bearings found in any group.")

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
# PRE-PRODUCTION RETRAINING
# ====================================================================

@app.post("/preprod/trigger", tags=["Pre-Production"])
async def trigger_preprod_retraining(group: str):
    """
    Manually trigger Pre-Production retraining for a specific group.
    Only the specified group's model is retrained — other groups keep
    serving uninterrupted.
    """
    if group not in ("1", "2", "3"):
        raise HTTPException(status_code=400, detail="group must be '1', '2', or '3'")

    preprod_run_id = (
        f"preprod_g{group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:6]}"
    )
    try:
        preprod_pid = _process_manager.start_preprod(preprod_run_id, group=group)
        return {
            "status":         "started",
            "group":          group,
            "preprod_run_id": preprod_run_id,
            "preprod_pid":    preprod_pid,
            "message": (
                f"Group {group} retraining started. "
                f"Other groups continue serving uninterrupted."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# PROCESS STATUS
# ====================================================================

@app.get("/bearing/processes", tags=["Bearing Lifecycle"])
def get_process_status():
    """Current status of SCADA, all Serving pipelines, and Pre-Production."""
    return _process_manager.status()


# ====================================================================
# RUL PREDICTION (single feature vector, synchronous)
# ====================================================================

@app.post("/predict/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    try:
        registry    = ModelRegistry()
        model_entry = registry.get_deployed_model("RUL_s")
        if not model_entry:
            raise HTTPException(status_code=404, detail="No deployed model found.")

        from Live_implementation.live_predictor import LivePredictor
        predictor   = LivePredictor.from_path(model_entry["model_path"])
        feature_vec = np.array(list(request.features.values()), dtype=np.float32)
        rul_s       = float(predictor.predict(feature_vec))

        return RULPredictionResponse(
            predicted_rul_s=rul_s,
            predicted_rul_min=rul_s / 60.0,
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
    """Return the current live bearing for each group."""
    from orchestrator import BearingRegistry
    reg = BearingRegistry("config/bearings.json")
    result = {}
    for group in reg.all_groups():
        b = reg.current_live_bearing(group)
        result[f"group_{group}"] = b["name"] if b else None
    return result


@app.get("/bearing/queue", operation_id="get_bearing_queue",
         tags=["Bearing Lifecycle"])
def get_bearing_queue():
    """Show all group queues and their current positions."""
    from orchestrator import BearingRegistry
    reg = BearingRegistry("config/bearings.json")
    result = {}
    for group in reg.all_groups():
        grp_cfg = reg.groups[group]
        queue   = grp_cfg.get("live_bearing_queue", [])
        idx     = grp_cfg.get("current_live_index", 0)
        result[f"group_{group}"] = {
            "queue":         queue,
            "current_index": idx,
            "current":       queue[idx] if idx < len(queue) else None,
            "remaining":     len(queue) - idx,
        }
    return result


@app.post("/bearing/reset-queue", operation_id="reset_bearing_queue",
          tags=["Bearing Lifecycle"])
def reset_bearing_queue():
    """
    Reset all group queues back to index 0.
    Stops all running processes. Resets all live bearing statuses to 'available'.
    """
    from orchestrator import BearingRegistry
    _process_manager.stop_all()
    reg = BearingRegistry("config/bearings.json")

    reset_bearings = []
    for b in reg.live_bearings():
        if b.get("status") != "available":
            reg.set_status(b["name"], "available")
            reset_bearings.append(b["name"])

    for group in reg.all_groups():
        reg.groups[group]["current_live_index"] = 0
    reg._save()

    logger.info(f"[reset-queue] All group queues reset. Bearings reset: {reset_bearings}")
    return {
        "status":         "reset",
        "bearings_reset": reset_bearings,
    }


@app.post("/bearing/confirm-fault", operation_id="confirm_bearing_fault",
          tags=["Bearing Lifecycle"])
def confirm_fault(request: FaultConfirmRequest):
    """
    Maintenance tech confirms a fault.

    1. Re-labels features.csv with confirmed RUL values.
    2. Pushes labelled data to MongoDB feature_store_mirrored (tagged with group).
    3. Sets bearing status → 'confirmed'.
    4. Starts run_preprod.py --group N for ONLY the affected group.
       Other groups' models and serving pipelines are completely unaffected.

    Call POST /bearing/continue afterwards to advance this group's queue.
    """
    from orchestrator import WorkflowExecutor
    try:
        executor = WorkflowExecutor()
        result   = executor.confirm_fault_and_push_to_store(
            bearing_name   = request.bearing_name,
            run_id         = request.run_id,
            rul_at_failure = request.rul_at_failure,
        )

        group = result.get("group")
        if not group:
            raise ValueError(
                f"Could not determine group for bearing {request.bearing_name}"
            )

        preprod_run_id = (
            f"preprod_g{group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
        preprod_pid = _process_manager.start_preprod(preprod_run_id, group=group)
        logger.info(
            f"[confirm-fault] {request.bearing_name} confirmed (Group {group}) — "
            f"retraining started (PID={preprod_pid}, run_id={preprod_run_id})"
        )

        return {
            "status":          "confirmed",
            "bearing":         request.bearing_name,
            "group":           group,
            "worker":          request.worker_name,
            "confirmed_at":    datetime.now().isoformat(),
            "store_result":    result,
            "preprod_run_id":  preprod_run_id,
            "preprod_pid":     preprod_pid,
            "message": (
                f"Fault confirmed. Group {group} data pushed to FS Mirrored. "
                f"Group {group} retraining started (run_id={preprod_run_id}). "
                f"Other groups continue serving uninterrupted. "
                f"Model will hot-swap if new model is better."
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
    Sets bearing status → 'denied'. No retraining triggered.
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
        "group":     bearing.get("group"),
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

    1. Determines which group the bearing belongs to.
    2. Advances ONLY that group's queue index.
    3. Restarts ONLY that group's serving process + SCADA with updated list.
       All other groups' serving processes keep running uninterrupted.
    """
    from orchestrator import BearingRegistry

    reg   = BearingRegistry("config/bearings.json")
    group = reg.group_of(request.bearing_name)

    if not group:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot determine group for bearing '{request.bearing_name}'."
        )

    # Advance only this group's queue
    next_b = reg.advance_live_bearing(group)
    logger.info(
        f"[continue] Group {group} advanced — "
        f"next bearing: {next_b['name'] if next_b else 'queue exhausted'}"
    )

    run_id = None
    if request.trigger_new_run and next_b:
        run_id = (
            f"bearing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
        background_tasks.add_task(
            _restart_group_bg,
            run_id,
            group,
            next_b["name"],
            request.realtime,
        )
    elif not next_b:
        # Queue exhausted for this group — just stop its processes
        background_tasks.add_task(
            _stop_group_bg, group
        )

    return {
        "status":         "continued",
        "group_advanced": group,
        "next_bearing":   next_b["name"] if next_b else None,
        "run_id":         run_id,
        "message": (
            f"Group {group} advanced to "
            f"{'queue exhausted' if not next_b else next_b['name']}. "
            f"Other groups continue uninterrupted."
            + (f" Background run {run_id} started." if run_id else "")
        ),
    }


async def _restart_group_bg(
    run_id:      str,
    group:       str,
    new_bearing: str,
    realtime:    bool,
):
    """
    Restart only one group's serving process + SCADA.
    Other groups keep running untouched.
    """
    from orchestrator import BearingRegistry
    reg = BearingRegistry("config/bearings.json")

    # Build the full updated bearing list across all groups
    all_bearings = []
    for g in reg.all_groups():
        b = reg.current_live_bearing(g)
        if b:
            all_bearings.append(b["name"])

    try:
        result = _process_manager.restart_group(
            group        = group,
            new_bearing  = new_bearing,
            all_bearings = all_bearings,
            realtime     = realtime,
        )
        logger.info(
            f"[{run_id}] Group {group} restarted → {new_bearing}. "
            f"SCADA now serving: {all_bearings}"
        )
    except Exception as e:
        logger.error(f"[{run_id}] Failed to restart Group {group}: {e}", exc_info=True)


async def _stop_group_bg(group: str):
    """Stop only the serving process for an exhausted group."""
    remaining_serving = []
    for bearing_name, proc in _process_manager._serving_procs:
        if _group_from_bearing(bearing_name) == group:
            if proc and proc.poll() is None:
                proc.terminate()
                logger.info(
                    f"[ProcessManager] Group {group} queue exhausted — "
                    f"serving stopped for {bearing_name}."
                )
        else:
            remaining_serving.append((bearing_name, proc))
    _process_manager._serving_procs    = remaining_serving
    _process_manager._current_bearings = [
        b for b in _process_manager._current_bearings
        if _group_from_bearing(b) != group
    ]


async def _run_all_groups_bg(run_id: str, realtime: bool):
    """Collect current live bearing from each group and restart everything.
    Used only on initial startup / full reset."""
    from orchestrator import BearingRegistry
    reg = BearingRegistry("config/bearings.json")
    live_bearings = []
    for group in reg.all_groups():
        b = reg.current_live_bearing(group)
        if b:
            live_bearings.append(b["name"])

    if live_bearings:
        try:
            _process_manager.start_bearing(
                bearing_names=live_bearings,
                realtime=realtime,
            )
            logger.info(f"[{run_id}] Started serving for {live_bearings}")
        except Exception as e:
            logger.error(f"[{run_id}] Failed to start serving: {e}", exc_info=True)
    else:
        logger.info(f"[{run_id}] All group queues exhausted.")


# ====================================================================
# LIVE SERVING — manual start for 1–3 bearings
# ====================================================================

@app.post("/bearing/serve", tags=["Bearing Lifecycle"])
async def start_live_serving(
    request: LiveServingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Manually start SCADA + Serving for 1–3 bearings (one per group).
    Each bearing automatically uses its group-specific champion file.
    """
    run_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(
        _start_serving_bg, run_id, request.bearing_names, request.realtime
    )
    return {
        "run_id":   run_id,
        "status":   "started",
        "bearings": request.bearing_names,
        "message": (
            f"SCADA + Serving starting for {request.bearing_names}. "
            f"Check /bearing/processes for status."
        ),
    }


async def _start_serving_bg(run_id: str, bearing_names: List[str], realtime: bool):
    try:
        _process_manager.start_bearing(bearing_names=bearing_names, realtime=realtime)
        logger.info(f"[{run_id}] SCADA + Serving started for {bearing_names}")
    except Exception as e:
        logger.error(f"[{run_id}] Failed to start serving: {e}", exc_info=True)


# ====================================================================
# AUDIT SERVICE
# ====================================================================

@app.post("/audit/flush", tags=["Audit & Export"])
def flush_audit_records(request: AuditFlushRequest):
    """Batch-flush Serving History records to the external CSV destination."""
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
# EXPORT SERVICE
# ====================================================================

@app.get("/export/paths", tags=["Audit & Export"])
def get_export_paths():
    try:
        from utils.export_service import get_exporter
        exporter = get_exporter()
        return exporter.get_export_paths()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))