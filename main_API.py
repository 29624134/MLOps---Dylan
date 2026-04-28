"""
main_API.py
═══════════════════════════════════════════════════════════════════════════════
PHM MLOps System Entry Point

Fix #4 — Only two GUIs are launched:
    dashboard/fault_review.py  →  http://localhost:8501
        Maintenance Worker: Confirm / Deny faults
    dashboard/rul_monitor.py   →  http://localhost:8502
        RUL Predictions: Live bearing health & alert log

The old multi-page dashboard/app.py is no longer started.

Usage
─────
    python main_API.py

Stops with Ctrl+C — all subprocesses are terminated cleanly.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
import threading
import signal

# ── Force UTF-8 across the board before anything else ─────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn

# ── Shared env for all subprocesses ───────────────────────────────────────────
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# ── Script paths — Streamlit apps live in the dashboard/ folder ───────────────
_FAULT_REVIEW_SCRIPT = os.path.join("dashboard", "fault_review.py")
_RUL_MONITOR_SCRIPT  = os.path.join("dashboard", "rul_monitor.py")

# ── Track GUI subprocesses so we can terminate them on exit ───────────────────
_gui_procs: list[subprocess.Popen] = []


def _run_streamlit(script: str, port: int) -> None:
    """
    Launch a standalone Streamlit GUI as a subprocess and keep it running.
    Each GUI is its own independent Streamlit app — NOT imported as a module.
    """
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", script,
            "--server.port",              str(port),
            "--server.headless",          "true",
            "--browser.gatherUsageStats", "false",
        ],
        env=_ENV,
    )
    _gui_procs.append(proc)
    proc.wait()   # block this thread until the GUI process exits


def _stop_all_guis(*_) -> None:
    """Terminate all GUI subprocesses cleanly on Ctrl+C / SIGTERM."""
    for proc in _gui_procs:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    # ── Verify GUI scripts exist before launching ──────────────────────────────
    for _path in (_FAULT_REVIEW_SCRIPT, _RUL_MONITOR_SCRIPT):
        if not os.path.exists(_path):
            print(f"[WARNING] GUI script not found: {_path}")

    # ── Register cleanup handlers ──────────────────────────────────────────────
    signal.signal(signal.SIGINT,  _stop_all_guis)
    signal.signal(signal.SIGTERM, _stop_all_guis)

    # ── GUI 1 — Fault Review (Maintenance Worker) ──────────────────────────────
    fault_review_thread = threading.Thread(
        target=_run_streamlit,
        args=(_FAULT_REVIEW_SCRIPT, 8501),
        daemon=True,
        name="GUI-FaultReview",
    )
    fault_review_thread.start()

    # ── GUI 2 — RUL Monitor ────────────────────────────────────────────────────
    rul_monitor_thread = threading.Thread(
        target=_run_streamlit,
        args=(_RUL_MONITOR_SCRIPT, 8502),
        daemon=True,
        name="GUI-RULMonitor",
    )
    rul_monitor_thread.start()

    # ── Startup banner ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ⚙️   PHM MLOps System Starting")
    print("=" * 60)
    print("  API docs       →  http://localhost:8000/docs")
    print("  Fault Review   →  http://localhost:8501   (Maintenance Worker)")
    print("  RUL Monitor    →  http://localhost:8502   (Predictions)")
    print("=" * 60)
    print()

    # ── FastAPI (blocking — keeps the process alive) ───────────────────────────
    uvicorn.run(
        "API:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True conflicts with subprocess management
    )