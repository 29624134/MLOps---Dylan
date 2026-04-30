import os
import sys
import subprocess
import threading
import time
import webbrowser

# Force UTF-8 encoding on stdout/stderr before uvicorn spawns any subprocesses.
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn


def _streamlit(script: str, port: int) -> None:
    """Launch a Streamlit app as a subprocess on the given port."""
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", script,
            "--server.port",     str(port),
            "--server.headless", "true",   # suppress Streamlit's own browser pop-up
        ],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


if __name__ == "__main__":
    # ── Start both Streamlit apps in background threads ───────────────────────
    apps = [
        ("dashboard/rul.py",          8501),   # RUL monitor
        ("dashboard/fault_review.py", 8502),   # fault review
    ]

    for script, port in apps:
        t = threading.Thread(target=_streamlit, args=(script, port), daemon=True)
        t.start()

    print("=" * 55)
    print("  ⚙️  PHM MLOps System Starting")
    print("=" * 55)
    print("  API docs      →  http://localhost:8000/docs")
    print("  RUL Monitor   →  http://localhost:8501")
    print("  Fault Review  →  http://localhost:8502")
    print("=" * 55)

    # ── FastAPI (blocking — keeps the process alive) ──────────────────────────
    uvicorn.run(
        "API:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )