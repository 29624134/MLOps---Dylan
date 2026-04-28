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


def run_streamlit():
    """Launch the Streamlit dashboard as a subprocess."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", "8501",
         "--server.headless", "true"],   # suppress the browser auto-open from Streamlit
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

if __name__ == "__main__":
    # Start Streamlit in a background thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()

    print("=" * 55)
    print("  ⚙️  PHM MLOps System Starting")
    print("=" * 55)
    print("  API docs  →  http://localhost:8000/docs")
    print("  Dashboard →  http://localhost:8501")
    print("=" * 55)

    # Run FastAPI (blocking — keeps the process alive)
    uvicorn.run(
        "API:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )