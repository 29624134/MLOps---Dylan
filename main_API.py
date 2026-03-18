import os
import sys

# Force UTF-8 encoding on stdout/stderr before uvicorn spawns any subprocesses.
# This prevents UnicodeEncodeError on Windows (CP1252) when logging Unicode characters.
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "API:app",              # <--- FILE:VARIABLE  (API.py contains app)
        host="0.0.0.0",
        port=8000,
        reload=True             # auto-reload for development
    )