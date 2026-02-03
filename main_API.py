import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "API:app",              # <--- FILE:VARIABLE  (API.py contains app)
        host="0.0.0.0",
        port=8000,
        reload=True             # auto-reload for development
    )
