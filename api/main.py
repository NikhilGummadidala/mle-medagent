"""
FastAPI application entry point.

Run with:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Then open: http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.graph_runner import llm_available
from api.routes import auth_router, router

app = FastAPI(
    title="MedAgent API",
    description="Multi-agent medical assistant powered by LangGraph",
    version="0.2.0",
)

# The frontend is served from this same origin, so no CORS is needed by
# default. To call the API from another origin, list it explicitly:
#   MEDAGENT_CORS_ORIGINS=https://app.example.com,https://staging.example.com
_origins = [o.strip() for o in os.environ.get("MEDAGENT_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

app.include_router(router, prefix="/api")
app.include_router(auth_router, prefix="/api/auth")

# Serve the frontend (medichat/) as static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "medichat"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "medagent", "llm_available": llm_available()}


@app.get("/")
def serve_frontend() -> FileResponse:
    """Serve the main frontend page."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))
