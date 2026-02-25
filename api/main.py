"""FastAPI backend for AIOps Agent dashboard."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.routes.metrics import router as metrics_router
from api.routes.agent import router as agent_router
from api.routes.evaluation import router as evaluation_router

app = FastAPI(
    title="AIOps Agent API",
    description="Autonomous AIOps Agent — AIOpsLab-aligned evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
app.include_router(agent_router, prefix="/agent", tags=["agent"])
app.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])


@app.get("/")
async def root():
    return {
        "name": "AIOps Autonomous Agent",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
