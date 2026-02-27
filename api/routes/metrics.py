"""Metrics API routes — uses the shared simulation state."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.shared_state import run_registry, state
from simulator.service_topology import build_topology

router = APIRouter()


def _get_session(run_id: str | None):
    """Return session for run_id, or default session when omitted."""
    if run_id is None:
        return state
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return session


@router.get("/current")
async def get_current_metrics(run_id: str | None = None):
    """Return current metrics. Optional run_id for sessionized runs."""
    session = _get_session(run_id)
    orch = session.orchestrator
    if orch is None or orch.env.metrics_data is None:
        return {}
    return orch.env.get_current_metrics()


@router.get("/history/{service}")
async def get_metrics_history(service: str, lookback: int = 60, run_id: str | None = None):
    """Return metrics history for a service. Optional run_id for sessionized runs."""
    session = _get_session(run_id)
    orch = session.orchestrator
    if orch is None or orch.env.metrics_data is None:
        return {"error": "No scenario running."}
    try:
        df = orch.env.get_metrics_history(service, lookback_steps=lookback)
        return {
            "service": service,
            "timestamps": [str(t) for t in df.index],
            "metrics": df.to_dict(orient="list"),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/topology")
async def get_topology():
    graph = build_topology()
    nodes = [{"id": n, **graph.nodes[n]} for n in graph.nodes]
    edges = [{"source": u, "target": v} for u, v in graph.edges]
    return {"nodes": nodes, "edges": edges}
