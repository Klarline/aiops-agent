"""Metrics API routes — uses the shared simulation state."""

from __future__ import annotations

from fastapi import APIRouter

from api.shared_state import state
from simulator.service_topology import build_topology

router = APIRouter()


@router.get("/current")
async def get_current_metrics():
    orch = state.orchestrator
    if orch is None or orch.env.metrics_data is None:
        return {}
    return orch.env.get_current_metrics()


@router.get("/history/{service}")
async def get_metrics_history(service: str, lookback: int = 60):
    orch = state.orchestrator
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
