"""Metrics API routes."""

from __future__ import annotations

from fastapi import APIRouter

from simulator.environment import SimulatedEnvironment

router = APIRouter()

_env: SimulatedEnvironment | None = None


def _get_env() -> SimulatedEnvironment:
    global _env
    if _env is None:
        _env = SimulatedEnvironment(seed=42)
    return _env


@router.get("/current")
async def get_current_metrics():
    """Get current metrics snapshot for all services."""
    env = _get_env()
    if env.metrics_data is None:
        return {"error": "No scenario running. Start one first."}
    return env.get_current_metrics()


@router.get("/history/{service}")
async def get_metrics_history(service: str, lookback: int = 30):
    """Get recent metrics history for a service."""
    env = _get_env()
    if env.metrics_data is None:
        return {"error": "No scenario running."}
    try:
        df = env.get_metrics_history(service, lookback_steps=lookback)
        return {
            "service": service,
            "timestamps": [str(t) for t in df.index],
            "metrics": df.to_dict(orient="list"),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/topology")
async def get_topology():
    """Get service dependency topology."""
    env = _get_env()
    graph = env.get_topology()
    nodes = [
        {"id": n, **graph.nodes[n]} for n in graph.nodes
    ]
    edges = [{"source": u, "target": v} for u, v in graph.edges]
    return {"nodes": nodes, "edges": edges}
