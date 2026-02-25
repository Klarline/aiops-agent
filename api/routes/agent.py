"""Agent API routes for scenario execution and status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

import numpy as np

from orchestrator.orchestrator import Orchestrator
from orchestrator.scenario_registry import list_scenarios
from agent.agent import AIOpsAgent
from features.feature_extractor import extract_features_batch
from detection.ensemble import EnsembleDetector
from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics

router = APIRouter()

_state: dict[str, Any] = {
    "orchestrator": None,
    "agent": None,
    "ensemble": None,
    "running": False,
    "history": [],
}


def _ensure_ensemble() -> EnsembleDetector:
    if _state["ensemble"] is None:
        topology = build_topology()
        metrics = generate_metrics(topology, 3600, 10, seed=42)
        all_feats = []
        for svc, df in metrics.items():
            feats = extract_features_batch(df, window_size=6)
            all_feats.append(feats)
        features = np.vstack(all_feats)
        features = features[~np.isnan(features).any(axis=1)]
        ensemble = EnsembleDetector()
        ensemble.fit(features)
        _state["ensemble"] = ensemble
    return _state["ensemble"]


@router.get("/scenarios")
async def list_available_scenarios():
    """List all available fault scenarios."""
    return {"scenarios": list_scenarios()}


class StartRequest(BaseModel):
    scenario_id: str
    seed: int = 42


@router.post("/scenarios/start")
async def start_scenario(req: StartRequest):
    """Start a fault scenario."""
    ensemble = _ensure_ensemble()
    orch = Orchestrator(seed=req.seed)
    ctx = orch.init_problem(req.scenario_id)

    agent = AIOpsAgent()
    agent.set_ensemble(ensemble)

    _state["orchestrator"] = orch
    _state["agent"] = agent
    _state["running"] = True
    _state["history"] = []

    return {
        "status": "started",
        "scenario": req.scenario_id,
        "services": ctx.services,
        "description": ctx.description,
    }


@router.post("/step")
async def agent_step(n_steps: int = 1):
    """Advance the simulation by n steps."""
    orch = _state.get("orchestrator")
    agent = _state.get("agent")
    if orch is None or agent is None:
        return {"error": "No scenario running. Start one first."}

    results = []
    for _ in range(n_steps):
        obs = orch.env.step()
        if obs is None:
            _state["running"] = False
            break

        action = agent.get_action(obs)
        if action.action != "continue_monitoring":
            orch.env.execute_action(action.action, action.target, **action.details)

        results.append({
            "step": obs.step_index,
            "action": action.action,
            "target": action.target,
            "explanation": action.explanation,
            "confidence": action.confidence,
        })

    return {"steps": results, "running": _state["running"]}


@router.get("/status")
async def get_agent_status():
    """Get current agent state."""
    agent = _state.get("agent")
    orch = _state.get("orchestrator")
    return {
        "running": _state["running"],
        "step": orch.env.current_step if orch else 0,
        "resolved": orch.env.is_resolved() if orch else False,
        "reasoning_log_length": len(agent.reasoning_log) if agent else 0,
    }


@router.get("/log")
async def get_agent_log():
    """Get full reasoning log with NL summaries."""
    agent = _state.get("agent")
    if agent is None:
        return {"log": []}
    return {"log": agent.reasoning_log}


@router.get("/evaluate")
async def evaluate_agent():
    """Evaluate the agent on the current scenario."""
    orch = _state.get("orchestrator")
    if orch is None:
        return {"error": "No scenario running."}
    result = orch.evaluate()
    return {
        "detection": result.detection,
        "localization": result.localization,
        "diagnosis": result.diagnosis,
        "mitigation": result.mitigation,
        "score": result.score,
        "details": result.details,
    }
