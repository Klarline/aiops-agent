"""Agent API routes for scenario execution, live streaming, and comparison."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from orchestrator.scenario_registry import list_scenarios
from api.shared_state import run_registry, state
from api.auth import require_api_key

router = APIRouter()


@router.get("/scenarios")
async def list_available_scenarios():
    return {"scenarios": list_scenarios()}


class StartRequest(BaseModel):
    scenario_id: str
    seed: int = 42


@router.post("/scenarios/start", dependencies=[Depends(require_api_key)])
async def start_scenario(req: StartRequest):
    """Legacy: uses default session. For sessionized runs, use POST /runs."""
    s = state
    ctx = s.reset(req.scenario_id, req.seed)
    return {
        "status": "started",
        "scenario": req.scenario_id,
        "services": ctx.services,
        "description": ctx.description,
    }


@router.post("/step", dependencies=[Depends(require_api_key)])
async def agent_step(n_steps: int = 1):
    """Legacy: uses default session."""
    results = []
    for _ in range(n_steps):
        step_data = state.step_once()
        if step_data is None:
            state.running = False
            break
        results.append(step_data)

    return {
        "steps": results,
        "running": state.running,
        "detection_step": state.detection_step,
    }


# --- Sessionized endpoints (Phase C) ---

class CreateRunRequest(BaseModel):
    scenario_id: str
    seed: int = 42


@router.post("/runs", dependencies=[Depends(require_api_key)])
async def create_run(req: CreateRunRequest):
    """Create a new run session. Returns run_id and incident_id."""
    run_id, incident_id, session = run_registry.create_run()
    ctx = session.reset(req.scenario_id, req.seed, incident_id=incident_id)
    return {
        "run_id": run_id,
        "incident_id": incident_id,
        "status": "started",
        "scenario": req.scenario_id,
        "services": ctx.services,
        "description": ctx.description,
    }


@router.post("/runs/{run_id}/step", dependencies=[Depends(require_api_key)])
async def run_step(run_id: str, n_steps: int = 1):
    """Advance the run by n steps."""
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    results = []
    for _ in range(n_steps):
        step_data = session.step_once()
        if step_data is None:
            session.running = False
            break
        results.append(step_data)

    return {
        "steps": results,
        "running": session.running,
        "detection_step": session.detection_step,
        "run_id": run_id,
        "incident_id": session.incident_id,
    }


@router.get("/runs/{run_id}/status")
async def get_run_status(run_id: str):
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    orch = session.orchestrator
    agent = session.agent
    return {
        "run_id": run_id,
        "incident_id": session.incident_id,
        "running": session.running,
        "step": orch.env.current_step if orch else 0,
        "total_steps": orch.env.n_steps if orch else 0,
        "resolved": orch.env.is_resolved() if orch else False,
        "detection_step": session.detection_step,
        "reasoning_log_length": len(agent.reasoning_log) if agent else 0,
    }


@router.get("/runs/{run_id}/log")
async def get_run_log(run_id: str):
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    agent = session.agent
    if agent is None:
        return {"log": [], "reasoning_chain": [], "run_id": run_id}
    return {
        "run_id": run_id,
        "log": agent.reasoning_log,
        "reasoning_chain": agent.reasoning_chain,
    }


@router.get("/runs/{run_id}/shap")
async def get_run_shap(run_id: str, service: str | None = None):
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    data = session.get_shap_values(service)
    data["run_id"] = run_id
    return data


@router.get("/runs/{run_id}/evaluate")
async def evaluate_run(run_id: str):
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    orch = session.orchestrator
    if orch is None:
        return {"error": "No scenario running.", "run_id": run_id}
    result = orch.evaluate()
    return {
        "run_id": run_id,
        "incident_id": session.incident_id,
        "detection": result.detection,
        "localization": result.localization,
        "diagnosis": result.diagnosis,
        "mitigation": result.mitigation,
        "score": result.score,
        "details": result.details,
    }


@router.get("/runs/{run_id}/anomaly-timeline")
async def get_run_anomaly_timeline(run_id: str):
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "timeline": session.anomaly_timeline}


@router.get("/runs/{run_id}/metrics-history")
async def get_run_metrics_history(run_id: str):
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "history": session.get_full_history()}


@router.get("/status")
async def get_agent_status():
    orch = state.orchestrator
    agent = state.agent
    return {
        "running": state.running,
        "step": orch.env.current_step if orch else 0,
        "total_steps": orch.env.n_steps if orch else 0,
        "resolved": orch.env.is_resolved() if orch else False,
        "detection_step": state.detection_step,
        "reasoning_log_length": len(agent.reasoning_log) if agent else 0,
    }


@router.get("/log")
async def get_agent_log():
    agent = state.agent
    if agent is None:
        return {"log": [], "reasoning_chain": []}
    return {
        "log": agent.reasoning_log,
        "reasoning_chain": agent.reasoning_chain,
    }


@router.get("/shap")
async def get_shap(service: str | None = None):
    return state.get_shap_values(service)


@router.get("/anomaly-timeline")
async def get_anomaly_timeline():
    return {"timeline": state.anomaly_timeline}


@router.get("/metrics-history")
async def get_metrics_history():
    return {"history": state.get_full_history()}


@router.get("/evaluate")
async def evaluate_agent():
    orch = state.orchestrator
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


class RunScenarioRequest(BaseModel):
    scenario_id: str
    seed: int = 42
    speed_ms: int = 50


@router.post("/run-scenario", dependencies=[Depends(require_api_key)])
async def run_full_scenario(req: RunScenarioRequest):
    """Run a full scenario to completion and return all data at once."""
    ctx = state.reset(req.scenario_id, req.seed)

    all_steps: list[dict] = []
    while state.running:
        step_data = state.step_once()
        if step_data is None:
            break
        all_steps.append(step_data)

    orch = state.orchestrator
    eval_result = orch.evaluate() if orch else None
    shap_data = state.get_shap_values()

    return {
        "scenario": req.scenario_id,
        "description": ctx.description,
        "services": ctx.services,
        "total_steps": len(all_steps),
        "detection_step": state.detection_step,
        "steps": all_steps,
        "shap": shap_data,
        "evaluation": {
            "detection": eval_result.detection,
            "localization": eval_result.localization,
            "diagnosis": eval_result.diagnosis,
            "mitigation": eval_result.mitigation,
            "score": eval_result.score,
            "details": eval_result.details,
        } if eval_result else None,
        "log": state.agent.reasoning_log if state.agent else [],
        "reasoning_chain": state.agent.reasoning_chain if state.agent else [],
        "anomaly_timeline": state.anomaly_timeline,
    }


class CompareRequest(BaseModel):
    scenario_id: str
    seed: int = 42


@router.post("/compare", dependencies=[Depends(require_api_key)])
async def compare_agents(req: CompareRequest):
    """Run same scenario with ML agent vs no agent. Returns side-by-side data."""
    from evaluation.baseline_agents import StaticThresholdAgent

    state.reset(req.scenario_id, req.seed)
    ml_steps: list[dict] = []
    while state.running:
        step_data = state.step_once()
        if step_data is None:
            break
        ml_steps.append(step_data)

    ml_eval = state.orchestrator.evaluate() if state.orchestrator else None
    ml_detection_step = state.detection_step
    ml_log = state.agent.reasoning_log if state.agent else []
    ml_shap = state.get_shap_values()

    from orchestrator.orchestrator import Orchestrator
    baseline_orch = Orchestrator(seed=req.seed)
    baseline_orch.init_problem(req.scenario_id)
    baseline_agent = StaticThresholdAgent()
    baseline_agent.set_ensemble(state.ensemble)

    baseline_detection_step = -1
    baseline_steps: list[dict] = []
    for _ in range(360):
        obs = baseline_orch.env.step()
        if obs is None:
            break
        action = baseline_agent.get_action(obs)
        step_info = {
            "step": obs.step_index,
            "action": action.action,
            "target": action.target,
        }
        if action.action != "continue_monitoring":
            if baseline_detection_step < 0:
                baseline_detection_step = obs.step_index
            baseline_orch._agent_detected = True
            baseline_orch._agent_localized = action.target
            baseline_orch._agent_diagnosed = action.details.get("diagnosis", "")
            baseline_orch.env.execute_action(action.action, action.target, **action.details)
            baseline_orch._agent_mitigated = baseline_orch.env.is_resolved()
            if baseline_orch.env.is_resolved():
                baseline_steps.append(step_info)
                break
        baseline_steps.append(step_info)

    baseline_eval = baseline_orch.evaluate()

    return {
        "scenario": req.scenario_id,
        "ml_agent": {
            "name": "ML Ensemble Agent",
            "detection_step": ml_detection_step,
            "mttr_seconds": ml_detection_step * 10 if ml_detection_step > 0 else None,
            "total_steps": len(ml_steps),
            "evaluation": {
                "detection": ml_eval.detection,
                "localization": ml_eval.localization,
                "diagnosis": ml_eval.diagnosis,
                "mitigation": ml_eval.mitigation,
                "score": ml_eval.score,
            } if ml_eval else None,
            "log": ml_log,
            "shap": ml_shap,
        },
        "baseline": {
            "name": "Static Threshold",
            "detection_step": baseline_detection_step,
            "mttr_seconds": baseline_detection_step * 10 if baseline_detection_step > 0 else None,
            "total_steps": len(baseline_steps),
            "evaluation": {
                "detection": baseline_eval.detection,
                "localization": baseline_eval.localization,
                "diagnosis": baseline_eval.diagnosis,
                "mitigation": baseline_eval.mitigation,
                "score": baseline_eval.score,
            },
        },
    }


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Live metric streaming. Send {scenario_id, seed?, speed_ms?, api_key?} to start.
    Creates its own session; returns run_id and incident_id in init payload."""
    await ws.accept()
    try:
        init = await ws.receive_json()
        scenario_id = init.get("scenario_id", "cpu_saturation_order")
        seed = init.get("seed", 42)
        speed_ms = init.get("speed_ms", 100)

        # API key check when AIOPS_API_KEY is set
        configured = os.environ.get("AIOPS_API_KEY")
        if configured:
            api_key = init.get("api_key") or init.get("X-API-Key")
            if not api_key or api_key != configured:
                await ws.send_json({"type": "error", "message": "Invalid or missing API key"})
                await ws.close(code=4001)
                return

        run_id, incident_id, session = run_registry.create_run()
        ctx = session.reset(scenario_id, seed, incident_id=incident_id)

        await ws.send_json({
            "type": "init",
            "run_id": run_id,
            "incident_id": incident_id,
            "scenario": scenario_id,
            "description": ctx.description,
            "services": ctx.services,
        })

        while session.running:
            step_data = session.step_once()
            if step_data is None:
                break

            scores = step_data.get("anomaly_scores", {})
            msg: dict[str, Any] = {
                "type": "step",
                "step": step_data["step"],
                "metrics": step_data["metrics"],
                "anomaly_scores": scores,
                "action": step_data["action"],
                "run_id": run_id,
                "incident_id": incident_id,
            }

            if step_data["action"] != "continue_monitoring":
                shap_data = session.get_shap_values()
                msg["type"] = "detection"
                msg["target"] = step_data["target"]
                msg["diagnosis"] = step_data["diagnosis"]
                msg["explanation"] = step_data["explanation"]
                msg["confidence"] = step_data["confidence"]
                msg["shap"] = shap_data
                msg["log"] = session.agent.reasoning_log if session.agent else []
                msg["reasoning_chain"] = session.agent.reasoning_chain if session.agent else []

            await ws.send_json(msg)
            await asyncio.sleep(speed_ms / 1000.0)

        orch = session.orchestrator
        eval_result = orch.evaluate() if orch else None
        await ws.send_json({
            "type": "complete",
            "run_id": run_id,
            "incident_id": incident_id,
            "evaluation": {
                "detection": eval_result.detection,
                "localization": eval_result.localization,
                "diagnosis": eval_result.diagnosis,
                "mitigation": eval_result.mitigation,
                "score": eval_result.score,
                "details": eval_result.details,
            } if eval_result else None,
            "detection_step": session.detection_step,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
