"""Agent API routes for scenario execution, live streaming, and comparison."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from orchestrator.scenario_registry import list_scenarios
from api.shared_state import state

router = APIRouter()


@router.get("/scenarios")
async def list_available_scenarios():
    return {"scenarios": list_scenarios()}


class StartRequest(BaseModel):
    scenario_id: str
    seed: int = 42


@router.post("/scenarios/start")
async def start_scenario(req: StartRequest):
    ctx = state.reset(req.scenario_id, req.seed)
    return {
        "status": "started",
        "scenario": req.scenario_id,
        "services": ctx.services,
        "description": ctx.description,
    }


@router.post("/step")
async def agent_step(n_steps: int = 1):
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


@router.post("/run-scenario")
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


@router.post("/compare")
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
    """Live metric streaming. Send {scenario_id, seed?, speed_ms?} to start."""
    await ws.accept()
    try:
        init = await ws.receive_json()
        scenario_id = init.get("scenario_id", "cpu_saturation_order")
        seed = init.get("seed", 42)
        speed_ms = init.get("speed_ms", 100)

        ctx = state.reset(scenario_id, seed)
        await ws.send_json({
            "type": "init",
            "scenario": scenario_id,
            "description": ctx.description,
            "services": ctx.services,
        })

        while state.running:
            step_data = state.step_once()
            if step_data is None:
                break

            scores = step_data.get("anomaly_scores", {})
            msg: dict[str, Any] = {
                "type": "step",
                "step": step_data["step"],
                "metrics": step_data["metrics"],
                "anomaly_scores": scores,
                "action": step_data["action"],
            }

            if step_data["action"] != "continue_monitoring":
                shap_data = state.get_shap_values()
                msg["type"] = "detection"
                msg["target"] = step_data["target"]
                msg["diagnosis"] = step_data["diagnosis"]
                msg["explanation"] = step_data["explanation"]
                msg["confidence"] = step_data["confidence"]
                msg["shap"] = shap_data
                msg["log"] = state.agent.reasoning_log if state.agent else []
                msg["reasoning_chain"] = state.agent.reasoning_chain if state.agent else []

            await ws.send_json(msg)
            await asyncio.sleep(speed_ms / 1000.0)

        orch = state.orchestrator
        eval_result = orch.evaluate() if orch else None
        await ws.send_json({
            "type": "complete",
            "evaluation": {
                "detection": eval_result.detection,
                "localization": eval_result.localization,
                "diagnosis": eval_result.diagnosis,
                "mitigation": eval_result.mitigation,
                "score": eval_result.score,
                "details": eval_result.details,
            } if eval_result else None,
            "detection_step": state.detection_step,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
