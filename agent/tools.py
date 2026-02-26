"""Tool registry wrapping ML modules for LLM agent tool-calling.

Each tool exposes a name, description, parameter schema, and callable.
The LLM agent selects tools in a ReAct loop to investigate and remediate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from features.feature_extractor import extract_features
from detection.ensemble import EnsembleDetector
from detection.explainer import ShapExplainer
from diagnosis.localizer import ServiceLocalizer
from diagnosis.diagnoser import FaultDiagnoser
from knowledge_base.incident_store import IncidentStore
from simulator.environment import SimulatedEnvironment, Observation


@dataclass
class Tool:
    """Single callable tool exposed to the LLM agent."""

    name: str
    description: str
    parameters: dict[str, str]
    function: Callable[..., Any]
    is_terminal: bool = False


@dataclass
class ToolResult:
    """Result returned after executing a tool."""

    tool_name: str
    success: bool
    data: dict[str, Any]

    def summary(self) -> str:
        return json.dumps(self.data, default=str)


TERMINAL_TOOLS = frozenset({
    "restart_service", "scale_service", "rollback_deploy",
    "block_ip", "set_rate_limit", "alert_human",
})


class ToolRegistry:
    """Registry of tools available to the LLM agent.

    Wraps the ML modules (ensemble, SHAP, localizer, diagnoser) and
    environment actions as callable tools with structured I/O.
    """

    def __init__(
        self,
        env: SimulatedEnvironment,
        ensemble: EnsembleDetector,
        explainer: ShapExplainer | None,
        localizer: ServiceLocalizer,
        diagnoser: FaultDiagnoser,
        incident_store: IncidentStore | None = None,
    ):
        self.env = env
        self.ensemble = ensemble
        self.explainer = explainer
        self.localizer = localizer
        self.diagnoser = diagnoser
        self.incident_store = incident_store or IncidentStore()
        self._metrics_history: dict[str, list[dict]] = {}
        self._tools: dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        tools = [
            Tool("get_metrics",
                 "Get current metrics and anomaly score for a service.",
                 {"service": "string - service name"},
                 self._get_metrics),
            Tool("explain_anomaly",
                 "Get SHAP explanation for why a service was flagged. Returns top contributing features.",
                 {"service": "string - service name"},
                 self._explain_anomaly),
            Tool("get_topology",
                 "Get the service dependency graph. Useful for tracing root cause upstream.",
                 {},
                 self._get_topology),
            Tool("localize_root_cause",
                 "Run ML-based root cause localization across all services.",
                 {},
                 self._localize_root_cause),
            Tool("diagnose",
                 "Classify the fault type for a service using metric pattern matching.",
                 {"service": "string - service name"},
                 self._diagnose),
            Tool("check_similar_incidents",
                 "Check incident memory for similar past incidents and their outcomes.",
                 {"fault_type": "string - suspected fault type"},
                 self._check_similar_incidents),
            Tool("restart_service",
                 "Restart a service. Effective for memory leaks or stuck processes.",
                 {"service": "string - service to restart"},
                 self._restart_service, is_terminal=True),
            Tool("scale_service",
                 "Scale out a service by adding replicas. Effective for CPU saturation.",
                 {"service": "string - service to scale"},
                 self._scale_service, is_terminal=True),
            Tool("rollback_deploy",
                 "Rollback to previous deployment. Effective for deployment regressions.",
                 {"service": "string - service to rollback"},
                 self._rollback_deploy, is_terminal=True),
            Tool("block_ip",
                 "Block an IP address. Effective for brute force or suspicious access.",
                 {"ip": "string - IP to block (default: scenario source IP)"},
                 self._block_ip, is_terminal=True),
            Tool("set_rate_limit",
                 "Enable rate limiting on a service. Effective for DDoS attacks.",
                 {"service": "string - service to rate limit"},
                 self._set_rate_limit, is_terminal=True),
            Tool("alert_human",
                 "Escalate to human operator when confidence is low.",
                 {"message": "string - alert message"},
                 self._alert_human, is_terminal=True),
        ]
        self._tools = {t.name: t for t in tools}

    @property
    def tools(self) -> dict[str, Tool]:
        return self._tools

    def descriptions_text(self) -> str:
        """Format tool descriptions for the system prompt."""
        lines = []
        for t in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items()) or "none"
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, {"error": f"Unknown tool: {name}"})
        try:
            data = tool.function(**args)
            return ToolResult(name, True, data)
        except Exception as e:
            return ToolResult(name, False, {"error": str(e)})

    def is_terminal(self, name: str) -> bool:
        return name in TERMINAL_TOOLS

    def update_metrics_history(self, observation: Observation) -> None:
        """Ingest latest observation into rolling history per service."""
        for svc, metrics in observation.metrics.items():
            if svc not in self._metrics_history:
                self._metrics_history[svc] = []
            self._metrics_history[svc].append(metrics)
            if len(self._metrics_history[svc]) > 60:
                self._metrics_history[svc] = self._metrics_history[svc][-60:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _features_for(self, service: str) -> np.ndarray | None:
        history = self._metrics_history.get(service, [])
        if len(history) < 6:
            return None
        return extract_features(pd.DataFrame(history), window_size=6)

    # ------------------------------------------------------------------
    # Investigation tools
    # ------------------------------------------------------------------

    def _get_metrics(self, service: str) -> dict:
        metrics = self.env.get_current_metrics().get(service)
        if metrics is None:
            return {"error": f"Unknown service: {service}"}
        features = self._features_for(service)
        anomaly = {"anomaly_score": 0.0, "is_anomalous": False, "uncertainty": 0.0}
        if features is not None:
            r = self.ensemble.detect(features)
            anomaly = {"anomaly_score": round(r.score, 3),
                       "is_anomalous": r.is_anomalous,
                       "uncertainty": round(r.uncertainty, 3)}
        return {"service": service,
                "metrics": {k: round(v, 2) for k, v in metrics.items()},
                **anomaly}

    def _explain_anomaly(self, service: str) -> dict:
        if self.explainer is None:
            return {"error": "SHAP explainer not available"}
        features = self._features_for(service)
        if features is None:
            return {"error": f"Not enough metric history for {service}"}
        expl = self.explainer.explain(features)
        return {"service": service,
                "top_features": [{"name": n, "attribution": round(v, 4)}
                                 for n, v in expl.top_features]}

    def _get_topology(self) -> dict:
        edges = [{"from": u, "to": v} for u, v in self.env.topology.edges()]
        return {"services": list(self.env.topology.nodes()),
                "dependencies": edges}

    def _localize_root_cause(self) -> dict:
        scores: dict[str, float] = {}
        for svc in self.env.topology.nodes():
            feats = self._features_for(svc)
            if feats is not None:
                scores[svc] = self.ensemble.detect(feats).score
            else:
                scores[svc] = 0.0
        root = self.localizer.localize(scores, self.env.topology)
        return {"root_cause_service": root,
                "all_scores": {k: round(v, 3) for k, v in scores.items()}}

    def _diagnose(self, service: str) -> dict:
        current_metrics = self.env.get_current_metrics()
        history_df = None
        history = self._metrics_history.get(service, [])
        if len(history) >= 6:
            history_df = pd.DataFrame(history)
        feats = self._features_for(service)
        score = self.ensemble.detect(feats).score if feats is not None else 0.0
        d = self.diagnoser.diagnose(current_metrics, history_df, service, score)
        return {"service": service,
                "fault_type": d.fault_type,
                "confidence": round(d.confidence, 2),
                "severity": round(d.severity, 2),
                "is_security": d.is_security}

    def _check_similar_incidents(self, fault_type: str) -> dict:
        similar = self.incident_store.find_similar(fault_type, "")
        return {"fault_type": fault_type,
                "similar_count": len(similar),
                "past_incidents": [{"fault_type": i.fault_type,
                                    "service": i.service,
                                    "action_taken": i.action_taken,
                                    "resolved": i.resolved}
                                   for i in similar[:5]]}

    # ------------------------------------------------------------------
    # Remediation tools (terminal actions)
    # ------------------------------------------------------------------

    def _restart_service(self, service: str) -> dict:
        r = self.env.execute_action("restart_service", service)
        return {"action": "restart_service", "target": service,
                "success": r.success, "message": r.message}

    def _scale_service(self, service: str) -> dict:
        r = self.env.execute_action("scale_out", service)
        return {"action": "scale_out", "target": service,
                "success": r.success, "message": r.message}

    def _rollback_deploy(self, service: str) -> dict:
        r = self.env.execute_action("rollback", service)
        return {"action": "rollback", "target": service,
                "success": r.success, "message": r.message}

    def _block_ip(self, ip: str = "") -> dict:
        if not ip and self.env.scenario:
            ip = self.env.scenario.metadata.get("source_ip", "10.0.0.99")
        target = self.env.scenario.target_service if self.env.scenario else ""
        r = self.env.execute_action("block_ip", target, ip=ip)
        return {"action": "block_ip", "ip": ip,
                "success": r.success, "message": r.message}

    def _set_rate_limit(self, service: str) -> dict:
        r = self.env.execute_action("rate_limit", service)
        return {"action": "rate_limit", "target": service,
                "success": r.success, "message": r.message}

    def _alert_human(self, message: str = "") -> dict:
        target = self.env.scenario.target_service if self.env.scenario else ""
        r = self.env.execute_action("alert_human", target)
        return {"action": "alert_human", "message": message,
                "success": r.success}
