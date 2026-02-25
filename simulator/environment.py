"""Unified simulated environment interface.

Combines service topology, metrics generation, and fault injection into
a single interface used by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from simulator.service_topology import build_topology, METRIC_NAMES
from simulator.metrics_generator import generate_metrics
from simulator.fault_injector import FaultScenario, inject_fault


@dataclass
class Observation:
    """Single time-step observation returned to the agent."""

    timestamp: Any
    metrics: dict[str, dict[str, float]]
    topology: nx.DiGraph
    step_index: int


@dataclass
class ActionResult:
    """Result of executing a remediation action."""

    success: bool
    action: str
    target: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class SimulatedEnvironment:
    """Unified environment that generates metrics, injects faults,
    and handles remediation actions."""

    DURATION_SECONDS = 3600  # 1 hour of simulated data
    INTERVAL_SECONDS = 10

    def __init__(self, seed: int = 42):
        self.topology = build_topology()
        self.seed = seed
        self.current_step = 0
        self.metrics_data: dict[str, pd.DataFrame] | None = None
        self.scenario: FaultScenario | None = None
        self.actions_taken: list[dict] = []
        self.blocked_ips: set[str] = set()
        self.audit_log: list[dict] = []
        self._resolved = False

    @property
    def n_steps(self) -> int:
        return self.DURATION_SECONDS // self.INTERVAL_SECONDS

    def reset(self, scenario: FaultScenario) -> None:
        """Generate metrics, inject the fault, and reset state."""
        self.scenario = scenario
        self.current_step = 0
        self.actions_taken = []
        self.blocked_ips = set()
        self.audit_log = []
        self._resolved = False

        normal = generate_metrics(
            self.topology,
            self.DURATION_SECONDS,
            self.INTERVAL_SECONDS,
            seed=self.seed,
        )
        self.metrics_data = inject_fault(normal, scenario, self.topology)

    def step(self) -> Observation | None:
        """Advance one time step and return observation."""
        if self.metrics_data is None:
            raise RuntimeError("Call reset() before step()")
        if self.current_step >= self.n_steps:
            return None

        snapshot = self.get_current_metrics()
        first_svc = list(self.metrics_data.keys())[0]
        ts = self.metrics_data[first_svc].index[self.current_step]

        obs = Observation(
            timestamp=ts,
            metrics=snapshot,
            topology=self.topology,
            step_index=self.current_step,
        )
        self.current_step += 1
        return obs

    def get_current_metrics(self) -> dict[str, dict[str, float]]:
        """Return current metrics for all services as nested dict."""
        if self.metrics_data is None:
            raise RuntimeError("Call reset() before get_current_metrics()")
        result = {}
        idx = min(self.current_step, self.n_steps - 1)
        for svc, df in self.metrics_data.items():
            result[svc] = df.iloc[idx].to_dict()
        return result

    def get_metrics_history(
        self, service: str, lookback_steps: int = 30
    ) -> pd.DataFrame:
        """Return recent metrics history for a single service."""
        if self.metrics_data is None:
            raise RuntimeError("Call reset() first")
        df = self.metrics_data[service]
        start = max(0, self.current_step - lookback_steps)
        end = self.current_step + 1
        return df.iloc[start:end]

    def execute_action(
        self, action: str, target: str, **kwargs: Any
    ) -> ActionResult:
        """Simulate remediation action execution."""
        record = {
            "step": self.current_step,
            "action": action,
            "target": target,
            "kwargs": kwargs,
        }
        self.actions_taken.append(record)

        if action == "restart_service":
            return self._handle_restart(target)
        elif action == "scale_out":
            return self._handle_scale(target)
        elif action == "rollback":
            return self._handle_rollback(target)
        elif action == "block_ip":
            return self._handle_block_ip(target, **kwargs)
        elif action == "rate_limit":
            return self._handle_rate_limit(target)
        elif action == "alert_human":
            return self._handle_alert_human(target)
        elif action == "continue_monitoring":
            return ActionResult(True, action, target, "Continuing to monitor")
        else:
            return ActionResult(False, action, target, f"Unknown action: {action}")

    def _handle_restart(self, target: str) -> ActionResult:
        if self.scenario and self.scenario.target_service == target:
            if self.scenario.fault_type in ("memory_leak", "cascading_failure"):
                self._resolve_fault(target)
                return ActionResult(True, "restart_service", target, "Service restarted, metrics recovering")
        return ActionResult(True, "restart_service", target, "Service restarted")

    def _handle_scale(self, target: str) -> ActionResult:
        if self.scenario and self.scenario.target_service == target:
            if self.scenario.fault_type == "cpu_saturation":
                self._resolve_fault(target)
                return ActionResult(True, "scale_out", target, "Scaled to 4 replicas, load distributed")
        return ActionResult(True, "scale_out", target, "Service scaled")

    def _handle_rollback(self, target: str) -> ActionResult:
        if self.scenario and self.scenario.target_service == target:
            if self.scenario.fault_type == "deployment_regression":
                self._resolve_fault(target)
                return ActionResult(True, "rollback", target, "Rolled back to previous version")
        return ActionResult(True, "rollback", target, "Rollback executed")

    def _handle_block_ip(self, target: str, **kwargs: Any) -> ActionResult:
        ip = kwargs.get("ip", self.scenario.metadata.get("source_ip", "unknown") if self.scenario else "unknown")
        self.blocked_ips.add(ip)
        self.audit_log.append({
            "timestamp": str(pd.Timestamp.now()),
            "event": "ip_blocked",
            "ip": ip,
            "service": target,
            "reason": f"Automated block due to {self.scenario.fault_type if self.scenario else 'unknown'}",
        })
        if self.scenario and self.scenario.fault_type == "brute_force":
            self._resolve_fault(target)
        return ActionResult(True, "block_ip", target, f"IP {ip} blocked", {"ip": ip})

    def _handle_rate_limit(self, target: str) -> ActionResult:
        if self.scenario and self.scenario.fault_type == "ddos":
            self._resolve_fault(target)
            return ActionResult(True, "rate_limit", target, "Rate limiting applied")
        return ActionResult(True, "rate_limit", target, "Rate limit configured")

    def _handle_alert_human(self, target: str) -> ActionResult:
        self._resolved = True
        return ActionResult(True, "alert_human", target, "Human operator alerted")

    def _resolve_fault(self, target: str) -> None:
        """Simulate metrics recovering after correct remediation."""
        self._resolved = True
        if self.metrics_data is None or self.scenario is None:
            return

        normal = generate_metrics(
            self.topology,
            self.DURATION_SECONDS,
            self.INTERVAL_SECONDS,
            seed=self.seed,
        )
        remaining = slice(self.current_step, None)
        for svc in self.metrics_data:
            self.metrics_data[svc].iloc[remaining] = normal[svc].iloc[remaining]

    def get_topology(self) -> nx.DiGraph:
        return self.topology

    def get_ground_truth(self) -> FaultScenario | None:
        return self.scenario

    def is_resolved(self) -> bool:
        return self._resolved
