"""Baseline agents for benchmark comparison.

StaticThresholdAgent: Simple threshold rules — CPU > 80% → restart, error > 0.1 → alert.
RandomAgent: Picks random actions to establish a lower bound.

These exist to prove the ML agent is better than naive approaches.
"""

from __future__ import annotations

import random

from simulator.environment import Observation
from orchestrator.problem_context import AgentAction


class StaticThresholdAgent:
    """Naive baseline: fixed per-metric thresholds, always restarts.

    This is what most teams build first and what many production
    monitoring systems still use. No ML, no topology awareness,
    no explainability.
    """

    def __init__(self):
        self._acted = False
        self._warmup = 5

    def set_ensemble(self, ensemble):
        pass

    def get_action(self, observation: Observation) -> AgentAction:
        if observation.step_index < self._warmup:
            return AgentAction(action="continue_monitoring")

        if self._acted:
            return AgentAction(action="continue_monitoring")

        for svc, metrics in observation.metrics.items():
            cpu = metrics.get("cpu_percent", 0)
            mem = metrics.get("memory_percent", 0)
            err = metrics.get("error_rate", 0)
            lat = metrics.get("latency_p50_ms", 0)

            if cpu > 80:
                self._acted = True
                return AgentAction(
                    action="restart_service", target=svc,
                    details={"diagnosis": "cpu_saturation"},
                )
            if mem > 75:
                self._acted = True
                return AgentAction(
                    action="restart_service", target=svc,
                    details={"diagnosis": "memory_leak"},
                )
            if err > 0.1:
                self._acted = True
                return AgentAction(
                    action="alert_human", target=svc,
                    details={"diagnosis": "unknown"},
                )
            if lat > 100:
                self._acted = True
                return AgentAction(
                    action="restart_service", target=svc,
                    details={"diagnosis": "deployment_regression"},
                )

        return AgentAction(action="continue_monitoring")


class RandomAgent:
    """Lower bound: takes random actions after detecting any threshold breach."""

    ACTIONS = ["restart_service", "scale_out", "rollback", "block_ip",
               "rate_limit", "alert_human"]
    FAULT_TYPES = ["cpu_saturation", "memory_leak", "brute_force",
                   "deployment_regression", "ddos", "transaction_stall"]

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._acted = False

    def set_ensemble(self, ensemble):
        pass

    def get_action(self, observation: Observation) -> AgentAction:
        if observation.step_index < 10:
            return AgentAction(action="continue_monitoring")
        if self._acted:
            return AgentAction(action="continue_monitoring")

        for svc, metrics in observation.metrics.items():
            if metrics.get("cpu_percent", 0) > 60 or metrics.get("error_rate", 0) > 0.05:
                self._acted = True
                services = list(observation.metrics.keys())
                return AgentAction(
                    action=self._rng.choice(self.ACTIONS),
                    target=self._rng.choice(services),
                    details={"diagnosis": self._rng.choice(self.FAULT_TYPES)},
                )

        return AgentAction(action="continue_monitoring")
