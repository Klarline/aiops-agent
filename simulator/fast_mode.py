"""Fast simulation mode for RL training.

Stripped-down environment that runs < 0.1s per episode for
Q-learning training without plotting or delays.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from simulator.fault_injector import FaultScenario

FAULT_TYPES = [
    "memory_leak", "cpu_saturation", "brute_force", "transaction_stall",
    "cascading_failure", "deployment_regression", "anomalous_access", "ddos",
]

SERVICES = ["api-gateway", "auth-service", "order-service", "user-db", "order-db"]

CORRECT_ACTIONS = {
    "memory_leak": "restart_service",
    "cpu_saturation": "scale_out",
    "brute_force": "block_ip",
    "transaction_stall": "alert_human",
    "cascading_failure": "restart_service",
    "deployment_regression": "rollback",
    "anomalous_access": "alert_human",
    "ddos": "rate_limit",
}

ACTIONS = ["restart_service", "scale_out", "rollback", "block_ip", "rate_limit", "alert_human"]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]


@dataclass
class FastState:
    """Discretized state for Q-learning."""

    fault_type_idx: int
    severity_idx: int
    is_security: int

    @property
    def as_tuple(self) -> tuple:
        return (self.fault_type_idx, self.severity_idx, self.is_security)


class FastSimulator:
    """Minimal simulator for RL training — no DataFrames, no time series."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def sample_episode(self) -> tuple[FastState, str, str]:
        """Sample a random fault scenario and return (state, correct_action, fault_type)."""
        fault_type = self.rng.choice(FAULT_TYPES)
        severity = self.rng.choice(range(4))
        is_security = int(fault_type in ("brute_force", "anomalous_access", "ddos"))
        fault_idx = FAULT_TYPES.index(fault_type)

        state = FastState(fault_idx, severity, is_security)
        correct_action = CORRECT_ACTIONS[fault_type]
        return state, correct_action, fault_type

    @staticmethod
    def compute_reward(action: str, correct_action: str, fault_type: str) -> float:
        """Compute reward for an action given the correct action."""
        if action == correct_action:
            return 10.0

        is_security = fault_type in ("brute_force", "anomalous_access", "ddos")
        if is_security and action == "alert_human":
            return 3.0
        if is_security and action not in ("block_ip", "rate_limit", "alert_human"):
            return -20.0

        if action == "alert_human":
            return -2.0

        return -5.0

    @staticmethod
    def n_states() -> tuple[int, int, int]:
        return (len(FAULT_TYPES), len(SEVERITY_LEVELS), 2)

    @staticmethod
    def n_actions() -> int:
        return len(ACTIONS)
