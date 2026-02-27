"""Expected utility scorer for remediation actions.

Computes risk-adjusted expected utility for each candidate action
to support decision-making beyond simple rule matching.

Supports optional cost model (downtime, SLA penalty) for business-aware scoring.
When no cost model is provided, uses default action costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ACTION_BENEFITS = {
    "restart_service": 100.0,
    "scale_out": 80.0,
    "rollback": 90.0,
    "block_ip": 95.0,
    "rate_limit": 85.0,
    "alert_human": 50.0,
}

ACTION_COSTS = {
    "restart_service": 30.0,
    "scale_out": 50.0,
    "rollback": 40.0,
    "block_ip": 10.0,
    "rate_limit": 15.0,
    "alert_human": 5.0,
}

FAULT_ACTION_SUCCESS_RATES = {
    ("memory_leak", "restart_service"): 0.85,
    ("cpu_saturation", "scale_out"): 0.80,
    ("deployment_regression", "rollback"): 0.90,
    ("brute_force", "block_ip"): 0.90,
    ("ddos", "rate_limit"): 0.85,
    ("cascading_failure", "restart_service"): 0.70,
    ("transaction_stall", "alert_human"): 0.60,
    ("anomalous_access", "alert_human"): 0.55,
}

# Optional cost model for business-aware utility (action_cost, downtime_seconds).
# When provided, EU = success_prob * benefit - action_cost - (downtime_seconds * 0.1).
# Default matches ACTION_COSTS (downtime_seconds=0) for backward compatibility.
COST_MODEL: dict[str, dict[str, float]] = {
    "restart_service": {"action_cost": 30.0, "downtime_seconds": 0.0},
    "scale_out": {"action_cost": 50.0, "downtime_seconds": 0.0},
    "rollback": {"action_cost": 40.0, "downtime_seconds": 0.0},
    "block_ip": {"action_cost": 10.0, "downtime_seconds": 0.0},
    "rate_limit": {"action_cost": 15.0, "downtime_seconds": 0.0},
    "alert_human": {"action_cost": 5.0, "downtime_seconds": 0.0},
}


@dataclass
class UtilityScore:
    """Result of utility computation for a candidate action."""

    action: str
    expected_utility: float
    risk_score: float
    auto_execute: bool


class UtilityScorer:
    """Compute expected utility for candidate remediation actions."""

    def __init__(
        self,
        risk_threshold: float = 0.5,
        cost_model: dict[str, dict[str, float]] | None = None,
    ):
        self.risk_threshold = risk_threshold
        self.cost_model = cost_model or COST_MODEL

    def score(
        self,
        action: str,
        fault_type: str,
        severity: float,
        confidence: float,
        blast_radius: int = 1,
        uncertainty: float = 0.2,
    ) -> UtilityScore:
        """Compute expected utility and risk for a single action."""
        base_success = FAULT_ACTION_SUCCESS_RATES.get(
            (fault_type, action), 0.3
        )
        success_prob = base_success * confidence
        benefit = ACTION_BENEFITS.get(action, 50.0)
        cost_entry = self.cost_model.get(action, {"action_cost": 20.0, "downtime_seconds": 0.0})
        action_cost = cost_entry.get("action_cost", ACTION_COSTS.get(action, 20.0))
        downtime_sec = cost_entry.get("downtime_seconds", 0.0)
        downtime_cost = downtime_sec * 0.1  # 0.1 utility per second of downtime
        total_cost = action_cost + downtime_cost
        eu = success_prob * benefit - total_cost

        risk = severity * uncertainty * blast_radius
        auto_execute = risk < self.risk_threshold

        return UtilityScore(
            action=action,
            expected_utility=eu,
            risk_score=risk,
            auto_execute=auto_execute,
        )

    def rank_actions(
        self,
        fault_type: str,
        severity: float,
        confidence: float,
        blast_radius: int = 1,
        uncertainty: float = 0.2,
    ) -> list[UtilityScore]:
        """Score and rank all possible actions for a diagnosis."""
        candidates = list(ACTION_BENEFITS.keys())
        scores = [
            self.score(a, fault_type, severity, confidence, blast_radius, uncertainty)
            for a in candidates
        ]
        scores.sort(key=lambda s: s.expected_utility, reverse=True)
        return scores
