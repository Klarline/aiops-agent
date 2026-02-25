"""Expected utility scorer for remediation actions.

Computes risk-adjusted expected utility for each candidate action
to support decision-making beyond simple rule matching.

Example (from DESIGN.md):
    Incident: DB latency spike (severity=0.7, confidence=0.8)

    Action A — Restart DB:
      EU = 0.65 × 100 - 30 = 35.0  <- SELECTED

    Action B — Scale out:
      EU = 0.65 × 80 - 50 = 2.0

    Risk check: 0.7 × 0.2 × 2 = 0.28 < threshold(0.5) -> auto-execute
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class UtilityScore:
    """Result of utility computation for a candidate action."""

    action: str
    expected_utility: float
    risk_score: float
    auto_execute: bool


class UtilityScorer:
    """Compute expected utility for candidate remediation actions."""

    def __init__(self, risk_threshold: float = 0.5):
        self.risk_threshold = risk_threshold

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
        cost = ACTION_COSTS.get(action, 20.0)

        eu = success_prob * benefit - cost

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
