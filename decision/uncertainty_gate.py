"""Uncertainty gate for safe autonomous decision-making.

Forces escalation to human when model confidence is insufficient
or when risk exceeds safe auto-execution thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

from detection.ensemble import AnomalyResult


@dataclass
class GateDecision:
    """Result of the uncertainty gate check."""

    should_escalate: bool
    reason: str
    requires_approval: bool = False


class UncertaintyGate:
    """Gate that forces human escalation when confidence is insufficient."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.4,
        risk_threshold: float = 0.5,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.risk_threshold = risk_threshold

    def check(
        self,
        anomaly_result: AnomalyResult,
        severity: float = 0.5,
        blast_radius: int = 1,
    ) -> GateDecision:
        """Determine if action should be escalated or requires approval.

        Escalate if:
        - Models strongly disagree (high uncertainty).
        - Risk score (severity * uncertainty * blast_radius) exceeds threshold.
        """
        if anomaly_result.uncertainty > self.uncertainty_threshold:
            return GateDecision(
                should_escalate=True,
                reason=(
                    f"Model disagreement too high ({anomaly_result.uncertainty:.2f} "
                    f"> {self.uncertainty_threshold})"
                ),
            )

        risk = severity * anomaly_result.uncertainty * blast_radius
        if risk > self.risk_threshold:
            return GateDecision(
                should_escalate=False,
                requires_approval=True,
                reason=f"Risk score {risk:.2f} exceeds threshold {self.risk_threshold}",
            )

        return GateDecision(should_escalate=False, reason="Within safe parameters")
