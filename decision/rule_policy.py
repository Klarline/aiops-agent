"""Rule-based expert policy for remediation decisions.

The reliable default decision engine. Maps diagnosed fault types
to remediation actions with confidence-gated escalation.
"""

from __future__ import annotations

from diagnosis.diagnoser import Diagnosis


ACTION_MAP = {
    "memory_leak": "restart_service",
    "cpu_saturation": "scale_out",
    "deployment_regression": "rollback",
    "transaction_stall": "alert_human",
    "cascading_failure": "restart_service",
}

SECURITY_ACTION_MAP = {
    "brute_force": "block_ip",
    "ddos": "rate_limit",
    "anomalous_access": "alert_human",
}


class RuleBasedPolicy:
    """Expert rule-based policy for remediation decisions."""

    def __init__(
        self,
        security_confidence_threshold: float = 0.7,
        action_confidence_threshold: float = 0.5,
    ):
        self.security_threshold = security_confidence_threshold
        self.action_threshold = action_confidence_threshold

    def decide(self, diagnosis: Diagnosis) -> str:
        """Choose a remediation action based on diagnosis.

        Decision logic:
        1. Security faults with sufficient confidence -> specific security action.
        2. Operational faults with high confidence -> mapped action.
        3. Low confidence -> escalate to human.
        """
        if diagnosis.is_security and diagnosis.confidence >= self.security_threshold:
            action = SECURITY_ACTION_MAP.get(diagnosis.fault_type, "alert_human")
            return action

        if diagnosis.confidence >= self.action_threshold:
            action = ACTION_MAP.get(diagnosis.fault_type, "alert_human")
            return action

        return "alert_human"
