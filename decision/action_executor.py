"""Action executor with security remediation and audit logging.

Executes remediation actions in the simulated environment with
special handling for security events (IP blocking, audit trails).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulator.environment import SimulatedEnvironment, ActionResult
from diagnosis.diagnoser import Diagnosis


class ActionExecutor:
    """Execute remediation actions with audit trail."""

    def __init__(self, environment: SimulatedEnvironment):
        self.env = environment
        self.audit_log: list[dict[str, Any]] = []
        self.blocked_ips: set[str] = set()

    def execute(
        self,
        action: str,
        diagnosis: Diagnosis,
        **kwargs: Any,
    ) -> ActionResult:
        """Execute a remediation action.

        For security events, also updates the audit log and blocked IPs.
        """
        target = diagnosis.localized_service

        if action == "block_ip":
            return self._execute_block_ip(target, diagnosis, **kwargs)
        elif action == "rate_limit":
            return self._execute_rate_limit(target, diagnosis)

        result = self.env.execute_action(action, target, **kwargs)

        self.audit_log.append({
            "event": "action_executed",
            "action": action,
            "target": target,
            "fault_type": diagnosis.fault_type,
            "confidence": diagnosis.confidence,
            "success": result.success,
        })

        return result

    def _execute_block_ip(
        self, target: str, diagnosis: Diagnosis, **kwargs: Any
    ) -> ActionResult:
        ip = kwargs.get("ip", "unknown")
        if ip == "unknown" and self.env.scenario:
            ip = self.env.scenario.metadata.get("source_ip", "unknown")

        self.blocked_ips.add(ip)

        result = self.env.execute_action("block_ip", target, ip=ip)

        self.audit_log.append({
            "event": "ip_blocked",
            "ip": ip,
            "target": target,
            "fault_type": diagnosis.fault_type,
            "confidence": diagnosis.confidence,
            "reason": f"Automated block: {diagnosis.fault_type} detected with {diagnosis.confidence:.0%} confidence",
        })

        return result

    def _execute_rate_limit(
        self, target: str, diagnosis: Diagnosis
    ) -> ActionResult:
        result = self.env.execute_action("rate_limit", target)

        self.audit_log.append({
            "event": "rate_limit_applied",
            "target": target,
            "fault_type": diagnosis.fault_type,
            "confidence": diagnosis.confidence,
        })

        return result
