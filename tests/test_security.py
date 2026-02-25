"""Tests for security remediation: IP blocking + audit logging."""

from __future__ import annotations

from simulator.environment import SimulatedEnvironment
from simulator.fault_injector import FaultScenario
from diagnosis.diagnoser import Diagnosis
from decision.action_executor import ActionExecutor


class TestSecurityRemediation:
    def test_brute_force_blocks_ip(self):
        env = SimulatedEnvironment(seed=42)
        scenario = FaultScenario(
            fault_type="brute_force",
            target_service="auth-service",
            start_time=150,
            duration=300,
            severity=0.85,
            is_security=True,
            metadata={"source_ip": "10.0.0.99"},
        )
        env.reset(scenario)

        executor = ActionExecutor(env)
        diagnosis = Diagnosis("brute_force", 0.85, "auth-service", 0.85, True)
        result = executor.execute("block_ip", diagnosis, ip="10.0.0.99")

        assert result.success
        assert "10.0.0.99" in executor.blocked_ips

    def test_brute_force_creates_audit_entry(self):
        env = SimulatedEnvironment(seed=42)
        scenario = FaultScenario(
            fault_type="brute_force",
            target_service="auth-service",
            start_time=150,
            duration=300,
            severity=0.85,
            is_security=True,
            metadata={"source_ip": "10.0.0.99"},
        )
        env.reset(scenario)

        executor = ActionExecutor(env)
        diagnosis = Diagnosis("brute_force", 0.85, "auth-service", 0.85, True)
        executor.execute("block_ip", diagnosis, ip="10.0.0.99")

        assert len(executor.audit_log) > 0
        ip_blocked_entries = [e for e in executor.audit_log if e["event"] == "ip_blocked"]
        assert len(ip_blocked_entries) >= 1
        assert ip_blocked_entries[0]["ip"] == "10.0.0.99"

    def test_rate_limit_creates_audit(self):
        env = SimulatedEnvironment(seed=42)
        scenario = FaultScenario(
            fault_type="ddos",
            target_service="api-gateway",
            start_time=100,
            duration=500,
            severity=0.95,
            is_security=True,
        )
        env.reset(scenario)

        executor = ActionExecutor(env)
        diagnosis = Diagnosis("ddos", 0.9, "api-gateway", 0.95, True)
        result = executor.execute("rate_limit", diagnosis)

        assert result.success
        rate_limit_entries = [e for e in executor.audit_log if e["event"] == "rate_limit_applied"]
        assert len(rate_limit_entries) >= 1
