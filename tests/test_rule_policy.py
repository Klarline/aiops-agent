"""Tests for rule-based policy decisions."""

from __future__ import annotations

from diagnosis.diagnoser import Diagnosis
from decision.rule_policy import RuleBasedPolicy


class TestRuleBasedPolicy:
    def setup_method(self):
        self.policy = RuleBasedPolicy()

    def test_memory_leak_restart(self):
        d = Diagnosis("memory_leak", 0.8, "auth-service", 0.7, False)
        assert self.policy.decide(d) == "restart_service"

    def test_cpu_saturation_scale(self):
        d = Diagnosis("cpu_saturation", 0.9, "order-service", 0.9, False)
        assert self.policy.decide(d) == "scale_out"

    def test_transaction_stall_alert(self):
        d = Diagnosis("transaction_stall", 0.9, "order-service", 1.0, False)
        assert self.policy.decide(d) == "alert_human"

    def test_brute_force_block(self):
        d = Diagnosis("brute_force", 0.85, "auth-service", 0.85, True)
        assert self.policy.decide(d) == "block_ip"

    def test_ddos_rate_limit(self):
        d = Diagnosis("ddos", 0.9, "api-gateway", 0.95, True)
        assert self.policy.decide(d) == "rate_limit"

    def test_deployment_regression_rollback(self):
        d = Diagnosis("deployment_regression", 0.75, "order-service", 0.7, False)
        assert self.policy.decide(d) == "rollback"

    def test_low_confidence_escalates(self):
        d = Diagnosis("cpu_saturation", 0.3, "order-service", 0.5, False)
        assert self.policy.decide(d) == "alert_human"

    def test_unknown_fault_escalates(self):
        d = Diagnosis("unknown", 0.6, "order-service", 0.5, False)
        assert self.policy.decide(d) == "alert_human"
