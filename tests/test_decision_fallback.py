"""Tests for decision engine fallback behavior."""

from __future__ import annotations

from diagnosis.diagnoser import Diagnosis
from decision.rule_policy import RuleBasedPolicy
from decision.uncertainty_gate import UncertaintyGate
from detection.ensemble import AnomalyResult


class TestDecisionFallback:
    def test_low_confidence_falls_back_to_alert(self):
        policy = RuleBasedPolicy()
        d = Diagnosis("cpu_saturation", 0.2, "order-service", 0.5, False)
        assert policy.decide(d) == "alert_human"

    def test_high_uncertainty_escalates(self):
        gate = UncertaintyGate(uncertainty_threshold=0.3)
        result = AnomalyResult(
            is_anomalous=True,
            score=0.8,
            uncertainty=0.5,
            individual_scores={"isolation_forest": 0.9, "statistical": 0.4},
        )
        decision = gate.check(result, severity=0.7, blast_radius=2)
        assert decision.should_escalate

    def test_low_uncertainty_proceeds(self):
        gate = UncertaintyGate(uncertainty_threshold=0.4)
        result = AnomalyResult(
            is_anomalous=True,
            score=0.8,
            uncertainty=0.1,
            individual_scores={"isolation_forest": 0.8, "statistical": 0.7},
        )
        decision = gate.check(result, severity=0.5, blast_radius=1)
        assert not decision.should_escalate
        assert not decision.requires_approval

    def test_high_risk_requires_approval(self):
        gate = UncertaintyGate(uncertainty_threshold=0.4, risk_threshold=0.3)
        result = AnomalyResult(
            is_anomalous=True,
            score=0.8,
            uncertainty=0.35,
            individual_scores={"isolation_forest": 0.9, "statistical": 0.55},
        )
        decision = gate.check(result, severity=0.9, blast_radius=3)
        assert decision.requires_approval or decision.should_escalate
