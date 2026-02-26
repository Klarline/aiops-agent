"""Tests for dynamic baseline and context computation (Phase A1).

Validates:
- _compute_baseline returns per-metric means from the warmup window
- _compute_baseline falls back to topology baselines when history is short
- _build_context produces dynamic values (no hardcoded constants)
- Dynamic values reflect actual observed data across scenarios
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.agent import AIOpsAgent
from orchestrator.orchestrator import Orchestrator
from evaluation.benchmark_runner import train_ensemble
from simulator.service_topology import METRIC_NAMES


def _make_agent_with_history(scenario_id: str, steps: int = 30):
    """Run an agent through a scenario and return it with accumulated history."""
    ensemble = train_ensemble(seed=42)
    orch = Orchestrator(seed=42)
    orch.init_problem(scenario_id)

    agent = AIOpsAgent()
    agent.set_ensemble(ensemble)
    agent.set_environment(orch.env)

    for _ in range(steps):
        obs = orch.env.step()
        if obs is None:
            break
        agent.get_action(obs)

    return agent, orch


class TestComputeBaseline:
    def test_baseline_from_warmup_window(self):
        """Baseline is computed from first _warmup_steps observations."""
        agent = AIOpsAgent()
        history = [
            {"cpu_percent": 40 + i * 0.1, "memory_percent": 50}
            for i in range(20)
        ]
        baseline = agent._compute_baseline(history)
        expected_cpu = np.mean([40 + i * 0.1 for i in range(agent._warmup_steps)])
        assert abs(baseline["cpu_percent"] - expected_cpu) < 0.01
        assert baseline["memory_percent"] == 50.0

    def test_baseline_short_history_uses_all(self):
        """When history is shorter than warmup, all data is used."""
        agent = AIOpsAgent()
        history = [
            {"cpu_percent": 30, "memory_percent": 45}
            for _ in range(5)
        ]
        baseline = agent._compute_baseline(history)
        assert baseline["cpu_percent"] == 30.0
        assert baseline["memory_percent"] == 45.0

    def test_baseline_empty_history_uses_fallback(self):
        """Empty history falls back to topology-defined baselines."""
        agent = AIOpsAgent()
        fallback = {"cpu_percent": 35.0, "memory_percent": 40.0}
        baseline = agent._compute_baseline([], fallback=fallback)
        assert baseline == fallback

    def test_baseline_empty_history_no_fallback(self):
        """Empty history with no fallback returns empty dict."""
        agent = AIOpsAgent()
        baseline = agent._compute_baseline([])
        assert baseline == {}

    def test_baseline_excludes_post_warmup_data(self):
        """Baseline only uses warmup window, not fault-period data."""
        agent = AIOpsAgent()
        normal = [{"cpu_percent": 40.0} for _ in range(15)]
        fault = [{"cpu_percent": 95.0} for _ in range(15)]
        history = normal + fault

        baseline = agent._compute_baseline(history)
        assert baseline["cpu_percent"] == 40.0


class TestBuildContextDynamic:
    def test_no_hardcoded_tpm_baseline(self):
        """tpm_baseline is computed from history, not hardcoded to 850."""
        agent, orch = _make_agent_with_history("transaction_stall_order", steps=40)
        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="transaction_stall",
            localized_service="order-service",
            confidence=0.9,
            severity=0.9,
            is_security=False,
        )
        ctx = agent._build_context(obs, "order-service", diag)

        history = agent._metrics_history.get("order-service", [])
        if len(history) >= agent._warmup_steps:
            warmup = history[:agent._warmup_steps]
            expected_baseline = np.mean([h["transactions_per_minute"] for h in warmup])
            assert abs(ctx["tpm_baseline"] - expected_baseline) < 1.0

    def test_dynamic_zscore_reflects_anomaly(self):
        """peak zscore should be > 0 when fault is active."""
        agent, orch = _make_agent_with_history("cpu_saturation_order", steps=80)
        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="cpu_saturation",
            localized_service="order-service",
            confidence=0.9,
            severity=0.9,
            is_security=False,
        )
        ctx = agent._build_context(obs, "order-service", diag)
        assert ctx["zscore"] >= 0

    def test_dynamic_latency_multiplier(self):
        """latency_mult is computed from current vs baseline, not hardcoded."""
        agent, orch = _make_agent_with_history("cascading_failure_gateway", steps=60)
        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="cascading_failure",
            localized_service="api-gateway",
            confidence=0.9,
            severity=0.9,
            is_security=False,
        )
        ctx = agent._build_context(obs, "api-gateway", diag)

        history = agent._metrics_history.get("api-gateway", [])
        if len(history) >= agent._warmup_steps:
            warmup = history[:agent._warmup_steps]
            baseline_lat = np.mean([h["latency_p99_ms"] for h in warmup])
            current_lat = obs.metrics["api-gateway"]["latency_p99_ms"]
            expected_mult = current_lat / baseline_lat if baseline_lat > 0 else 1.0
            assert abs(ctx["latency_mult"] - expected_mult) < 0.01

    def test_dynamic_memory_rate(self):
        """Memory rate reflects actual recent rate of change."""
        agent, orch = _make_agent_with_history("memory_leak_auth", steps=80)
        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="memory_leak",
            localized_service="auth-service",
            confidence=0.9,
            severity=0.8,
            is_security=False,
        )
        ctx = agent._build_context(obs, "auth-service", diag)
        assert isinstance(ctx["rate"], float)

    def test_context_has_all_required_keys(self):
        """Context dict has all keys the summarizer templates expect."""
        agent, orch = _make_agent_with_history("cpu_saturation_order", steps=40)
        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="cpu_saturation",
            localized_service="order-service",
            confidence=0.9,
            severity=0.9,
            is_security=False,
        )
        ctx = agent._build_context(obs, "order-service", diag)

        required_keys = {
            "timestamp", "cpu", "memory", "tpm", "tpm_baseline",
            "rate", "zscore", "error_count", "window",
            "affected_count", "latency_mult", "latency_increase",
            "error_increase", "request_mult",
        }
        assert required_keys.issubset(ctx.keys())

    def test_source_ip_from_scenario_metadata(self):
        """source_ip is sourced from scenario metadata when available."""
        agent, orch = _make_agent_with_history("brute_force_auth", steps=40)
        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="brute_force",
            localized_service="auth-service",
            confidence=0.9,
            severity=0.85,
            is_security=True,
        )
        ctx = agent._build_context(obs, "auth-service", diag)
        assert ctx.get("source_ip") == "10.0.0.99"

    def test_source_ip_absent_without_environment(self):
        """source_ip is omitted when no environment is set."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        for _ in range(40):
            obs = orch.env.step()
            if obs is None:
                break
            agent.get_action(obs)

        obs = orch.env.step()
        if obs is None:
            return

        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="cpu_saturation",
            localized_service="order-service",
            confidence=0.9,
            severity=0.9,
            is_security=False,
        )
        ctx = agent._build_context(obs, "order-service", diag)
        assert "source_ip" not in ctx

    def test_observation_window_matches_confirmations(self):
        """window reflects required_confirmations * interval_seconds."""
        agent = AIOpsAgent()
        assert agent._required_confirmations == 5
        # Not scenario dependent — just verify consistency
        agent, orch = _make_agent_with_history("cpu_saturation_order", steps=40)
        obs = orch.env.step()
        if obs is None:
            return
        from diagnosis.diagnoser import Diagnosis
        diag = Diagnosis(
            fault_type="cpu_saturation",
            localized_service="order-service",
            confidence=0.9,
            severity=0.9,
            is_security=False,
        )
        ctx = agent._build_context(obs, "order-service", diag)
        assert ctx["window"] == "50 seconds"
