"""Tests for the LLM ReAct agent layer.

Validates:
- Agent works normally when no LLM is provided (rule-based fallback)
- Agent falls back gracefully when LLM raises an error
- Tool registry correctly wraps ML modules
- Response parsing handles well-formed and malformed LLM output
- Reasoning chain is recorded for dashboard display
"""

from __future__ import annotations

import numpy as np

from agent.agent import AIOpsAgent
from agent.tools import ToolRegistry
from agent.prompts import parse_react_response, format_system_prompt
from features.feature_extractor import extract_features_batch
from detection.ensemble import EnsembleDetector
from orchestrator.orchestrator import Orchestrator


def _train_ensemble(seed: int = 42) -> EnsembleDetector:
    from simulator.service_topology import build_topology
    from simulator.metrics_generator import generate_metrics
    topology = build_topology()
    metrics = generate_metrics(topology, 3600, 10, seed=seed)
    all_feats = []
    for svc, df in metrics.items():
        feats = extract_features_batch(df, window_size=6)
        all_feats.append(feats)
    features = np.vstack(all_feats)
    features = features[~np.isnan(features).any(axis=1)]
    ensemble = EnsembleDetector()
    ensemble.fit(features)
    return ensemble


class FaultyLLM:
    """Mock LLM that always raises an exception."""

    is_available = True

    def generate(self, system, messages, temperature=0.2):
        raise RuntimeError("Simulated LLM failure")


class MockLLM:
    """Mock LLM that returns scripted responses."""

    is_available = True

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0

    def generate(self, system, messages, temperature=0.2):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return 'THOUGHT: No more steps needed.\nACTION: done\nARGS: {}'


class TestAgentFallback:
    def test_agent_works_without_llm(self):
        """Agent works with rule-based pipeline when no LLM is provided."""
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent(llm_client=None)
        agent.set_ensemble(ensemble)

        result = orch.run_episode(agent, max_steps=200)
        assert result.detection, "Rule-based fallback should detect CPU saturation"

    def test_agent_fallback_on_llm_error(self):
        """Agent falls back to rules when LLM raises exception."""
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("brute_force_auth")

        agent = AIOpsAgent(llm_client=FaultyLLM())
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        result = orch.run_episode(agent, max_steps=200)
        assert result.detection, "Should detect even when LLM fails"

    def test_agent_no_crash_any_scenario(self):
        """Agent never crashes regardless of LLM state."""
        ensemble = _train_ensemble()
        for scenario_id in ["memory_leak_auth", "ddos_gateway", "transaction_stall_order"]:
            orch = Orchestrator(seed=42)
            orch.init_problem(scenario_id)

            agent = AIOpsAgent(llm_client=FaultyLLM())
            agent.set_ensemble(ensemble)
            agent.set_environment(orch.env)

            for _ in range(100):
                obs = orch.env.step()
                if obs is None:
                    break
                action = agent.get_action(obs)
                assert action is not None
                assert action.action in [
                    "continue_monitoring", "restart_service", "scale_out",
                    "rollback", "block_ip", "rate_limit", "alert_human",
                ]


class TestReactParsing:
    def test_parse_well_formed(self):
        text = (
            'THOUGHT: The auth-service has high error rate.\n'
            'ACTION: get_metrics\n'
            'ARGS: {"service": "auth-service"}'
        )
        thought, tool, args = parse_react_response(text)
        assert thought == "The auth-service has high error rate."
        assert tool == "get_metrics"
        assert args == {"service": "auth-service"}

    def test_parse_done(self):
        text = 'THOUGHT: All resolved.\nACTION: done\nARGS: {}'
        thought, tool, args = parse_react_response(text)
        assert "resolved" in thought.lower()
        assert tool is None
        assert args == {}

    def test_parse_malformed_returns_none(self):
        text = "This is not in the expected format at all."
        thought, tool, args = parse_react_response(text)
        assert tool is None

    def test_parse_bad_json_args(self):
        text = 'THOUGHT: checking.\nACTION: get_metrics\nARGS: {bad json}'
        thought, tool, args = parse_react_response(text)
        assert tool == "get_metrics"
        assert args == {}


class TestToolRegistry:
    def test_registry_has_all_tools(self):
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        from diagnosis.localizer import ServiceLocalizer
        from diagnosis.diagnoser import FaultDiagnoser

        registry = ToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=None,
            localizer=ServiceLocalizer(),
            diagnoser=FaultDiagnoser(),
        )
        expected = {
            "get_metrics", "explain_anomaly", "get_topology",
            "localize_root_cause", "diagnose", "check_similar_incidents",
            "restart_service", "scale_service", "rollback_deploy",
            "block_ip", "set_rate_limit", "alert_human",
        }
        assert set(registry.tools.keys()) == expected

    def test_get_metrics_tool(self):
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        from diagnosis.localizer import ServiceLocalizer
        from diagnosis.diagnoser import FaultDiagnoser

        registry = ToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=None,
            localizer=ServiceLocalizer(),
            diagnoser=FaultDiagnoser(),
        )

        for _ in range(10):
            obs = orch.env.step()
            if obs:
                registry.update_metrics_history(obs)

        result = registry.call("get_metrics", {"service": "order-service"})
        assert result.success
        assert "metrics" in result.data
        assert "anomaly_score" in result.data

    def test_unknown_tool_returns_error(self):
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        from diagnosis.localizer import ServiceLocalizer
        from diagnosis.diagnoser import FaultDiagnoser

        registry = ToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=None,
            localizer=ServiceLocalizer(),
            diagnoser=FaultDiagnoser(),
        )
        result = registry.call("nonexistent_tool", {})
        assert not result.success
        assert "error" in result.data

    def test_terminal_tools_identified(self):
        from agent.tools import TERMINAL_TOOLS
        assert "restart_service" in TERMINAL_TOOLS
        assert "block_ip" in TERMINAL_TOOLS
        assert "get_metrics" not in TERMINAL_TOOLS


class TestSystemPrompt:
    def test_format_includes_services(self):
        prompt = format_system_prompt(
            service_list=["auth-service", "order-service"],
            topology_edges=[("api-gateway", "auth-service")],
            tool_descriptions="- get_metrics: Get metrics",
        )
        assert "auth-service" in prompt
        assert "order-service" in prompt
        assert "get_metrics" in prompt


class TestReactIntegration:
    def test_mock_llm_react_loop(self):
        """End-to-end ReAct with a mock LLM that follows the protocol."""
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        mock = MockLLM([
            'THOUGHT: Let me check the root cause.\n'
            'ACTION: localize_root_cause\n'
            'ARGS: {}',
            'THOUGHT: order-service is the root cause with high CPU.\n'
            'ACTION: diagnose\n'
            'ARGS: {"service": "order-service"}',
            'THOUGHT: CPU saturation detected. Scaling out.\n'
            'ACTION: scale_service\n'
            'ARGS: {"service": "order-service"}',
        ])

        agent = AIOpsAgent(llm_client=mock)
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        acted = False
        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                acted = True
                assert action.action == "scale_out"
                assert action.target == "order-service"
                assert len(agent.reasoning_chain) > 0
                break

        assert acted, "Agent should have taken action via ReAct loop"
