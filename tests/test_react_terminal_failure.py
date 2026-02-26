"""Tests for terminal action failure handling in the ReAct loop (Phase A3).

Validates:
- Failed terminal tool call does not end episode as successful mitigation
- Failure is fed back to LLM for alternate action selection
- Per-type guardrail: max 2 attempts of the same terminal action type
- Total guardrail: max 3 terminal attempts per episode
- Guardrail breach triggers forced alert_human escalation
- Failure path is visible in reasoning chain/log
"""

from __future__ import annotations

from agent.agent import AIOpsAgent
from agent.tools import ToolRegistry, ToolResult
from orchestrator.orchestrator import Orchestrator
from evaluation.benchmark_runner import train_ensemble


class FailingToolRegistry(ToolRegistry):
    """Tool registry where specific terminal tools always fail."""

    def __init__(self, *args, fail_tools: set[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_tools = fail_tools or set()

    def call(self, name: str, args: dict) -> ToolResult:
        if name in self._fail_tools:
            return ToolResult(name, False, {
                "error": f"Simulated failure for {name}",
                "action": name,
                "success": False,
            })
        return super().call(name, args)


class MockLLMForFailure:
    """Mock LLM that attempts a failing action, then escalates."""

    is_available = True

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0

    def generate(self, system, messages, temperature=0.2):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return (
            'THOUGHT: All attempts failed, escalating.\n'
            'ACTION: alert_human\n'
            'ARGS: {"message": "Remediation failed"}'
        )


class TestTerminalFailureFeedback:
    def test_failed_terminal_does_not_count_as_success(self):
        """A failed terminal action should not mark the episode as mitigated."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        responses = [
            'THOUGHT: Checking root cause.\n'
            'ACTION: localize_root_cause\n'
            'ARGS: {}',
            'THOUGHT: CPU saturation on order-service, scaling out.\n'
            'ACTION: scale_service\n'
            'ARGS: {"service": "order-service"}',
        ]
        mock_llm = MockLLMForFailure(responses)

        agent = AIOpsAgent(llm_client=mock_llm)
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        agent._tool_registry = FailingToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=agent.explainer,
            localizer=agent.localizer,
            diagnoser=agent.diagnoser,
            fail_tools={"scale_service"},
        )

        acted = False
        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                acted = True
                assert action.action == "alert_human", (
                    f"Expected escalation after failed scale, got {action.action}"
                )
                break

        assert acted, "Agent should have acted (escalated)"

    def test_failure_fed_back_to_llm(self):
        """After a terminal failure, the LLM sees the failure message."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        call_log = []

        class SpyLLM:
            is_available = True
            _call_count = 0
            _responses = [
                'THOUGHT: Investigating.\n'
                'ACTION: localize_root_cause\n'
                'ARGS: {}',
                'THOUGHT: Scaling order-service.\n'
                'ACTION: scale_service\n'
                'ARGS: {"service": "order-service"}',
                'THOUGHT: Scale failed. Restarting instead.\n'
                'ACTION: restart_service\n'
                'ARGS: {"service": "order-service"}',
            ]

            def generate(self, system, messages, temperature=0.2):
                call_log.append(messages[-1] if messages else {})
                if self._call_count < len(self._responses):
                    resp = self._responses[self._call_count]
                    self._call_count += 1
                    return resp
                return 'THOUGHT: Escalating.\nACTION: alert_human\nARGS: {}'

        spy_llm = SpyLLM()
        agent = AIOpsAgent(llm_client=spy_llm)
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        agent._tool_registry = FailingToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=agent.explainer,
            localizer=agent.localizer,
            diagnoser=agent.diagnoser,
            fail_tools={"scale_service"},
        )

        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                break

        failure_messages = [
            m for m in call_log
            if isinstance(m, dict) and "ACTION FAILED" in m.get("content", "")
        ]
        assert len(failure_messages) > 0, (
            "LLM should receive an ACTION FAILED message after terminal failure"
        )

    def test_failure_visible_in_reasoning_chain(self):
        """Reasoning chain records the failed tool observation."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        responses = [
            'THOUGHT: Investigating.\nACTION: localize_root_cause\nARGS: {}',
            'THOUGHT: Scaling.\nACTION: scale_service\n'
            'ARGS: {"service": "order-service"}',
        ]
        mock_llm = MockLLMForFailure(responses)

        agent = AIOpsAgent(llm_client=mock_llm)
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        agent._tool_registry = FailingToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=agent.explainer,
            localizer=agent.localizer,
            diagnoser=agent.diagnoser,
            fail_tools={"scale_service"},
        )

        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                break

        chain = agent.reasoning_chain
        failure_observations = [
            entry for entry in chain
            if "observation" in entry
            and isinstance(entry["observation"], dict)
            and entry["observation"].get("success") is False
        ]
        assert len(failure_observations) > 0, (
            "Reasoning chain should contain the failed tool observation"
        )


class TestTerminalGuardrails:
    def test_per_type_limit_triggers_escalation(self):
        """Hitting max attempts for one action type forces escalation."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        responses = [
            'THOUGHT: Investigating.\nACTION: localize_root_cause\nARGS: {}',
            'THOUGHT: Scaling.\nACTION: scale_service\n'
            'ARGS: {"service": "order-service"}',
            'THOUGHT: Retry scale.\nACTION: scale_service\n'
            'ARGS: {"service": "order-service"}',
        ]
        mock_llm = MockLLMForFailure(responses)

        agent = AIOpsAgent(llm_client=mock_llm)
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        agent._tool_registry = FailingToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=agent.explainer,
            localizer=agent.localizer,
            diagnoser=agent.diagnoser,
            fail_tools={"scale_service"},
        )

        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                assert action.action == "alert_human"
                assert "scale_service" in action.explanation
                return

        raise AssertionError("Agent should have escalated after 2 failed scale_service")

    def test_total_limit_triggers_escalation(self):
        """Hitting total terminal attempts across types forces escalation."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        responses = [
            'THOUGHT: Investigating.\nACTION: localize_root_cause\nARGS: {}',
            'THOUGHT: Scaling.\nACTION: scale_service\n'
            'ARGS: {"service": "order-service"}',
            'THOUGHT: Try restart.\nACTION: restart_service\n'
            'ARGS: {"service": "order-service"}',
            'THOUGHT: Try rollback.\nACTION: rollback_deploy\n'
            'ARGS: {"service": "order-service"}',
        ]
        mock_llm = MockLLMForFailure(responses)

        agent = AIOpsAgent(llm_client=mock_llm)
        agent.set_ensemble(ensemble)
        agent.set_environment(orch.env)

        agent._tool_registry = FailingToolRegistry(
            env=orch.env,
            ensemble=ensemble,
            explainer=agent.explainer,
            localizer=agent.localizer,
            diagnoser=agent.diagnoser,
            fail_tools={"scale_service", "restart_service", "rollback_deploy"},
        )

        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                assert action.action == "alert_human"
                assert agent._total_terminal_attempts <= agent._max_terminal_total
                return

        raise AssertionError("Agent should have escalated after 3 total failures")

    def test_should_force_escalation_logic(self):
        """Unit test for _should_force_escalation thresholds."""
        agent = AIOpsAgent()

        assert not agent._should_force_escalation("scale_service")

        agent._terminal_attempts_by_type["scale_service"] = 2
        agent._total_terminal_attempts = 2
        assert agent._should_force_escalation("scale_service")

        agent2 = AIOpsAgent()
        agent2._total_terminal_attempts = 3
        assert agent2._should_force_escalation("restart_service")

        agent3 = AIOpsAgent()
        agent3._terminal_attempts_by_type["restart_service"] = 1
        agent3._total_terminal_attempts = 1
        assert not agent3._should_force_escalation("restart_service")
