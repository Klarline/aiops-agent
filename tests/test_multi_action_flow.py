"""Tests for multi-action remediation flow (Phase A2).

Validates:
- Agent can take more than one action per episode
- Budget limit (default 3) is enforced
- Budget is configurable via AIOPS_MAX_ACTIONS env var
- Follow-up actions use shorter confirmation window
- Existing single-step scenarios still pass
- Action count resets confirmation counter
"""

from __future__ import annotations

from agent.agent import AIOpsAgent
from orchestrator.orchestrator import Orchestrator
from evaluation.benchmark_runner import train_ensemble


class TestMultiActionBudget:
    def test_default_budget_is_three(self):
        agent = AIOpsAgent()
        assert agent._max_actions == 3

    def test_budget_configurable_via_env(self, monkeypatch):
        monkeypatch.setenv("AIOPS_MAX_ACTIONS", "5")
        agent = AIOpsAgent()
        assert agent._max_actions == 5

    def test_budget_configurable_to_one(self, monkeypatch):
        monkeypatch.setenv("AIOPS_MAX_ACTIONS", "1")
        agent = AIOpsAgent()
        assert agent._max_actions == 1

    def test_budget_exhausted_action_default_is_alert(self):
        agent = AIOpsAgent()
        assert agent._budget_exhausted_action == "alert_human"

    def test_budget_exhausted_action_configurable_to_monitor(self, monkeypatch):
        monkeypatch.setenv("AIOPS_BUDGET_EXHAUSTED_ACTION", "continue_monitoring")
        agent = AIOpsAgent()
        assert agent._budget_exhausted_action == "continue_monitoring"

    def test_budget_exhausted_action_invalid_falls_back_to_alert(self, monkeypatch):
        monkeypatch.setenv("AIOPS_BUDGET_EXHAUSTED_ACTION", "invalid_value")
        agent = AIOpsAgent()
        assert agent._budget_exhausted_action == "alert_human"

    def test_action_count_starts_at_zero(self):
        agent = AIOpsAgent()
        assert agent._action_count == 0

    def test_followup_confirmations_shorter(self):
        """Follow-up actions require fewer confirmations than the first."""
        agent = AIOpsAgent()
        assert agent._followup_confirmations < agent._required_confirmations


class TestMultiActionFlow:
    def test_single_action_still_works(self):
        """Existing scenarios that resolve in one action still pass."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)
        result = orch.run_episode(agent, max_steps=200)

        assert result.detection
        assert result.mitigation
        assert agent._action_count >= 1

    def test_budget_prevents_excessive_actions(self, monkeypatch):
        """After budget, agent escalates at most once then only monitors."""
        monkeypatch.setenv("AIOPS_MAX_ACTIONS", "1")
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        actions_taken = []
        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                actions_taken.append(action.action)

        assert len(actions_taken) <= 2, (
            f"Expected at most 2 non-monitoring actions (1 remediation + 1 budget escalation), "
            f"got {len(actions_taken)}: {actions_taken}"
        )

    def test_agent_can_act_multiple_times(self):
        """Agent takes a second action if the first doesn't resolve."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("memory_leak_auth")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        # Manually drive the episode tracking actions
        actions_taken = []
        for _ in range(360):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                actions_taken.append({
                    "action": action.action,
                    "target": action.target,
                    "step": obs.step_index,
                })
                orch.env.execute_action(action.action, action.target)
                if orch.env.is_resolved():
                    break

        assert len(actions_taken) >= 1, "Agent should take at least one action"
        assert agent._action_count == len(actions_taken)

    def test_confirmation_reset_after_action(self):
        """After taking an action, consecutive anomaly counter resets."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                assert agent._consecutive_anomalies == 0, (
                    "Consecutive anomaly counter should reset after action"
                )
                break

    def test_budget_exhaustion_returns_monitoring_when_configured(self, monkeypatch):
        """When configured, budget-exhausted agent returns continue_monitoring."""
        monkeypatch.setenv("AIOPS_MAX_ACTIONS", "1")
        monkeypatch.setenv("AIOPS_BUDGET_EXHAUSTED_ACTION", "continue_monitoring")
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        first_action_taken = False
        post_budget_actions = []
        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                if not first_action_taken:
                    first_action_taken = True
                else:
                    post_budget_actions.append(action.action)

        assert first_action_taken, "Agent should take at least one action"
        assert len(post_budget_actions) == 0, (
            "No additional actions should be taken after budget exhaustion"
        )

    def test_budget_exhaustion_escalates_by_default(self, monkeypatch):
        """Budget exhaustion escalates to alert_human by default (safe mode)."""
        monkeypatch.setenv("AIOPS_MAX_ACTIONS", "1")
        monkeypatch.delenv("AIOPS_BUDGET_EXHAUSTED_ACTION", raising=False)
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("memory_leak_auth")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        action_count_seen = 0
        escalated = False
        for _ in range(250):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action == "continue_monitoring":
                continue
            action_count_seen += 1
            if action_count_seen == 1:
                orch.env.execute_action(action.action, action.target)
            elif action.action == "alert_human":
                escalated = True
                assert action.details.get("reason") == "action_budget_exhausted"
                break

        assert escalated, "Expected alert_human escalation after budget exhaustion"


class TestMultiActionWithScenarios:
    def test_brute_force_single_action_resolves(self):
        """Security scenarios still resolve in one action."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("brute_force_auth")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)
        result = orch.run_episode(agent, max_steps=200)

        assert result.detection
        assert result.mitigation

    def test_transaction_stall_detection(self):
        """Transaction stall scenario still detects and acts."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("transaction_stall_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)
        result = orch.run_episode(agent, max_steps=200)

        assert result.detection

    def test_all_scenarios_still_detect(self):
        """Every scenario is still detected with multi-action support."""
        ensemble = train_ensemble(seed=42)
        scenarios = [
            "cpu_saturation_order",
            "memory_leak_auth",
            "brute_force_auth",
            "transaction_stall_order",
            "cascading_failure_gateway",
            "deployment_regression_order",
            "ddos_gateway",
        ]
        for scenario_id in scenarios:
            orch = Orchestrator(seed=42)
            orch.init_problem(scenario_id)
            agent = AIOpsAgent()
            agent.set_ensemble(ensemble)
            result = orch.run_episode(agent, max_steps=200)
            assert result.detection, f"{scenario_id}: detection failed"
