"""AIOpsLab-inspired orchestrator implementing the Agent-Cloud Interface.

Mediates between the AI agent and the simulated environment, handling
problem setup, action execution, and 4-task evaluation.
"""

from __future__ import annotations

from environments.base import BaseEnvironment
from environments.types import Observation
from simulator.environment import SimulatedEnvironment
from orchestrator.scenario_registry import get_scenario
from orchestrator.problem_context import (
    ProblemContext,
    AgentAction,
    EvaluationResult,
)


class Orchestrator:
    """Lightweight AIOpsLab-inspired orchestrator."""

    def __init__(
        self,
        seed: int = 42,
        eval_profile=None,
        env: BaseEnvironment | None = None,
    ):
        self.env: BaseEnvironment = env if env is not None else SimulatedEnvironment(seed=seed, profile=eval_profile)
        self.history: list[dict] = []
        self._agent_detected: bool = False
        self._agent_localized: str = ""
        self._agent_diagnosed: str = ""
        self._agent_mitigated: bool = False

    def init_live(self) -> ProblemContext:
        """Initialize for live monitoring — no scenario injection, no ground truth."""
        self.history = []
        self._agent_detected = False
        self._agent_localized = ""
        self._agent_diagnosed = ""
        self._agent_mitigated = False

        services = list(self.env.get_topology().nodes)
        return ProblemContext(
            scenario_id="live_monitoring",
            description="Live monitoring mode — Prometheus data source. No injected faults.",
            services=services,
        )

    def init_problem(self, scenario_id: str) -> ProblemContext:
        """Load scenario from registry, reset environment, return context."""
        scenario = get_scenario(scenario_id)
        self.env.reset(scenario)
        self.history = []
        self._agent_detected = False
        self._agent_localized = ""
        self._agent_diagnosed = ""
        self._agent_mitigated = False

        services = list(self.env.topology.nodes)
        return ProblemContext(
            scenario_id=scenario_id,
            description=scenario.description,
            services=services,
        )

    def step(self, agent_action: AgentAction) -> Observation | None:
        """Execute the agent's action and return the next observation."""
        if agent_action.action != "continue_monitoring":
            self._agent_detected = True
            self._agent_localized = agent_action.target
            self._agent_diagnosed = agent_action.details.get("diagnosis", "")

            result = self.env.execute_action(
                agent_action.action,
                agent_action.target,
                **agent_action.details,
            )
            self._agent_mitigated = self.env.is_resolved()

            self.history.append({
                "action": agent_action,
                "result": result,
                "step": self.env.current_step,
            })

        return self.env.step()

    def evaluate(self) -> EvaluationResult:
        """Score the agent on 4 AIOpsLab tasks against ground truth."""
        gt = self.env.get_ground_truth()
        if gt is None:
            return EvaluationResult()

        detection = self._agent_detected
        localization = self._agent_localized == gt.target_service
        diagnosis = self._agent_diagnosed == gt.fault_type
        mitigation = self._agent_mitigated

        return EvaluationResult(
            detection=detection,
            localization=localization,
            diagnosis=diagnosis,
            mitigation=mitigation,
            details={
                "expected_service": gt.target_service,
                "agent_localized": self._agent_localized,
                "expected_fault": gt.fault_type,
                "agent_diagnosed": self._agent_diagnosed,
            },
        )

    def run_episode(self, agent, max_steps: int = 360) -> EvaluationResult:
        """Convenience: run a full episode with the given agent."""
        for _ in range(max_steps):
            obs = self.env.step()
            if obs is None:
                break

            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                self._agent_detected = True
                self._agent_localized = action.target
                self._agent_diagnosed = action.details.get("diagnosis", "")

                self.env.execute_action(
                    action.action,
                    action.target,
                    **action.details,
                )
                self._agent_mitigated = self.env.is_resolved()
                self.history.append({
                    "action": action,
                    "step": self.env.current_step,
                })
                if self.env.is_resolved():
                    break

        return self.evaluate()
