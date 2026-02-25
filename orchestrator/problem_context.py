"""Problem context dataclass for AIOpsLab-inspired orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProblemContext:
    """Describes the problem presented to the agent by the orchestrator."""

    scenario_id: str
    description: str
    services: list[str]
    available_apis: list[str] = field(default_factory=lambda: [
        "get_metrics",
        "get_logs",
        "get_topology",
        "restart_service",
        "scale_service",
        "rollback_deploy",
        "block_ip",
        "set_rate_limit",
        "alert_human",
    ])


@dataclass
class AgentAction:
    """Action returned by the agent."""

    action: str
    target: str = ""
    explanation: str = ""
    confidence: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Scores on the 4 AIOpsLab tasks."""

    detection: bool = False
    localization: bool = False
    diagnosis: bool = False
    mitigation: bool = False
    details: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        tasks = [self.detection, self.localization, self.diagnosis, self.mitigation]
        return sum(tasks) / len(tasks)
