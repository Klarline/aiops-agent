"""Shared data types for environment observations and action results.

Defined here (not in simulator/) so both SimulatedEnvironment and
PrometheusEnvironment can use them without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class Observation:
    """Single time-step observation returned to the agent."""

    timestamp: Any
    metrics: dict[str, dict[str, float]]
    topology: nx.DiGraph
    step_index: int


@dataclass
class ActionResult:
    """Result of executing a remediation action."""

    success: bool
    action: str
    target: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
