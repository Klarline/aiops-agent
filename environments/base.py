"""Abstract base class for all environments (simulated and live).

Both SimulatedEnvironment and PrometheusEnvironment implement this interface,
allowing the orchestrator and agent to work with either without modification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
import pandas as pd

from environments.types import ActionResult, Observation


class BaseEnvironment(ABC):
    """Common interface for SimulatedEnvironment and PrometheusEnvironment."""

    # Subclasses must set these in __init__:
    #   self.current_step: int         — increments on each step()
    #   self.n_steps: int | float      — total steps (float('inf') for live mode)
    #   self.metrics_data: Any         — None means "not started"; non-None means active
    #   self.scenario: Any = None      — FaultScenario in sim mode, None in live mode

    scenario: Any = None

    @abstractmethod
    def step(self) -> Observation | None: ...

    @abstractmethod
    def get_current_metrics(self) -> dict[str, dict[str, float]]: ...

    @abstractmethod
    def get_metrics_history(self, service: str, lookback_steps: int = 30) -> pd.DataFrame: ...

    @abstractmethod
    def execute_action(self, action: str, target: str, **kwargs: Any) -> ActionResult: ...

    @abstractmethod
    def get_topology(self) -> nx.DiGraph: ...

    @abstractmethod
    def get_ground_truth(self) -> Any:
        """Return ground-truth FaultScenario in sim mode, None in live mode."""
        ...

    @abstractmethod
    def is_resolved(self) -> bool: ...
