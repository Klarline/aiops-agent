"""Q-learning agent for adaptive remediation policy.

Tabular Q-learning with epsilon-greedy exploration. Falls back
to rule-based policy when not converged.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from simulator.fast_mode import (
    FastState,
    FastSimulator,
    ACTIONS,
    FAULT_TYPES,
    SEVERITY_LEVELS,
)
from diagnosis.diagnoser import Diagnosis


class QLearningAgent:
    """Tabular Q-learning agent for remediation decisions."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 400,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        dims = (len(FAULT_TYPES), len(SEVERITY_LEVELS), 2, len(ACTIONS))
        self.q_table = np.zeros(dims)
        self.converged = False
        self.training_rewards: list[float] = []

    def choose_action(self, state: FastState, explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        return int(np.argmax(self.q_table[state.as_tuple]))

    def update(
        self,
        state: FastState,
        action_idx: int,
        reward: float,
        next_state: FastState | None = None,
    ) -> None:
        """Single-step Q-learning update."""
        current_q = self.q_table[state.as_tuple + (action_idx,)]

        if next_state is not None:
            future_q = float(np.max(self.q_table[next_state.as_tuple]))
        else:
            future_q = 0.0

        new_q = current_q + self.alpha * (reward + self.gamma * future_q - current_q)
        self.q_table[state.as_tuple + (action_idx,)] = new_q

    def decay_epsilon(self, episode: int) -> None:
        """Linear epsilon decay."""
        decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_episodes
        self.epsilon = max(self.epsilon_end, self.epsilon_start - decay_rate * episode)

    def decide(self, diagnosis: Diagnosis) -> str:
        """Map diagnosis to action using Q-table (no exploration)."""
        fault_idx = FAULT_TYPES.index(diagnosis.fault_type) if diagnosis.fault_type in FAULT_TYPES else 0
        severity_idx = min(int(diagnosis.severity * 3.99), 3)
        is_security = int(diagnosis.is_security)
        state = FastState(fault_idx, severity_idx, is_security)
        action_idx = self.choose_action(state, explore=False)
        return ACTIONS[action_idx]

    def train(self, n_episodes: int = 500, seed: int = 42) -> dict[str, Any]:
        """Train the Q-learning agent.

        Returns training stats including convergence determination.
        """
        sim = FastSimulator(seed=seed)
        episode_rewards = []

        for ep in range(n_episodes):
            state, correct_action, fault_type = sim.sample_episode()
            action_idx = self.choose_action(state, explore=True)
            action = ACTIONS[action_idx]
            reward = FastSimulator.compute_reward(action, correct_action, fault_type)
            self.update(state, action_idx, reward)
            self.decay_epsilon(ep)
            episode_rewards.append(reward)

        self.training_rewards = episode_rewards

        early_avg = np.mean(episode_rewards[:100]) if len(episode_rewards) >= 100 else 0
        late_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else 0
        improvement = (late_avg - early_avg) / abs(early_avg) if abs(early_avg) > 1e-6 else 0

        self.converged = improvement > 0.5

        return {
            "n_episodes": n_episodes,
            "early_avg_reward": float(early_avg),
            "late_avg_reward": float(late_avg),
            "improvement": float(improvement),
            "converged": self.converged,
            "final_epsilon": self.epsilon,
        }

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        self.q_table = np.load(path)
        self.converged = True
