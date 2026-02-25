"""Tests for Q-learning agent."""

from __future__ import annotations

from decision.rl_agent import QLearningAgent
from simulator.fast_mode import FastSimulator, FastState, ACTIONS


class TestQLearningAgent:
    def test_q_table_updates(self):
        agent = QLearningAgent()
        state = FastState(0, 1, 0)
        initial_q = agent.q_table[state.as_tuple + (0,)]
        agent.update(state, 0, 10.0)
        assert agent.q_table[state.as_tuple + (0,)] != initial_q

    def test_epsilon_decays(self):
        agent = QLearningAgent(epsilon_start=1.0, epsilon_end=0.05)
        agent.decay_epsilon(200)
        assert agent.epsilon < 1.0
        agent.decay_epsilon(500)
        assert agent.epsilon >= 0.05

    def test_training_produces_rewards(self):
        agent = QLearningAgent()
        stats = agent.train(n_episodes=100, seed=42)
        assert len(agent.training_rewards) == 100
        assert stats["n_episodes"] == 100

    def test_trained_agent_outperforms_random(self):
        agent = QLearningAgent()
        agent.train(n_episodes=500, seed=42)

        sim = FastSimulator(seed=99)
        correct = 0
        total = 50
        for _ in range(total):
            state, correct_action, _ = sim.sample_episode()
            action_idx = agent.choose_action(state, explore=False)
            if ACTIONS[action_idx] == correct_action:
                correct += 1

        assert correct / total > 0.3, f"Agent accuracy {correct/total:.0%} too low"

    def test_fast_simulator_reward_correct(self):
        r = FastSimulator.compute_reward("block_ip", "block_ip", "brute_force")
        assert r == 10.0

    def test_fast_simulator_reward_wrong(self):
        r = FastSimulator.compute_reward("restart_service", "block_ip", "brute_force")
        assert r < 0
