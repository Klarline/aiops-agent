"""Train Q-learning remediation agent.

Trains for 500 episodes with epsilon-greedy exploration,
evaluates convergence, and saves Q-table.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.rl_agent import QLearningAgent


def main():
    print("Training Q-learning agent (500 episodes)...")
    agent = QLearningAgent()
    stats = agent.train(n_episodes=500, seed=42)

    print(f"\n  Episodes:          {stats['n_episodes']}")
    print(f"  Early avg reward:  {stats['early_avg_reward']:.2f}")
    print(f"  Late avg reward:   {stats['late_avg_reward']:.2f}")
    print(f"  Improvement:       {stats['improvement']:.1%}")
    print(f"  Converged:         {stats['converged']}")
    print(f"  Final epsilon:     {stats['final_epsilon']:.3f}")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "detection", "models")
    os.makedirs(models_dir, exist_ok=True)
    q_path = os.path.join(models_dir, "q_table.npy")
    agent.save(q_path)
    print(f"\n  Q-table saved to {q_path}")

    if stats["converged"]:
        print("\n  RL agent converged — will be used as primary policy")
    else:
        print("\n  RL agent did not converge — rule-based policy remains default")
        print("  (This is expected with limited episodes; documented in EVALUATION.md)")


if __name__ == "__main__":
    main()
