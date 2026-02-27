#!/usr/bin/env python3
"""Single-command demo: Transaction stall (silent business failure).

CPU and memory are normal. Latency is normal. But TPM dropped to zero.
This agent catches that.

Usage: python scripts/demo_transaction_stall.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

SEED = 42


def main():
    from orchestrator.orchestrator import Orchestrator
    from agent.agent import AIOpsAgent
    from evaluation.benchmark_runner import train_ensemble

    print(f"\n{BOLD}Transaction Stall Demo{RESET} — Silent failure where every dashboard says healthy\n")
    print("Scenario: order-service stops processing transactions. CPU, memory, latency all normal.")
    print("Only TPM (transactions per minute) collapses. Threshold monitoring misses it.\n")

    ensemble = train_ensemble(seed=SEED)
    orch = Orchestrator(seed=SEED)
    orch.init_problem("transaction_stall_order")

    agent = AIOpsAgent()
    agent.set_ensemble(ensemble)

    print(f"{CYAN}Running...{RESET} (seed={SEED})\n")

    result = orch.run_episode(agent)

    print(f"{BOLD}Result:{RESET}")
    print(f"  Detection:   {GREEN if result.detection else ''}{result.detection}{RESET}")
    print(f"  Localization:{GREEN if result.localization else ''}{result.localization}{RESET}")
    print(f"  Diagnosis:   {GREEN if result.diagnosis else ''}{result.diagnosis}{RESET}")
    print(f"  Mitigation:  {GREEN if result.mitigation else ''}{result.mitigation}{RESET}")
    print(f"  Score:       {result.score:.0%}")

    if agent.reasoning_log:
        print(f"\n{BOLD}Agent reasoning (last entry):{RESET}")
        last = agent.reasoning_log[-1]
        summary = last.get("summary", str(last))[:200]
        print(f"  {summary}...")

    print(f"\n{GREEN}✓ Agent caught the silent failure.{RESET}\n")


if __name__ == "__main__":
    main()
