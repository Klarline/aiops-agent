#!/usr/bin/env python3
"""Single-command demo: Brute force attack → IP block → audit trail.

Detects brute force on auth-service, blocks source IP, creates audit entry.
Uses the security remediation path (ActionExecutor) to show block_ip + audit.

Usage: python scripts/demo_security.py
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
    from simulator.environment import SimulatedEnvironment
    from simulator.fault_injector import FaultScenario
    from diagnosis.diagnoser import Diagnosis
    from decision.action_executor import ActionExecutor

    print(f"\n{BOLD}Security Demo{RESET} — Brute force → IP block → audit trail\n")
    print("Scenario: Brute force login attack on auth-service from 10.0.0.99")
    print("Security path: detect → block IP → create audit log entry.\n")

    scenario = FaultScenario(
        fault_type="brute_force",
        target_service="auth-service",
        start_time=150,
        duration=300,
        severity=0.85,
        is_security=True,
        metadata={"source_ip": "10.0.0.99"},
    )
    env = SimulatedEnvironment(seed=SEED)
    env.reset(scenario)

    executor = ActionExecutor(env)
    diagnosis = Diagnosis("brute_force", 0.85, "auth-service", 0.85, True)

    print(f"{CYAN}Executing block_ip...{RESET} (seed={SEED})\n")

    result = executor.execute("block_ip", diagnosis, ip="10.0.0.99")

    print(f"{BOLD}Result:{RESET}")
    print("  Action:      block_ip")
    print(f"  Success:    {GREEN}{result.success}{RESET}")

    if executor.blocked_ips:
        print(f"\n{BOLD}Blocked IPs:{RESET}")
        for ip in executor.blocked_ips:
            print(f"  {ip}")

    if executor.audit_log:
        print(f"\n{BOLD}Audit log:{RESET}")
        for entry in executor.audit_log:
            event = entry.get("event", "action")
            if event == "ip_blocked":
                print(f"  [ip_blocked] IP {entry.get('ip')} blocked on {entry.get('target')}")
                print(f"    Reason: {entry.get('reason', '')[:80]}...")
            else:
                print(f"  [{event}] {entry}")

    print(f"\n{GREEN}✓ Security threat remediated with audit trail.{RESET}\n")


if __name__ == "__main__":
    main()
