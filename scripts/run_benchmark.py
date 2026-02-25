"""CLI entry point for running the full benchmark.

Usage:
    python scripts/run_benchmark.py --seed 42 --episodes 13
    python scripts/run_benchmark.py --fast  # Quick run with 3 episodes
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark_runner import run_benchmark
from evaluation.report_generator import generate_report, save_expected_results


def main():
    parser = argparse.ArgumentParser(description="Run AIOpsLab benchmark evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=13, help="Episodes per scenario")
    parser.add_argument("--fast", action="store_true", help="Quick run (3 episodes)")
    args = parser.parse_args()

    if args.fast:
        args.episodes = 3

    print(f"Running benchmark (seed={args.seed}, episodes/scenario={args.episodes})...")
    results = run_benchmark(seed=args.seed, episodes_per_scenario=args.episodes)

    print(f"\nAggregate Results:")
    agg = results["aggregate"]
    print(f"  Detection:    {agg['detection']:.0%}")
    print(f"  Localization: {agg['localization']:.0%}")
    print(f"  Diagnosis:    {agg['diagnosis']:.0%}")
    print(f"  Mitigation:   {agg['mitigation']:.0%}")
    print(f"  Average:      {agg['average']:.0%}")

    print(f"\nPer-Scenario:")
    for sid, scores in results["per_scenario"].items():
        print(
            f"  {sid:35s} Det={scores['detection']:.0%} Loc={scores['localization']:.0%} "
            f"Diag={scores['diagnosis']:.0%} Mit={scores['mitigation']:.0%}"
        )

    report = generate_report(results)
    expected_path = save_expected_results(results)

    print(f"\nResults saved to: {report['results_path']}")
    print(f"Expected results saved to: {expected_path}")


if __name__ == "__main__":
    main()
