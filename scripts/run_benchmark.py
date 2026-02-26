"""CLI entry point for running the full benchmark.

Usage:
    python scripts/run_benchmark.py --seed 42 --episodes 13
    python scripts/run_benchmark.py --fast  # Quick run with 3 episodes
    python scripts/run_benchmark.py --leaderboard  # Compare all agents
    python scripts/run_benchmark.py --train-profile baseline --eval-profile moderate_shift
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark_runner import run_benchmark, run_leaderboard
from evaluation.report_generator import (
    generate_report,
    generate_leaderboard_report,
    save_expected_results,
)
from simulator.metrics_generator import PROFILES


def main():
    profile_names = list(PROFILES.keys())
    parser = argparse.ArgumentParser(description="Run AIOpsLab benchmark evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=13, help="Episodes per scenario")
    parser.add_argument("--fast", action="store_true", help="Quick run (3 episodes)")
    parser.add_argument("--leaderboard", action="store_true", help="Run all agents and show comparative leaderboard")
    parser.add_argument(
        "--train-profile",
        choices=profile_names,
        default=None,
        help="Metrics profile for training data (default: baseline)",
    )
    parser.add_argument(
        "--eval-profile",
        choices=profile_names,
        default=None,
        help="Metrics profile for evaluation episodes (default: baseline)",
    )
    args = parser.parse_args()

    if args.fast:
        args.episodes = 3

    if args.train_profile or args.eval_profile:
        tp = args.train_profile or "baseline"
        ep = args.eval_profile or "baseline"
        print(f"Profiles: train={tp}, eval={ep}")

    if args.leaderboard:
        _run_leaderboard(args)
    else:
        _run_single(args)


def _run_single(args):
    print(f"Running benchmark (seed={args.seed}, episodes/scenario={args.episodes})...")
    results = run_benchmark(
        seed=args.seed,
        episodes_per_scenario=args.episodes,
        train_profile=args.train_profile,
        eval_profile=args.eval_profile,
    )

    _print_results(results)

    report = generate_report(results)
    expected_path = save_expected_results(results)

    print(f"\nResults saved to: {report['results_path']}")
    print(f"Expected results saved to: {expected_path}")


def _run_leaderboard(args):
    print(f"Running leaderboard benchmark (seed={args.seed}, episodes/scenario={args.episodes})...")
    print("=" * 60)

    leaderboard = run_leaderboard(
        seed=args.seed,
        episodes_per_scenario=args.episodes,
        train_profile=args.train_profile,
        eval_profile=args.eval_profile,
    )

    print("\n" + "=" * 60)
    print("LEADERBOARD RESULTS")
    print("=" * 60)

    for name, data in leaderboard.items():
        agg = data["aggregate"]
        print(f"\n{name}:")
        print(f"  Detection:    {agg['detection']:.0%}")
        print(f"  Localization: {agg['localization']:.0%}")
        print(f"  Diagnosis:    {agg['diagnosis']:.0%}")
        print(f"  Mitigation:   {agg['mitigation']:.0%}")
        print(f"  Average:      {agg['average']:.0%}")

    report = generate_leaderboard_report(leaderboard)
    print(f"\n{'=' * 60}")
    print(report["leaderboard"])
    print(f"\nResults saved to: {report['results_path']}")

    ml_data = leaderboard.get("ML Ensemble Agent")
    if ml_data:
        single_result = {
            **ml_data,
            "seed": args.seed,
            "episodes_per_scenario": args.episodes,
            "total_episodes": len(ml_data["raw_results"]),
        }
        expected_path = save_expected_results(single_result)
        print(f"Expected results saved to: {expected_path}")


def _print_results(results):
    print("\nAggregate Results:")
    agg = results["aggregate"]
    print(f"  Detection:    {agg['detection']:.0%}")
    print(f"  Localization: {agg['localization']:.0%}")
    print(f"  Diagnosis:    {agg['diagnosis']:.0%}")
    print(f"  Mitigation:   {agg['mitigation']:.0%}")
    print(f"  Average:      {agg['average']:.0%}")

    print("\nPer-Scenario:")
    for sid, scores in results["per_scenario"].items():
        print(
            f"  {sid:35s} Det={scores['detection']:.0%} Loc={scores['localization']:.0%} "
            f"Diag={scores['diagnosis']:.0%} Mit={scores['mitigation']:.0%}"
        )

    mttr = results.get("mttr_by_scenario", {})
    if mttr:
        print("\nMTTR (Mean Time to Remediation):")
        for sid, seconds in mttr.items():
            print(f"  {sid:35s} {seconds:.0f}s")


if __name__ == "__main__":
    main()
