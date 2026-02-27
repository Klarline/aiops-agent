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
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run across seeds 42-46 and report mean ± std (robustness check)",
    )
    args = parser.parse_args()

    if args.fast:
        args.episodes = 3

    if args.train_profile or args.eval_profile:
        tp = args.train_profile or "baseline"
        ep = args.eval_profile or "baseline"
        print(f"Profiles: train={tp}, eval={ep}")

    if args.multi_seed:
        _run_multi_seed(args)
    elif args.leaderboard:
        _run_leaderboard(args)
    else:
        _run_single(args)


def _run_multi_seed(args):
    """Run benchmark across seeds 42-46 and report mean ± std."""
    import numpy as np

    seeds = [42, 43, 44, 45, 46]
    print(f"Running multi-seed robustness check (seeds={seeds}, episodes/scenario={args.episodes})...")
    print("=" * 60)

    all_agg: list[dict] = []
    for s in seeds:
        results = run_benchmark(
            seed=s,
            episodes_per_scenario=args.episodes,
            train_profile=args.train_profile,
            eval_profile=args.eval_profile,
        )
        all_agg.append(results["aggregate"])
        print(f"  Seed {s}: avg={results['aggregate']['average']:.0%}")

    det = np.array([a["detection"] for a in all_agg])
    loc = np.array([a["localization"] for a in all_agg])
    diag = np.array([a["diagnosis"] for a in all_agg])
    mit = np.array([a["mitigation"] for a in all_agg])
    avg = np.array([a["average"] for a in all_agg])

    print("\n" + "=" * 60)
    print("ROBUSTNESS (mean ± std across 5 seeds)")
    print("=" * 60)
    print(f"  Detection:    {det.mean():.0%} ± {det.std():.0%}")
    print(f"  Localization: {loc.mean():.0%} ± {loc.std():.0%}")
    print(f"  Diagnosis:    {diag.mean():.0%} ± {diag.std():.0%}")
    print(f"  Mitigation:   {mit.mean():.0%} ± {mit.std():.0%}")
    print(f"  Average:      {avg.mean():.0%} ± {avg.std():.0%}")


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
