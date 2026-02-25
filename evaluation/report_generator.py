"""Report generator for benchmark results.

Produces JSON results, plots, and markdown summaries.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from evaluation.metrics_calculator import format_leaderboard, compute_per_scenario_breakdown


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def generate_report(benchmark_results: dict[str, Any]) -> dict[str, Any]:
    """Generate full evaluation report from benchmark results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    serializable = _make_serializable(benchmark_results)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    leaderboard = _build_leaderboard(benchmark_results)
    breakdown = compute_per_scenario_breakdown(benchmark_results["raw_results"])

    try:
        _generate_plots(benchmark_results, breakdown)
    except Exception:
        pass

    markdown = _generate_markdown(benchmark_results, leaderboard, breakdown)

    return {
        "results_path": results_path,
        "leaderboard": leaderboard,
        "markdown": markdown,
    }


def save_expected_results(benchmark_results: dict[str, Any]) -> str:
    """Save expected results for reproducibility testing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "expected_results.json")
    expected = {
        "seed": benchmark_results["seed"],
        "aggregate": benchmark_results["aggregate"],
        "per_scenario": benchmark_results["per_scenario"],
    }
    with open(path, "w") as f:
        json.dump(expected, f, indent=2)
    return path


def _build_leaderboard(results: dict) -> str:
    agents = {
        "Expert Policy": results["aggregate"],
    }
    return format_leaderboard(agents)


def _generate_plots(results: dict, breakdown: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    scenarios = list(breakdown.keys())
    tasks = ["detection", "localization", "diagnosis", "mitigation"]
    data = np.array([[breakdown[s][t] for t in tasks] for s in scenarios])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data, annot=True, fmt=".0%", cmap="RdYlGn",
        xticklabels=tasks, yticklabels=[s.replace("_", " ") for s in scenarios],
        ax=ax,
    )
    ax.set_title("Per-Scenario Accuracy (AIOpsLab Format)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def _generate_markdown(results: dict, leaderboard: str, breakdown: dict) -> str:
    lines = [
        "# Evaluation Results",
        "",
        "## Aggregate Scores (AIOpsLab Leaderboard Format)",
        "",
        "```",
        leaderboard,
        "```",
        "",
        "## Per-Scenario Breakdown",
        "",
    ]
    for sid, scores in breakdown.items():
        det = scores["detection"]
        loc = scores["localization"]
        diag = scores["diagnosis"]
        mit = scores["mitigation"]
        lines.append(
            f"  {sid:35s} Det={det:.0%} Loc={loc:.0%} Diag={diag:.0%} Mit={mit:.0%}"
        )

    lines.extend([
        "",
        "## Configuration",
        f"- Seed: {results['seed']}",
        f"- Episodes per scenario: {results['episodes_per_scenario']}",
        f"- Total episodes: {results['total_episodes']}",
    ])

    return "\n".join(lines)


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
