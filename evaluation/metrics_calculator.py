"""Metrics calculator for AIOpsLab 4-task evaluation scores."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_task_accuracy(results: list[dict], task: str) -> float:
    """Compute accuracy for a single AIOpsLab task."""
    if not results:
        return 0.0
    return float(np.mean([r[task] for r in results]))


def compute_per_scenario_breakdown(results: list[dict]) -> dict[str, dict]:
    """Compute per-scenario accuracy for all 4 tasks."""
    scenarios: dict[str, list[dict]] = {}
    for r in results:
        sid = r["scenario"]
        if sid not in scenarios:
            scenarios[sid] = []
        scenarios[sid].append(r)

    breakdown = {}
    for sid, recs in scenarios.items():
        breakdown[sid] = {
            "detection": compute_task_accuracy(recs, "detection"),
            "localization": compute_task_accuracy(recs, "localization"),
            "diagnosis": compute_task_accuracy(recs, "diagnosis"),
            "mitigation": compute_task_accuracy(recs, "mitigation"),
        }
    return breakdown


def format_leaderboard(
    agent_results: dict[str, dict[str, float]],
) -> str:
    """Format results in AIOpsLab leaderboard style."""
    header = (
        f"{'AGENT':<20s} {'DETECTION':>10s} {'LOCALIZATION':>13s} "
        f"{'DIAGNOSIS':>10s} {'MITIGATION':>11s} {'AVERAGE':>8s}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for agent_name, scores in agent_results.items():
        avg = np.mean([scores.get("detection", 0), scores.get("localization", 0),
                        scores.get("diagnosis", 0), scores.get("mitigation", 0)])
        line = (
            f"{agent_name:<20s} "
            f"{scores.get('detection', 0):>9.0%} "
            f"{scores.get('localization', 0):>12.0%} "
            f"{scores.get('diagnosis', 0):>9.0%} "
            f"{scores.get('mitigation', 0):>10.0%} "
            f"{avg:>7.1%}"
        )
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)
