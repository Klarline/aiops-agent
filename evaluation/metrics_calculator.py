"""Metrics calculator for AIOpsLab 4-task evaluation scores.

Includes standard accuracy, precision/recall/FPR, and confidence
calibration buckets for ablation/calibration reporting (Phase B2).
"""

from __future__ import annotations

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


def compute_precision_recall(results: list[dict]) -> dict[str, float]:
    """Compute precision, recall, and FPR for detection decisions.

    Every episode has a real fault, so a correct detection is a true
    positive and a missed detection is a false negative.  There are
    no true-negative episodes in the current benchmark design, so FPR
    is reported as 0.0 (placeholder for future negative-class episodes).
    """
    if not results:
        return {"precision": 0.0, "recall": 0.0, "fpr": 0.0, "f1": 0.0}

    tp = sum(1 for r in results if r["detection"])
    fn = sum(1 for r in results if not r["detection"])
    total = tp + fn

    precision = tp / total if total > 0 else 0.0
    recall = tp / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": 0.0,
        "f1": float(f1),
    }


def compute_calibration_buckets(
    results: list[dict],
    confidence_key: str = "diagnosis",
    n_buckets: int = 5,
) -> list[dict]:
    """Bin episodes by diagnosis confidence and measure actual accuracy.

    Each bucket reports: confidence range, mean predicted confidence,
    actual accuracy (diagnosis correct), and count.

    The confidence value is taken from the raw result records.  When
    records don't carry a ``confidence`` field (e.g. older runs),
    the diagnosis score (1.0 or 0.0) is used as a proxy.
    """
    pairs: list[tuple[float, bool]] = []
    for r in results:
        conf = r.get("confidence", float(r.get(confidence_key, 0)))
        correct = bool(r.get(confidence_key, False))
        pairs.append((float(conf), correct))

    if not pairs:
        return []

    pairs.sort(key=lambda x: x[0])
    bucket_size = max(1, len(pairs) // n_buckets)

    buckets: list[dict] = []
    for i in range(0, len(pairs), bucket_size):
        chunk = pairs[i : i + bucket_size]
        confs = [c for c, _ in chunk]
        accs = [int(a) for _, a in chunk]
        buckets.append(
            {
                "conf_low": float(min(confs)),
                "conf_high": float(max(confs)),
                "mean_confidence": float(np.mean(confs)),
                "accuracy": float(np.mean(accs)),
                "count": len(chunk),
            }
        )

    return buckets


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
        avg = np.mean(
            [
                scores.get("detection", 0),
                scores.get("localization", 0),
                scores.get("diagnosis", 0),
                scores.get("mitigation", 0),
            ]
        )
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
