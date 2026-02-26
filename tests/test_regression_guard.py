"""CI regression guardrails for benchmark quality.

Enforces that aggregate and per-scenario scores do not degrade beyond
a configurable tolerance (default 3 percentage points) from the
recorded baseline in expected_results.json.

The fast guard runs 3 episodes per scenario — enough to catch large
regressions without the cost of a full 13-episode run.
"""

from __future__ import annotations

import json
import os

import pytest

from evaluation.benchmark_runner import run_benchmark

EXPECTED_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "evaluation",
    "results",
    "expected_results.json",
)

REGRESSION_TOLERANCE = 0.03  # 3 percentage points


def _load_expected() -> dict:
    with open(EXPECTED_RESULTS_PATH) as f:
        return json.load(f)


class TestRegressionGuard:
    """Quick benchmark regression checks (3 episodes, ~20s)."""

    @pytest.fixture(scope="class")
    def quick_results(self):
        return run_benchmark(seed=42, episodes_per_scenario=3, max_steps=200)

    def test_aggregate_average_above_floor(self, quick_results):
        expected = _load_expected()
        baseline_avg = expected["aggregate"]["average"]
        floor = baseline_avg - REGRESSION_TOLERANCE
        actual_avg = quick_results["aggregate"]["average"]
        assert actual_avg >= floor, (
            f"Aggregate average {actual_avg:.3f} dropped below floor "
            f"{floor:.3f} (baseline {baseline_avg:.3f}, tolerance {REGRESSION_TOLERANCE})"
        )

    @pytest.mark.parametrize(
        "task",
        [
            "detection",
            "localization",
            "diagnosis",
            "mitigation",
        ],
    )
    def test_aggregate_task_above_floor(self, quick_results, task):
        expected = _load_expected()
        baseline = expected["aggregate"][task]
        floor = baseline - REGRESSION_TOLERANCE
        actual = quick_results["aggregate"][task]
        assert actual >= floor, (
            f"Aggregate {task} {actual:.3f} dropped below floor {floor:.3f} (baseline {baseline:.3f})"
        )

    def test_no_scenario_catastrophic_drop(self, quick_results):
        """No single scenario should drop more than 10 points."""
        expected = _load_expected()
        catastrophic_tolerance = 0.10
        for scenario_id, expected_scores in expected["per_scenario"].items():
            if scenario_id not in quick_results["per_scenario"]:
                continue
            actual_scores = quick_results["per_scenario"][scenario_id]
            baseline_avg = expected_scores["average"]
            actual_avg = actual_scores["average"]
            assert actual_avg >= baseline_avg - catastrophic_tolerance, (
                f"{scenario_id} average {actual_avg:.3f} catastrophically below "
                f"baseline {baseline_avg:.3f} (tolerance {catastrophic_tolerance})"
            )

    def test_detection_never_below_90(self, quick_results):
        """Detection is the most critical task — hard floor at 90%."""
        assert quick_results["aggregate"]["detection"] >= 0.90

    def test_escalation_safety_maintained(self, quick_results):
        """Mitigation should never drop below 85% (safety-critical)."""
        assert quick_results["aggregate"]["mitigation"] >= 0.85


def check_regression(results: dict, tolerance: float = REGRESSION_TOLERANCE) -> list[str]:
    """Programmatic regression check for use in scripts/CI.

    Returns a list of violation messages (empty = pass).
    """
    expected = _load_expected()
    violations = []

    for task in ["detection", "localization", "diagnosis", "mitigation", "average"]:
        baseline = expected["aggregate"][task]
        actual = results["aggregate"][task]
        if actual < baseline - tolerance:
            violations.append(
                f"Aggregate {task}: {actual:.3f} < floor {baseline - tolerance:.3f} (baseline {baseline:.3f})"
            )

    for scenario_id, expected_scores in expected["per_scenario"].items():
        if scenario_id not in results.get("per_scenario", {}):
            continue
        actual_scores = results["per_scenario"][scenario_id]
        baseline_avg = expected_scores["average"]
        actual_avg = actual_scores["average"]
        if actual_avg < baseline_avg - tolerance:
            violations.append(
                f"{scenario_id}: avg {actual_avg:.3f} < floor "
                f"{baseline_avg - tolerance:.3f} (baseline {baseline_avg:.3f})"
            )

    return violations
