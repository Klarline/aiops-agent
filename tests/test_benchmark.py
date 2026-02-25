"""Tests for benchmark evaluation system."""

from __future__ import annotations

import pytest

from evaluation.benchmark_runner import run_benchmark


@pytest.mark.slow
class TestBenchmark:
    def test_benchmark_runs(self):
        results = run_benchmark(seed=42, episodes_per_scenario=2, max_steps=100)
        assert "aggregate" in results
        assert "per_scenario" in results
        assert results["total_episodes"] > 0

    def test_benchmark_has_all_tasks(self):
        results = run_benchmark(seed=42, episodes_per_scenario=2, max_steps=100)
        agg = results["aggregate"]
        for task in ["detection", "localization", "diagnosis", "mitigation"]:
            assert task in agg
            assert 0 <= agg[task] <= 1

    def test_detection_above_minimum(self):
        results = run_benchmark(seed=42, episodes_per_scenario=3, max_steps=150)
        assert results["aggregate"]["detection"] > 0.4, "Detection too low"
