"""Tests for distribution-shift benchmark profiles (Phase B1).

Validates:
- Profile objects are well-formed
- Shifted profiles produce different metrics than baseline
- Benchmark runner accepts profile arguments
- Agent maintains reasonable performance under moderate shift
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.service_topology import build_topology
from simulator.metrics_generator import (
    generate_metrics,
    PROFILE_BASELINE,
    PROFILE_MODERATE_SHIFT,
    PROFILE_STRESS,
    PROFILES,
    get_profile,
)
from evaluation.benchmark_runner import run_benchmark, train_ensemble


class TestMetricsProfiles:
    def test_all_profiles_registered(self):
        assert "baseline" in PROFILES
        assert "moderate_shift" in PROFILES
        assert "stress" in PROFILES

    def test_get_profile_valid(self):
        p = get_profile("baseline")
        assert p is PROFILE_BASELINE

    def test_get_profile_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_baseline_matches_original_defaults(self):
        p = PROFILE_BASELINE
        assert p.noise_fraction == 0.05
        assert p.diurnal_amplitude == 0.30
        assert p.propagation_delay_min == 2
        assert p.propagation_delay_max == 2
        assert p.propagation_factor_min == 0.70

    def test_moderate_shift_harder_than_baseline(self):
        assert PROFILE_MODERATE_SHIFT.noise_fraction > PROFILE_BASELINE.noise_fraction
        assert PROFILE_MODERATE_SHIFT.diurnal_amplitude > PROFILE_BASELINE.diurnal_amplitude

    def test_stress_harder_than_moderate(self):
        assert PROFILE_STRESS.noise_fraction > PROFILE_MODERATE_SHIFT.noise_fraction

    def test_profile_fixed_helpers(self):
        assert PROFILE_BASELINE.propagation_delay_fixed
        assert PROFILE_BASELINE.propagation_factor_fixed
        assert not PROFILE_MODERATE_SHIFT.propagation_delay_fixed
        assert not PROFILE_MODERATE_SHIFT.propagation_factor_fixed


class TestShiftedMetricsGeneration:
    @pytest.fixture(scope="class")
    def topology(self):
        return build_topology()

    def test_baseline_generates_without_error(self, topology):
        metrics = generate_metrics(topology, 600, 10, seed=42, profile=PROFILE_BASELINE)
        assert len(metrics) == 5
        for df in metrics.values():
            assert len(df) == 60

    def test_moderate_shift_generates(self, topology):
        metrics = generate_metrics(topology, 600, 10, seed=42, profile=PROFILE_MODERATE_SHIFT)
        assert len(metrics) == 5

    def test_stress_generates(self, topology):
        metrics = generate_metrics(topology, 600, 10, seed=42, profile=PROFILE_STRESS)
        assert len(metrics) == 5

    def test_shifted_metrics_differ_from_baseline(self, topology):
        base = generate_metrics(topology, 600, 10, seed=42, profile=PROFILE_BASELINE)
        shifted = generate_metrics(topology, 600, 10, seed=42, profile=PROFILE_MODERATE_SHIFT)
        diffs = 0
        for svc in base:
            if not np.allclose(base[svc].values, shifted[svc].values, atol=1e-6):
                diffs += 1
        assert diffs > 0, "Shifted profile should produce different metrics"

    def test_noise_level_increases_variance(self, topology):
        base = generate_metrics(topology, 3600, 10, seed=42, profile=PROFILE_BASELINE)
        shifted = generate_metrics(topology, 3600, 10, seed=42, profile=PROFILE_STRESS)
        base_var = base["api-gateway"]["cpu_percent"].var()
        shifted_var = shifted["api-gateway"]["cpu_percent"].var()
        assert shifted_var > base_var, "Higher noise should produce higher variance"

    def test_none_profile_defaults_to_baseline(self, topology):
        default = generate_metrics(topology, 600, 10, seed=42, profile=None)
        explicit = generate_metrics(topology, 600, 10, seed=42, profile=PROFILE_BASELINE)
        for svc in default:
            assert np.allclose(default[svc].values, explicit[svc].values)


class TestBenchmarkWithProfiles:
    def test_benchmark_accepts_profile_strings(self):
        result = run_benchmark(
            seed=42,
            episodes_per_scenario=1,
            max_steps=100,
            train_profile="baseline",
            eval_profile="moderate_shift",
        )
        assert result["train_profile"] == "baseline"
        assert result["eval_profile"] == "moderate_shift"
        assert "aggregate" in result

    def test_benchmark_default_profiles(self):
        result = run_benchmark(seed=42, episodes_per_scenario=1, max_steps=100)
        assert result["train_profile"] == "baseline"
        assert result["eval_profile"] == "baseline"

    def test_shifted_eval_still_detects(self):
        """Agent trained on baseline should still detect faults under moderate shift."""
        result = run_benchmark(
            seed=42,
            episodes_per_scenario=2,
            max_steps=200,
            train_profile="baseline",
            eval_profile="moderate_shift",
        )
        assert result["aggregate"]["detection"] >= 0.70, "Agent should maintain >70% detection under moderate shift"

    def test_train_ensemble_with_profile(self):
        ensemble = train_ensemble(seed=42, profile=PROFILE_MODERATE_SHIFT)
        assert ensemble is not None
