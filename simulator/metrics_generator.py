"""Realistic time-series metrics generator for simulated microservices.

Generates 8 metrics per service with diurnal patterns, Gaussian noise,
and cross-service correlation that propagates through the dependency graph.

Supports distribution profiles to test agent robustness under shifted
operating conditions (different noise levels, propagation characteristics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx

from simulator.service_topology import METRIC_NAMES

METRIC_RANGES: dict[str, tuple[float, float]] = {
    "cpu_percent": (0.0, 100.0),
    "memory_percent": (0.0, 100.0),
    "latency_p50_ms": (0.0, float("inf")),
    "latency_p99_ms": (0.0, float("inf")),
    "error_rate": (0.0, 1.0),
    "request_rate": (0.0, float("inf")),
    "disk_io_percent": (0.0, 100.0),
    "transactions_per_minute": (0.0, float("inf")),
}


@dataclass(frozen=True)
class MetricsProfile:
    """Distribution parameters for metrics generation.

    Profiles control noise, diurnal patterns, and inter-service propagation
    to simulate different operating conditions.
    """

    name: str
    noise_fraction: float  # std = base * noise_fraction
    diurnal_amplitude: float  # amplitude of diurnal sine wave
    propagation_delay_min: int  # min steps of cross-service delay
    propagation_delay_max: int  # max steps (randomized per edge)
    propagation_factor_min: float  # min attenuation factor
    propagation_factor_max: float  # max attenuation factor

    @property
    def propagation_delay_fixed(self) -> bool:
        return self.propagation_delay_min == self.propagation_delay_max

    @property
    def propagation_factor_fixed(self) -> bool:
        return abs(self.propagation_factor_min - self.propagation_factor_max) < 1e-9


PROFILE_BASELINE = MetricsProfile(
    name="baseline",
    noise_fraction=0.05,
    diurnal_amplitude=0.30,
    propagation_delay_min=2,
    propagation_delay_max=2,
    propagation_factor_min=0.70,
    propagation_factor_max=0.70,
)

PROFILE_MODERATE_SHIFT = MetricsProfile(
    name="moderate_shift",
    noise_fraction=0.08,
    diurnal_amplitude=0.40,
    propagation_delay_min=1,
    propagation_delay_max=3,
    propagation_factor_min=0.55,
    propagation_factor_max=0.85,
)

PROFILE_STRESS = MetricsProfile(
    name="stress",
    noise_fraction=0.11,
    diurnal_amplitude=0.50,
    propagation_delay_min=0,
    propagation_delay_max=4,
    propagation_factor_min=0.40,
    propagation_factor_max=0.90,
)

PROFILES: dict[str, MetricsProfile] = {p.name: p for p in [PROFILE_BASELINE, PROFILE_MODERATE_SHIFT, PROFILE_STRESS]}


def get_profile(name: str) -> MetricsProfile:
    """Look up a named metrics profile."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def _diurnal_factor(timestamp_seconds: np.ndarray, amplitude: float = 0.3) -> np.ndarray:
    """Compute diurnal multiplier: peak at hour 12, trough at hour 3."""
    hours = (timestamp_seconds % 86400) / 3600.0
    return 1.0 + amplitude * np.sin(2 * math.pi * hours / 24.0 - math.pi / 2.0)


def generate_metrics(
    topology: nx.DiGraph,
    duration_seconds: int,
    interval_seconds: int = 10,
    seed: int = 42,
    profile: MetricsProfile | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate normal (non-faulty) metrics for all services.

    Downstream services see correlated request_rate, latency, and CPU
    fluctuations from their upstream dependencies, with propagation
    delay and attenuation controlled by the metrics profile.

    Args:
        profile: Distribution profile controlling noise, diurnal amplitude,
                 and propagation parameters. Defaults to PROFILE_BASELINE.

    Returns:
        Dict mapping service name to DataFrame with metric columns
        and DatetimeIndex timestamps.
    """
    if profile is None:
        profile = PROFILE_BASELINE

    rng = np.random.RandomState(seed)
    n_steps = duration_seconds // interval_seconds
    timestamps = pd.date_range(
        start="2025-01-15 00:00:00",
        periods=n_steps,
        freq=f"{interval_seconds}s",
    )
    ts_seconds = np.arange(n_steps) * interval_seconds
    diurnal = _diurnal_factor(ts_seconds, amplitude=profile.diurnal_amplitude)

    result: dict[str, pd.DataFrame] = {}

    generation_order = list(nx.topological_sort(topology))

    for service in generation_order:
        attrs = topology.nodes[service]
        baseline = attrs["baseline"]
        data: dict[str, np.ndarray] = {}

        for metric in METRIC_NAMES:
            base = baseline[metric]
            noise_std = base * profile.noise_fraction if base > 0 else 0.001
            noise = rng.randn(n_steps) * noise_std
            values = base * diurnal + noise

            lo, hi = METRIC_RANGES[metric]
            values = np.clip(values, lo, hi)
            data[metric] = values

        result[service] = pd.DataFrame(data, index=timestamps)

    for service in generation_order:
        predecessors = list(topology.predecessors(service))
        if not predecessors:
            continue

        df = result[service]
        baseline = topology.nodes[service]["baseline"]

        for upstream in predecessors:
            up_df = result[upstream]
            up_baseline = topology.nodes[upstream]["baseline"]

            up_req_base = up_baseline["request_rate"]
            if up_req_base < 1e-6:
                continue

            if profile.propagation_delay_fixed:
                d = profile.propagation_delay_min
            else:
                d = rng.randint(
                    profile.propagation_delay_min,
                    profile.propagation_delay_max + 1,
                )

            if profile.propagation_factor_fixed:
                prop_factor = profile.propagation_factor_min
            else:
                prop_factor = rng.uniform(
                    profile.propagation_factor_min,
                    profile.propagation_factor_max,
                )

            request_deviation = (up_df["request_rate"].values - up_req_base * diurnal) / up_req_base
            delayed_deviation = np.zeros(n_steps)
            if d > 0:
                delayed_deviation[d:] = request_deviation[:-d]
            else:
                delayed_deviation = request_deviation.copy()
            propagated = delayed_deviation * prop_factor

            req_base = baseline["request_rate"]
            df["request_rate"] = df["request_rate"].values + req_base * propagated

            lat_base = baseline["latency_p50_ms"]
            lat99_base = baseline["latency_p99_ms"]
            latency_bump = np.abs(propagated) * lat_base * 0.3
            delayed_lat_bump = np.zeros(n_steps)
            if d > 0:
                delayed_lat_bump[d:] = latency_bump[:-d]
            else:
                delayed_lat_bump = latency_bump.copy()
            df["latency_p50_ms"] = df["latency_p50_ms"].values + delayed_lat_bump
            df["latency_p99_ms"] = df["latency_p99_ms"].values + delayed_lat_bump * (lat99_base / max(lat_base, 1))

            cpu_base = baseline["cpu_percent"]
            cpu_bump = np.abs(propagated) * cpu_base * 0.15
            df["cpu_percent"] = np.clip(df["cpu_percent"].values + cpu_bump, 0, 100)

        for metric in METRIC_NAMES:
            lo, hi = METRIC_RANGES[metric]
            df[metric] = np.clip(df[metric].values, lo, hi)

        result[service] = df

    return result
