"""Realistic time-series metrics generator for simulated microservices.

Generates 8 metrics per service with diurnal patterns, Gaussian noise,
and cross-service correlation that propagates through the dependency graph.
"""

from __future__ import annotations

import math

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

PROPAGATION_DELAY_STEPS = 2
PROPAGATION_FACTOR = 0.7


def _diurnal_factor(timestamp_seconds: np.ndarray) -> np.ndarray:
    """Compute diurnal multiplier: peak at hour 12, trough at hour 3."""
    hours = (timestamp_seconds % 86400) / 3600.0
    return 1.0 + 0.3 * np.sin(2 * math.pi * hours / 24.0 - math.pi / 2.0)


def generate_metrics(
    topology: nx.DiGraph,
    duration_seconds: int,
    interval_seconds: int = 10,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate normal (non-faulty) metrics for all services.

    Downstream services see correlated request_rate, latency, and CPU
    fluctuations from their upstream dependencies, with realistic
    propagation delay (20s) and attenuation (70% of upstream variance).

    Returns:
        Dict mapping service name to DataFrame with metric columns
        and DatetimeIndex timestamps.
    """
    rng = np.random.RandomState(seed)
    n_steps = duration_seconds // interval_seconds
    timestamps = pd.date_range(
        start="2025-01-15 00:00:00",
        periods=n_steps,
        freq=f"{interval_seconds}s",
    )
    ts_seconds = np.arange(n_steps) * interval_seconds
    diurnal = _diurnal_factor(ts_seconds)

    result: dict[str, pd.DataFrame] = {}

    generation_order = list(nx.topological_sort(topology))

    for service in generation_order:
        attrs = topology.nodes[service]
        baseline = attrs["baseline"]
        data: dict[str, np.ndarray] = {}

        for metric in METRIC_NAMES:
            base = baseline[metric]
            noise_std = base * 0.05 if base > 0 else 0.001
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
            request_deviation = (up_df["request_rate"].values - up_req_base * diurnal) / up_req_base
            delayed_deviation = np.zeros(n_steps)
            d = PROPAGATION_DELAY_STEPS
            delayed_deviation[d:] = request_deviation[:-d] if d > 0 else request_deviation
            propagated = delayed_deviation * PROPAGATION_FACTOR

            req_base = baseline["request_rate"]
            df["request_rate"] = df["request_rate"].values + req_base * propagated

            lat_base = baseline["latency_p50_ms"]
            lat99_base = baseline["latency_p99_ms"]
            latency_bump = np.abs(propagated) * lat_base * 0.3
            delayed_lat_bump = np.zeros(n_steps)
            delayed_lat_bump[d:] = latency_bump[:-d] if d > 0 else latency_bump
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
