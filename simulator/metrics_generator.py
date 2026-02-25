"""Realistic time-series metrics generator for simulated microservices.

Generates 8 metrics per service with diurnal patterns, Gaussian noise,
and cross-service correlation.
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

    gateway_request_noise = rng.randn(n_steps) * 0.05

    result: dict[str, pd.DataFrame] = {}

    for service in topology.nodes:
        attrs = topology.nodes[service]
        baseline = attrs["baseline"]
        data: dict[str, np.ndarray] = {}

        for metric in METRIC_NAMES:
            base = baseline[metric]
            noise_std = base * 0.05 if base > 0 else 0.001
            noise = rng.randn(n_steps) * noise_std
            values = base * diurnal + noise

            if metric == "request_rate" and service != "api-gateway":
                gw_baseline = topology.nodes["api-gateway"]["baseline"]["request_rate"]
                ratio = base / gw_baseline if gw_baseline > 0 else 0.5
                gw_modulation = 1.0 + gateway_request_noise
                values = base * diurnal * gw_modulation * ratio / ratio + noise
                values = base * diurnal * (1.0 + gateway_request_noise) + noise

            lo, hi = METRIC_RANGES[metric]
            values = np.clip(values, lo, hi)
            data[metric] = values

        result[service] = pd.DataFrame(data, index=timestamps)

    return result
