"""Fault injection for simulated microservice environment.

Supports 8 fault types: 4 operational, 3 security, 1 business-logic.
Each fault modifies metrics in a specific, detectable pattern while
preserving ground truth for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from simulator.service_topology import get_downstream


@dataclass
class FaultScenario:
    """Describes a fault to be injected into the simulated environment."""

    fault_type: str
    target_service: str
    start_time: float
    duration: float
    severity: float  # 0.0 – 1.0
    is_security: bool = False
    propagation: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def _time_mask(
    index: pd.DatetimeIndex,
    start_time: float,
    duration: float,
) -> np.ndarray:
    """Boolean mask for the fault window."""
    origin = index[0]
    elapsed = (index - origin).total_seconds().values
    return (elapsed >= start_time) & (elapsed < start_time + duration)


def _inject_memory_leak(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Linear memory increase ~0.5 %/min on target; other metrics unchanged."""
    target = scenario.target_service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    fault_indices = np.where(mask)[0]
    if len(fault_indices) == 0:
        metrics[target] = df
        return metrics

    elapsed_minutes = np.arange(len(fault_indices)) * 10.0 / 60.0
    leak_rate = 0.5 * scenario.severity
    increase = leak_rate * elapsed_minutes
    df.loc[df.index[mask], "memory_percent"] += increase
    df["memory_percent"] = df["memory_percent"].clip(0, 100)
    metrics[target] = df
    return metrics


def _inject_cpu_saturation(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Sudden CPU jump to 85-98 % on target, slight latency increase on downstream."""
    target = scenario.target_service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    rng = np.random.RandomState(99)

    cpu_target = 85.0 + scenario.severity * 13.0  # 85-98
    noise = rng.randn(int(mask.sum())) * 2.0
    df.loc[df.index[mask], "cpu_percent"] = cpu_target + noise
    df["cpu_percent"] = df["cpu_percent"].clip(0, 100)
    metrics[target] = df

    prop_delay_steps = int(rng.uniform(1, 3))
    downstream = get_downstream(topology, target)
    for ds in downstream:
        ds_df = metrics[ds].copy()
        ds_mask = _time_mask(ds_df.index, scenario.start_time + prop_delay_steps * 10, scenario.duration)
        latency_bump = 20.0 * scenario.severity
        ds_df.loc[ds_df.index[ds_mask], "latency_p50_ms"] += latency_bump
        ds_df.loc[ds_df.index[ds_mask], "latency_p99_ms"] += latency_bump * 2.5
        metrics[ds] = ds_df

    return metrics


def _inject_brute_force(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Auth-service error_rate spikes; request_rate increases from attack traffic."""
    target = scenario.target_service  # should be auth-service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    rng = np.random.RandomState(77)

    error_spike = 0.4 + scenario.severity * 0.4  # 0.4 - 0.8
    noise = rng.randn(int(mask.sum())) * 0.05
    df.loc[df.index[mask], "error_rate"] = error_spike + noise
    df["error_rate"] = df["error_rate"].clip(0, 1)

    request_bump = 200.0 * scenario.severity
    df.loc[df.index[mask], "request_rate"] += request_bump

    metrics[target] = df
    return metrics


def _inject_transaction_stall(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """TPM drops to near-zero on order-service; ALL other metrics stay normal.

    This is the critical 'silent failure' scenario — infrastructure looks
    healthy but the application has stopped processing transactions.
    """
    target = scenario.target_service  # should be order-service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    rng = np.random.RandomState(55)

    residual = rng.uniform(0, 5, size=int(mask.sum()))
    df.loc[df.index[mask], "transactions_per_minute"] = residual
    metrics[target] = df
    return metrics


def _inject_cascading_failure(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Multi-service failure propagating downstream with delays."""
    target = scenario.target_service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    rng = np.random.RandomState(88)

    df.loc[df.index[mask], "latency_p50_ms"] *= (2.0 + scenario.severity * 3.0)
    df.loc[df.index[mask], "latency_p99_ms"] *= (3.0 + scenario.severity * 5.0)
    df.loc[df.index[mask], "error_rate"] = np.clip(
        df.loc[df.index[mask], "error_rate"] + 0.15 * scenario.severity, 0, 1
    )
    metrics[target] = df

    delay_s = 0
    current_services = [target]
    for _ in range(3):
        next_services = []
        for svc in current_services:
            next_services.extend(get_downstream(topology, svc))
        if not next_services:
            break
        delay_s += int(rng.uniform(10, 30))
        for ds in set(next_services):
            ds_df = metrics[ds].copy()
            ds_mask = _time_mask(ds_df.index, scenario.start_time + delay_s, scenario.duration)
            ds_df.loc[ds_df.index[ds_mask], "latency_p50_ms"] *= (1.5 + scenario.severity)
            ds_df.loc[ds_df.index[ds_mask], "latency_p99_ms"] *= (2.0 + scenario.severity * 2)
            ds_df.loc[ds_df.index[ds_mask], "error_rate"] = np.clip(
                ds_df.loc[ds_df.index[ds_mask], "error_rate"] + 0.1 * scenario.severity, 0, 1
            )
            metrics[ds] = ds_df
        current_services = list(set(next_services))

    return metrics


def _inject_deployment_regression(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Post-deploy metric shift: latency jump + error rate increase."""
    target = scenario.target_service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)

    df.loc[df.index[mask], "latency_p50_ms"] *= (1.5 + scenario.severity)
    df.loc[df.index[mask], "latency_p99_ms"] *= (2.0 + scenario.severity * 2)
    df.loc[df.index[mask], "error_rate"] = np.clip(
        df.loc[df.index[mask], "error_rate"] + 0.08 * scenario.severity, 0, 1
    )
    metrics[target] = df
    return metrics


def _inject_anomalous_access(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Unusual endpoint patterns — subtle error_rate and latency changes."""
    target = scenario.target_service
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    rng = np.random.RandomState(66)

    noise = rng.randn(int(mask.sum())) * 0.03
    df.loc[df.index[mask], "error_rate"] += 0.05 * scenario.severity + noise
    df["error_rate"] = df["error_rate"].clip(0, 1)

    df.loc[df.index[mask], "latency_p99_ms"] *= (1.2 + 0.3 * scenario.severity)
    metrics[target] = df
    return metrics


def _inject_ddos(
    metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """10x request rate spike on api-gateway, cascading pressure."""
    target = scenario.target_service  # api-gateway
    df = metrics[target].copy()
    mask = _time_mask(df.index, scenario.start_time, scenario.duration)
    rng = np.random.RandomState(44)

    multiplier = 5.0 + scenario.severity * 5.0  # 5x - 10x
    noise = rng.randn(int(mask.sum())) * 50.0
    df.loc[df.index[mask], "request_rate"] *= multiplier
    df.loc[df.index[mask], "request_rate"] += noise

    df.loc[df.index[mask], "cpu_percent"] = np.clip(
        df.loc[df.index[mask], "cpu_percent"] + 30 * scenario.severity, 0, 100
    )
    df.loc[df.index[mask], "latency_p50_ms"] *= (2.0 + scenario.severity * 3.0)
    df.loc[df.index[mask], "latency_p99_ms"] *= (3.0 + scenario.severity * 5.0)
    metrics[target] = df

    downstream = get_downstream(topology, target)
    for ds in downstream:
        ds_df = metrics[ds].copy()
        ds_mask = _time_mask(ds_df.index, scenario.start_time + 10, scenario.duration)
        ds_df.loc[ds_df.index[ds_mask], "request_rate"] *= (multiplier * 0.6)
        ds_df.loc[ds_df.index[ds_mask], "latency_p50_ms"] *= (1.5 + scenario.severity)
        metrics[ds] = ds_df

    return metrics


_FAULT_HANDLERS = {
    "memory_leak": _inject_memory_leak,
    "cpu_saturation": _inject_cpu_saturation,
    "brute_force": _inject_brute_force,
    "transaction_stall": _inject_transaction_stall,
    "cascading_failure": _inject_cascading_failure,
    "deployment_regression": _inject_deployment_regression,
    "anomalous_access": _inject_anomalous_access,
    "ddos": _inject_ddos,
}


def inject_fault(
    normal_metrics: dict[str, pd.DataFrame],
    scenario: FaultScenario,
    topology: nx.DiGraph,
) -> dict[str, pd.DataFrame]:
    """Modify metrics according to fault scenario. Returns modified copy."""
    metrics = {svc: df.copy() for svc, df in normal_metrics.items()}
    handler = _FAULT_HANDLERS.get(scenario.fault_type)
    if handler is None:
        raise ValueError(f"Unknown fault type: {scenario.fault_type}")
    return handler(metrics, scenario, topology)
