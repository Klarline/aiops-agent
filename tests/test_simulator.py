"""Tests for simulator foundation: topology, metrics, fault injection, environment."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from simulator.service_topology import (
    get_upstream,
    get_downstream,
    get_blast_radius,
    METRIC_NAMES,
)
from simulator.metrics_generator import generate_metrics
from simulator.fault_injector import inject_fault
from simulator.environment import SimulatedEnvironment


class TestTopology:
    def test_topology_is_valid_dag(self, topology):
        assert nx.is_directed_acyclic_graph(topology)

    def test_has_five_services(self, topology):
        assert len(topology.nodes) == 5

    def test_expected_edges(self, topology):
        assert topology.has_edge("api-gateway", "auth-service")
        assert topology.has_edge("api-gateway", "order-service")
        assert topology.has_edge("auth-service", "user-db")
        assert topology.has_edge("order-service", "order-db")

    def test_upstream_downstream(self, topology):
        assert "api-gateway" in get_upstream(topology, "auth-service")
        assert "user-db" in get_downstream(topology, "auth-service")

    def test_blast_radius(self, topology):
        assert get_blast_radius(topology, "api-gateway") == 4
        assert get_blast_radius(topology, "order-db") == 0

    def test_service_attributes(self, topology):
        for svc in topology.nodes:
            attrs = topology.nodes[svc]
            assert "service_type" in attrs
            assert "baseline" in attrs
            assert len(attrs["baseline"]) == len(METRIC_NAMES)


class TestMetricsGenerator:
    def test_cpu_values_in_range(self, normal_metrics):
        for svc, df in normal_metrics.items():
            assert df["cpu_percent"].min() >= 0, f"{svc} CPU below 0"
            assert df["cpu_percent"].max() <= 100, f"{svc} CPU above 100"

    def test_memory_values_in_range(self, normal_metrics):
        for svc, df in normal_metrics.items():
            assert df["memory_percent"].min() >= 0
            assert df["memory_percent"].max() <= 100

    def test_error_rate_in_range(self, normal_metrics):
        for svc, df in normal_metrics.items():
            assert df["error_rate"].min() >= 0
            assert df["error_rate"].max() <= 1.0

    def test_has_all_metrics(self, normal_metrics):
        for svc, df in normal_metrics.items():
            for metric in METRIC_NAMES:
                assert metric in df.columns, f"{svc} missing {metric}"

    def test_diurnal_pattern_exists(self, topology):
        metrics = generate_metrics(topology, duration_seconds=86400, interval_seconds=60, seed=42)
        for svc, df in metrics.items():
            hour_col = df.index.hour
            noon_mask = (hour_col >= 11) & (hour_col <= 13)
            night_mask = (hour_col >= 2) & (hour_col <= 4)
            noon_mean = df.loc[noon_mask, "request_rate"].mean()
            night_mean = df.loc[night_mask, "request_rate"].mean()
            assert noon_mean > night_mean, f"{svc}: noon ({noon_mean:.1f}) should > night ({night_mean:.1f})"

    def test_reproducibility(self, topology):
        m1 = generate_metrics(topology, 600, 10, seed=42)
        m2 = generate_metrics(topology, 600, 10, seed=42)
        for svc in m1:
            pd.testing.assert_frame_equal(m1[svc], m2[svc])

    def test_different_seeds_differ(self, topology):
        m1 = generate_metrics(topology, 600, 10, seed=42)
        m2 = generate_metrics(topology, 600, 10, seed=99)
        diffs = [not m1[svc].equals(m2[svc]) for svc in m1]
        assert any(diffs)


class TestFaultInjection:
    def test_memory_leak_increases_memory(self, normal_metrics, memory_leak_scenario, topology):
        faulty = inject_fault(normal_metrics, memory_leak_scenario, topology)
        normal_mem = normal_metrics["auth-service"]["memory_percent"]
        faulty_mem = faulty["auth-service"]["memory_percent"]
        assert faulty_mem.mean() > normal_mem.mean()

    def test_cpu_saturation_raises_cpu(self, normal_metrics, cpu_saturation_scenario, topology):
        faulty = inject_fault(normal_metrics, cpu_saturation_scenario, topology)
        fault_start = int(cpu_saturation_scenario.start_time / 10)
        fault_end = int((cpu_saturation_scenario.start_time + cpu_saturation_scenario.duration) / 10)
        during_fault = faulty["order-service"]["cpu_percent"].iloc[fault_start:fault_end]
        assert during_fault.mean() > 80.0

    def test_brute_force_spikes_error_rate(self, normal_metrics, brute_force_scenario, topology):
        faulty = inject_fault(normal_metrics, brute_force_scenario, topology)
        fault_start = int(brute_force_scenario.start_time / 10)
        fault_end = int((brute_force_scenario.start_time + brute_force_scenario.duration) / 10)
        during_fault = faulty["auth-service"]["error_rate"].iloc[fault_start:fault_end]
        assert during_fault.mean() > 0.3

    def test_transaction_stall_infra_stays_healthy(
        self, normal_metrics, transaction_stall_scenario, topology
    ):
        """CRITICAL: TPM drops but CPU/memory/latency stay normal."""
        faulty = inject_fault(normal_metrics, transaction_stall_scenario, topology)
        fault_start = int(transaction_stall_scenario.start_time / 10)
        fault_end = int(
            (transaction_stall_scenario.start_time + transaction_stall_scenario.duration) / 10
        )
        order = faulty["order-service"].iloc[fault_start:fault_end]

        assert order["transactions_per_minute"].mean() < 10.0, "TPM should be near zero"

        normal_order = normal_metrics["order-service"].iloc[fault_start:fault_end]
        cpu_diff = abs(order["cpu_percent"].mean() - normal_order["cpu_percent"].mean())
        mem_diff = abs(order["memory_percent"].mean() - normal_order["memory_percent"].mean())
        lat_diff = abs(order["latency_p50_ms"].mean() - normal_order["latency_p50_ms"].mean())

        assert cpu_diff < 10.0, f"CPU should stay normal (diff={cpu_diff:.1f})"
        assert mem_diff < 10.0, f"Memory should stay normal (diff={mem_diff:.1f})"
        assert lat_diff < 20.0, f"Latency should stay normal (diff={lat_diff:.1f})"

    def test_fault_injection_timing(self, normal_metrics, cpu_saturation_scenario, topology):
        faulty = inject_fault(normal_metrics, cpu_saturation_scenario, topology)
        fault_start_idx = int(cpu_saturation_scenario.start_time / 10)
        before = faulty["order-service"]["cpu_percent"].iloc[: fault_start_idx - 1]
        normal_before = normal_metrics["order-service"]["cpu_percent"].iloc[: fault_start_idx - 1]
        pd.testing.assert_series_equal(before, normal_before)


class TestEnvironment:
    def test_environment_reset(self, transaction_stall_scenario):
        env = SimulatedEnvironment(seed=42)
        env.reset(transaction_stall_scenario)
        assert env.current_step == 0
        assert env.scenario is not None
        assert env.metrics_data is not None

    def test_step_advances(self, transaction_stall_scenario):
        env = SimulatedEnvironment(seed=42)
        env.reset(transaction_stall_scenario)
        obs = env.step()
        assert obs is not None
        assert obs.step_index == 0
        assert env.current_step == 1

    def test_get_current_metrics_returns_all(self, transaction_stall_scenario):
        env = SimulatedEnvironment(seed=42)
        env.reset(transaction_stall_scenario)
        metrics = env.get_current_metrics()
        assert len(metrics) == 5
        for svc, vals in metrics.items():
            assert len(vals) == len(METRIC_NAMES)
