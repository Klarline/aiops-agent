"""Shared test fixtures and configuration."""

from __future__ import annotations

import random

import numpy as np
import pytest

from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics
from simulator.fault_injector import FaultScenario


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
    random.seed(42)


@pytest.fixture
def topology():
    return build_topology()


@pytest.fixture
def normal_metrics(topology):
    return generate_metrics(topology, duration_seconds=3600, interval_seconds=10, seed=42)


@pytest.fixture
def memory_leak_scenario():
    return FaultScenario(
        fault_type="memory_leak",
        target_service="auth-service",
        start_time=300.0,
        duration=2400.0,
        severity=0.8,
    )


@pytest.fixture
def cpu_saturation_scenario():
    return FaultScenario(
        fault_type="cpu_saturation",
        target_service="order-service",
        start_time=200.0,
        duration=600.0,
        severity=0.9,
    )


@pytest.fixture
def brute_force_scenario():
    return FaultScenario(
        fault_type="brute_force",
        target_service="auth-service",
        start_time=150.0,
        duration=300.0,
        severity=0.85,
        is_security=True,
        metadata={"source_ip": "10.0.0.99"},
    )


@pytest.fixture
def transaction_stall_scenario():
    return FaultScenario(
        fault_type="transaction_stall",
        target_service="order-service",
        start_time=250.0,
        duration=1200.0,
        severity=1.0,
    )
