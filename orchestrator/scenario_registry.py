"""Registry of all fault scenarios with ground truth.

Defines 8 fault scenarios across operational, security, and business-logic
categories, each with specific parameters for reproducible evaluation.
"""

from __future__ import annotations

from simulator.fault_injector import FaultScenario

SCENARIO_REGISTRY: dict[str, FaultScenario] = {
    "memory_leak_auth": FaultScenario(
        fault_type="memory_leak",
        target_service="auth-service",
        start_time=300.0,
        duration=2400.0,
        severity=0.8,
        description="Gradual memory leak on auth-service causing eventual OOM risk",
    ),
    "cpu_saturation_order": FaultScenario(
        fault_type="cpu_saturation",
        target_service="order-service",
        start_time=200.0,
        duration=600.0,
        severity=0.9,
        description="Sudden CPU saturation on order-service from runaway query",
    ),
    "brute_force_auth": FaultScenario(
        fault_type="brute_force",
        target_service="auth-service",
        start_time=150.0,
        duration=300.0,
        severity=0.85,
        is_security=True,
        metadata={"source_ip": "10.0.0.99"},
        description="Brute force login attack targeting auth-service",
    ),
    "transaction_stall_order": FaultScenario(
        fault_type="transaction_stall",
        target_service="order-service",
        start_time=250.0,
        duration=1200.0,
        severity=1.0,
        description="Silent transaction processing failure — infra appears healthy but TPM drops to zero",
    ),
    "cascading_failure_gateway": FaultScenario(
        fault_type="cascading_failure",
        target_service="api-gateway",
        start_time=200.0,
        duration=900.0,
        severity=0.8,
        description="Cascading latency and error propagation from api-gateway through all services",
    ),
    "deployment_regression_order": FaultScenario(
        fault_type="deployment_regression",
        target_service="order-service",
        start_time=300.0,
        duration=1800.0,
        severity=0.7,
        metadata={"deploy_version": "v2.3.1"},
        description="Post-deployment latency and error rate regression on order-service",
    ),
    "anomalous_access_userdb": FaultScenario(
        fault_type="anomalous_access",
        target_service="user-db",
        start_time=400.0,
        duration=600.0,
        severity=0.6,
        is_security=True,
        metadata={"pattern": "unusual_query_volume"},
        description="Anomalous data access patterns detected on user-db",
    ),
    "ddos_gateway": FaultScenario(
        fault_type="ddos",
        target_service="api-gateway",
        start_time=100.0,
        duration=500.0,
        severity=0.95,
        is_security=True,
        metadata={"attack_type": "volumetric"},
        description="Distributed denial of service attack flooding api-gateway",
    ),
}


def get_scenario(scenario_id: str) -> FaultScenario:
    """Retrieve a scenario by ID."""
    if scenario_id not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {list(SCENARIO_REGISTRY.keys())}"
        )
    return SCENARIO_REGISTRY[scenario_id]


def list_scenarios() -> list[str]:
    """Return all registered scenario IDs."""
    return list(SCENARIO_REGISTRY.keys())
