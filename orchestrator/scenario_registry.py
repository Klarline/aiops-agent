"""Registry of all fault scenarios with ground truth.

Defines fault scenarios across operational, security, and business-logic
categories, each with specific parameters for reproducible evaluation.

Includes:
- 8 standard single-fault scenarios
- 1 unknown fault (network_partition) — agent should escalate safely
- 1 compound fault (DDoS + deployment_regression) — agent should detect ambiguity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulator.fault_injector import FaultScenario


@dataclass
class CompoundScenario:
    """Multiple simultaneous faults with compound ground truth."""

    faults: list[FaultScenario]
    description: str = ""
    expected_escalation: bool = False

    @property
    def fault_type(self) -> str:
        return "+".join(f.fault_type for f in self.faults)

    @property
    def target_service(self) -> str:
        return self.faults[0].target_service

    @property
    def is_security(self) -> bool:
        return any(f.is_security for f in self.faults)

    @property
    def metadata(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for f in self.faults:
            merged.update(f.metadata)
        return merged


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

UNKNOWN_SCENARIOS: dict[str, FaultScenario] = {
    "network_partition_order": FaultScenario(
        fault_type="network_partition",
        target_service="order-service",
        start_time=200.0,
        duration=800.0,
        severity=0.75,
        description=(
            "Intermittent network partition affecting order-service — "
            "packet loss, latency jitter, sporadic request drops. "
            "Does not match any standard fault signature."
        ),
    ),
}

COMPOUND_SCENARIOS: dict[str, CompoundScenario] = {
    "ddos_plus_deploy_regression": CompoundScenario(
        faults=[
            FaultScenario(
                fault_type="ddos",
                target_service="api-gateway",
                start_time=150.0,
                duration=500.0,
                severity=0.8,
                is_security=True,
                metadata={"attack_type": "volumetric"},
            ),
            FaultScenario(
                fault_type="deployment_regression",
                target_service="order-service",
                start_time=200.0,
                duration=600.0,
                severity=0.7,
                metadata={"deploy_version": "v2.4.0"},
            ),
        ],
        description=(
            "Simultaneous DDoS on api-gateway and deployment regression "
            "on order-service — ambiguous root cause, high diagnostic difficulty."
        ),
        expected_escalation=True,
    ),
}


def get_scenario(scenario_id: str) -> FaultScenario | CompoundScenario:
    """Retrieve a scenario by ID (standard, unknown, or compound)."""
    if scenario_id in SCENARIO_REGISTRY:
        return SCENARIO_REGISTRY[scenario_id]
    if scenario_id in UNKNOWN_SCENARIOS:
        return UNKNOWN_SCENARIOS[scenario_id]
    if scenario_id in COMPOUND_SCENARIOS:
        return COMPOUND_SCENARIOS[scenario_id]
    all_ids = list(SCENARIO_REGISTRY.keys()) + list(UNKNOWN_SCENARIOS.keys()) + list(COMPOUND_SCENARIOS.keys())
    raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {all_ids}")


def list_scenarios(include_extended: bool = False) -> list[str]:
    """Return registered scenario IDs.

    Args:
        include_extended: If True, include unknown and compound scenarios.
    """
    ids = list(SCENARIO_REGISTRY.keys())
    if include_extended:
        ids += list(UNKNOWN_SCENARIOS.keys())
        ids += list(COMPOUND_SCENARIOS.keys())
    return ids
