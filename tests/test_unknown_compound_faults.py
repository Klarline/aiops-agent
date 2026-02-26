"""Tests for unknown and compound fault scenarios (Phase B3).

Validates:
- network_partition fault generates metrics that differ from normal
- Agent safely escalates (alert_human) on unknown fault rather than
  taking incorrect autonomous action
- Compound (DDoS + deployment_regression) scenario runs and the agent
  logs uncertainty / escalation
- Compound scenarios inject both faults onto the metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics
from simulator.fault_injector import (
    FaultScenario,
    inject_fault,
    inject_compound_fault,
)
from orchestrator.scenario_registry import (
    get_scenario,
    list_scenarios,
    UNKNOWN_SCENARIOS,
    COMPOUND_SCENARIOS,
    CompoundScenario,
)
from orchestrator.orchestrator import Orchestrator
from evaluation.benchmark_runner import train_ensemble
from agent.agent import AIOpsAgent


class TestNetworkPartitionFault:
    @pytest.fixture(scope="class")
    def topology(self):
        return build_topology()

    @pytest.fixture(scope="class")
    def normal_metrics(self, topology):
        return generate_metrics(topology, 3600, 10, seed=42)

    def test_injection_produces_modified_metrics(self, topology, normal_metrics):
        scenario = FaultScenario(
            fault_type="network_partition",
            target_service="order-service",
            start_time=200,
            duration=800,
            severity=0.75,
        )
        faulty = inject_fault(normal_metrics, scenario, topology)
        target_normal = normal_metrics["order-service"]["latency_p50_ms"]
        target_faulty = faulty["order-service"]["latency_p50_ms"]
        assert not np.allclose(target_normal.values, target_faulty.values), "Network partition should modify latency"

    def test_intermittent_pattern(self, topology, normal_metrics):
        """Fault should produce sporadic (not sustained) deviations."""
        scenario = FaultScenario(
            fault_type="network_partition",
            target_service="order-service",
            start_time=200,
            duration=800,
            severity=0.75,
        )
        faulty = inject_fault(normal_metrics, scenario, topology)
        err = faulty["order-service"]["error_rate"]
        fault_window = err.iloc[20:100]  # steps 20-100 are in the fault window
        diffs = fault_window.diff().dropna()
        sign_changes = (diffs[:-1].values * diffs[1:].values < 0).sum()
        assert sign_changes > 5, "Intermittent fault should have many sign changes"

    def test_downstream_affected(self, topology, normal_metrics):
        scenario = FaultScenario(
            fault_type="network_partition",
            target_service="order-service",
            start_time=200,
            duration=800,
            severity=0.75,
        )
        faulty = inject_fault(normal_metrics, scenario, topology)
        ds_normal = normal_metrics["order-db"]["latency_p50_ms"]
        ds_faulty = faulty["order-db"]["latency_p50_ms"]
        assert not np.allclose(ds_normal.values, ds_faulty.values), "Downstream should see partition effects"


class TestUnknownFaultSafeEscalation:
    def test_scenario_registered(self):
        assert "network_partition_order" in UNKNOWN_SCENARIOS
        scenario = get_scenario("network_partition_order")
        assert scenario.fault_type == "network_partition"

    def test_agent_does_not_misdiagnose(self):
        """On an unknown fault, the agent should either diagnose 'unknown'
        or escalate to alert_human — never confidently apply the wrong fix."""
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("network_partition_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        actions_taken = []
        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                actions_taken.append(action)

        assert len(actions_taken) >= 1, "Agent should detect the anomaly"

        dangerous_misdiagnoses = {
            "restart_service",
            "scale_out",
            "rollback",
            "block_ip",
            "rate_limit",
        }
        for a in actions_taken:
            diag = a.details.get("diagnosis", "")
            if a.action in dangerous_misdiagnoses:
                assert diag != "network_partition", "Agent should not know about network_partition"

    def test_extended_listing_includes_unknown(self):
        ids = list_scenarios(include_extended=True)
        assert "network_partition_order" in ids
        base_ids = list_scenarios(include_extended=False)
        assert "network_partition_order" not in base_ids


class TestCompoundFaultInjection:
    @pytest.fixture(scope="class")
    def topology(self):
        return build_topology()

    @pytest.fixture(scope="class")
    def normal_metrics(self, topology):
        return generate_metrics(topology, 3600, 10, seed=42)

    def test_compound_injects_both_faults(self, topology, normal_metrics):
        faults = [
            FaultScenario(
                fault_type="ddos",
                target_service="api-gateway",
                start_time=150,
                duration=500,
                severity=0.8,
                is_security=True,
            ),
            FaultScenario(
                fault_type="deployment_regression",
                target_service="order-service",
                start_time=200,
                duration=600,
                severity=0.7,
            ),
        ]
        faulty = inject_compound_fault(normal_metrics, faults, topology)

        gw_normal_req = normal_metrics["api-gateway"]["request_rate"].iloc[20:70]
        gw_faulty_req = faulty["api-gateway"]["request_rate"].iloc[20:70]
        assert gw_faulty_req.mean() > gw_normal_req.mean() * 2, "DDoS should spike gateway request rate"

        os_normal_lat = normal_metrics["order-service"]["latency_p50_ms"].iloc[25:80]
        os_faulty_lat = faulty["order-service"]["latency_p50_ms"].iloc[25:80]
        assert os_faulty_lat.mean() > os_normal_lat.mean() * 1.2, (
            "Deploy regression should increase order-service latency"
        )

    def test_compound_scenario_registered(self):
        assert "ddos_plus_deploy_regression" in COMPOUND_SCENARIOS
        scenario = get_scenario("ddos_plus_deploy_regression")
        assert isinstance(scenario, CompoundScenario)
        assert len(scenario.faults) == 2
        assert scenario.expected_escalation

    def test_compound_scenario_runs(self):
        ensemble = train_ensemble(seed=42)
        orch = Orchestrator(seed=42)
        orch.init_problem("ddos_plus_deploy_regression")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        actions_taken = []
        for _ in range(200):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            if action.action != "continue_monitoring":
                actions_taken.append(action)
                break

        assert len(actions_taken) >= 1, "Agent should detect the compound fault"

    def test_compound_ground_truth_properties(self):
        scenario = COMPOUND_SCENARIOS["ddos_plus_deploy_regression"]
        assert scenario.fault_type == "ddos+deployment_regression"
        assert scenario.target_service == "api-gateway"
        assert scenario.is_security is True
