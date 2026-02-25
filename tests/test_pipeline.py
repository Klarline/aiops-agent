"""End-to-end pipeline integration tests."""

from __future__ import annotations

import numpy as np

from features.feature_extractor import extract_features_batch
from detection.ensemble import EnsembleDetector
from agent.agent import AIOpsAgent
from orchestrator.orchestrator import Orchestrator


def _train_ensemble(seed: int = 42) -> EnsembleDetector:
    from simulator.service_topology import build_topology
    from simulator.metrics_generator import generate_metrics
    topology = build_topology()
    metrics = generate_metrics(topology, 3600, 10, seed=seed)
    all_feats = []
    for svc, df in metrics.items():
        feats = extract_features_batch(df, window_size=6)
        all_feats.append(feats)
    features = np.vstack(all_feats)
    features = features[~np.isnan(features).any(axis=1)]
    ensemble = EnsembleDetector()
    ensemble.fit(features)
    return ensemble


class TestFullPipeline:
    def test_cpu_saturation_pipeline(self):
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("cpu_saturation_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        result = orch.run_episode(agent, max_steps=200)
        assert result.detection, "Failed to detect CPU saturation"

    def test_brute_force_pipeline(self):
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("brute_force_auth")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        result = orch.run_episode(agent, max_steps=200)
        assert result.detection, "Failed to detect brute force"

    def test_transaction_stall_pipeline(self):
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("transaction_stall_order")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        result = orch.run_episode(agent, max_steps=200)
        assert result.detection, "Failed to detect transaction stall"

    def test_agent_never_crashes(self):
        """Agent should never crash — falls back to alert_human on error."""
        ensemble = _train_ensemble()
        orch = Orchestrator(seed=42)
        orch.init_problem("memory_leak_auth")

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)

        for _ in range(100):
            obs = orch.env.step()
            if obs is None:
                break
            action = agent.get_action(obs)
            assert action is not None
            assert action.action in [
                "continue_monitoring", "restart_service", "scale_out",
                "rollback", "block_ip", "rate_limit", "alert_human",
            ]
