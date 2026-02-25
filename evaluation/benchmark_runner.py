"""Benchmark runner for AIOpsLab-format evaluation.

Runs all fault scenarios with multiple random seeds and scores
the agent on Detection, Localization, Diagnosis, and Mitigation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orchestrator.orchestrator import Orchestrator
from orchestrator.scenario_registry import list_scenarios
from orchestrator.problem_context import EvaluationResult
from agent.agent import AIOpsAgent
from features.feature_extractor import extract_features_batch
from detection.ensemble import EnsembleDetector
from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics


def train_ensemble(seed: int = 42) -> EnsembleDetector:
    """Train ensemble detector on normal data."""
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


def run_benchmark(
    seed: int = 42,
    episodes_per_scenario: int = 13,
    max_steps: int = 200,
) -> dict[str, Any]:
    """Run full benchmark across all scenarios.

    Returns results dict with per-scenario and aggregate scores.
    """
    ensemble = train_ensemble(seed)
    scenarios = list_scenarios()

    all_results: list[dict] = []
    per_scenario: dict[str, dict] = {}

    for scenario_id in scenarios:
        scenario_results = []
        for ep in range(episodes_per_scenario):
            ep_seed = seed + ep
            orch = Orchestrator(seed=ep_seed)
            ctx = orch.init_problem(scenario_id)

            agent = AIOpsAgent()
            agent.set_ensemble(ensemble)

            result = orch.run_episode(agent, max_steps=max_steps)
            record = {
                "scenario": scenario_id,
                "episode": ep,
                "seed": ep_seed,
                "detection": result.detection,
                "localization": result.localization,
                "diagnosis": result.diagnosis,
                "mitigation": result.mitigation,
                "score": result.score,
            }
            all_results.append(record)
            scenario_results.append(record)

        det = np.mean([r["detection"] for r in scenario_results])
        loc = np.mean([r["localization"] for r in scenario_results])
        diag = np.mean([r["diagnosis"] for r in scenario_results])
        mit = np.mean([r["mitigation"] for r in scenario_results])

        per_scenario[scenario_id] = {
            "detection": float(det),
            "localization": float(loc),
            "diagnosis": float(diag),
            "mitigation": float(mit),
            "average": float(np.mean([det, loc, diag, mit])),
        }

    aggregate = {
        "detection": float(np.mean([r["detection"] for r in all_results])),
        "localization": float(np.mean([r["localization"] for r in all_results])),
        "diagnosis": float(np.mean([r["diagnosis"] for r in all_results])),
        "mitigation": float(np.mean([r["mitigation"] for r in all_results])),
    }
    aggregate["average"] = float(np.mean(list(aggregate.values())))

    return {
        "seed": seed,
        "episodes_per_scenario": episodes_per_scenario,
        "total_episodes": len(all_results),
        "aggregate": aggregate,
        "per_scenario": per_scenario,
        "raw_results": all_results,
    }
