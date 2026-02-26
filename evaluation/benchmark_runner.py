"""Benchmark runner for AIOpsLab-format evaluation.

Runs all fault scenarios with multiple random seeds and scores
agents on Detection, Localization, Diagnosis, and Mitigation.
Supports running multiple agents for leaderboard comparison.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from orchestrator.orchestrator import Orchestrator
from orchestrator.scenario_registry import list_scenarios
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


def _run_agent_scenarios(
    agent_factory: Callable[[], Any],
    ensemble: EnsembleDetector,
    seed: int,
    episodes_per_scenario: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run all scenarios for a single agent type."""
    scenarios = list_scenarios()
    all_results: list[dict] = []
    per_scenario: dict[str, dict] = {}

    for scenario_id in scenarios:
        scenario_results = []
        for ep in range(episodes_per_scenario):
            ep_seed = seed + ep
            orch = Orchestrator(seed=ep_seed)
            orch.init_problem(scenario_id)

            agent = agent_factory()
            agent.set_ensemble(ensemble)

            result = orch.run_episode(agent, max_steps=max_steps)

            action_step = -1
            if orch.history:
                action_step = orch.history[0].get("step", -1)

            record = {
                "scenario": scenario_id,
                "episode": ep,
                "seed": ep_seed,
                "detection": result.detection,
                "localization": result.localization,
                "diagnosis": result.diagnosis,
                "mitigation": result.mitigation,
                "score": result.score,
                "action_step": action_step,
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

    mttr_by_scenario: dict[str, float] = {}
    for scenario_id in scenarios:
        steps = [r["action_step"] for r in all_results
                 if r["scenario"] == scenario_id and r["action_step"] > 0]
        if steps:
            mttr_by_scenario[scenario_id] = float(np.mean(steps)) * 10.0

    return {
        "aggregate": aggregate,
        "per_scenario": per_scenario,
        "raw_results": all_results,
        "mttr_by_scenario": mttr_by_scenario,
    }


def run_benchmark(
    seed: int = 42,
    episodes_per_scenario: int = 13,
    max_steps: int = 200,
) -> dict[str, Any]:
    """Run full benchmark with the ML agent (backward compatible)."""
    ensemble = train_ensemble(seed)
    result = _run_agent_scenarios(
        agent_factory=AIOpsAgent,
        ensemble=ensemble,
        seed=seed,
        episodes_per_scenario=episodes_per_scenario,
        max_steps=max_steps,
    )
    result["seed"] = seed
    result["episodes_per_scenario"] = episodes_per_scenario
    result["total_episodes"] = len(result["raw_results"])
    return result


def run_leaderboard(
    seed: int = 42,
    episodes_per_scenario: int = 13,
    max_steps: int = 200,
) -> dict[str, dict[str, Any]]:
    """Run benchmark for all agents and return leaderboard data.

    Returns dict mapping agent name to its full result set, including
    aggregate scores, per-scenario breakdown, and MTTR.
    """
    from evaluation.baseline_agents import StaticThresholdAgent, RandomAgent

    ensemble = train_ensemble(seed)

    agent_configs: list[tuple[str, Callable]] = [
        ("ML Ensemble Agent", AIOpsAgent),
        ("Static Threshold", StaticThresholdAgent),
        ("Random Agent", lambda: RandomAgent(seed=seed)),
    ]

    leaderboard: dict[str, dict[str, Any]] = {}
    for name, factory in agent_configs:
        print(f"  Benchmarking {name}...")
        result = _run_agent_scenarios(
            agent_factory=factory,
            ensemble=ensemble,
            seed=seed,
            episodes_per_scenario=episodes_per_scenario,
            max_steps=max_steps,
        )
        leaderboard[name] = result

    return leaderboard
