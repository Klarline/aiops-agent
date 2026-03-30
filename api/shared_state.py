"""Shared state between API routes.

Single source of truth for the orchestrator, agent, and ensemble
so that metrics, agent, and evaluation routes all reference the same simulation.

Phase C: Sessionized run state with per-run isolation and incident IDs.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from orchestrator.orchestrator import Orchestrator
from agent.agent import AIOpsAgent
from features.feature_extractor import extract_features_batch, extract_features, get_feature_names
from explanation.shap_labels import humanize_feature_name
from detection.ensemble import EnsembleDetector
from detection.explainer import ShapExplainer
from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics

DEFAULT_RUN_ID = "default"


class RunSession:
    """Per-run simulation state with run_id and incident_id for traceability."""

    def __init__(self, run_id: str, incident_id: str | None = None):
        self.run_id = run_id
        self.incident_id = incident_id or run_id
        self.orchestrator: Orchestrator | None = None
        self.agent: AIOpsAgent | None = None
        self.ensemble: EnsembleDetector | None = None
        self.explainer: ShapExplainer | None = None
        self.running: bool = False
        self.metrics_history: list[dict[str, dict[str, float]]] = []
        self.anomaly_timeline: list[dict[str, Any]] = []
        self.detection_step: int = -1
        self.feature_names: list[str] = get_feature_names()
        self._started_at: float = 0.0
        self._last_decision_latency_ms: float | None = None

    def ensure_ensemble(self) -> EnsembleDetector:
        if self.ensemble is None:
            topology = build_topology()
            metrics = generate_metrics(topology, 3600, 10, seed=42)
            all_feats = []
            for svc, df in metrics.items():
                feats = extract_features_batch(df, window_size=6)
                all_feats.append(feats)
            features = np.vstack(all_feats)
            features = features[~np.isnan(features).any(axis=1)]
            self.ensemble = EnsembleDetector()
            self.ensemble.fit(features)
            self.explainer = ShapExplainer(
                self.ensemble.iso_detector.model, self.feature_names,
            )
        return self.ensemble

    def reset(self, scenario_id: str, seed: int = 42, incident_id: str | None = None):
        self.incident_id = incident_id or str(uuid.uuid4())
        ensemble = self.ensure_ensemble()

        aiops_mode = os.getenv("AIOPS_MODE", "simulated").lower()

        if aiops_mode == "live":
            ctx = self._reset_live(ensemble)
        else:
            ctx = self._reset_simulated(scenario_id, seed, ensemble)

        self.running = True
        self.metrics_history = []
        self.anomaly_timeline = []
        self.detection_step = -1
        self._started_at = time.monotonic()
        return ctx

    def _reset_simulated(self, scenario_id: str, seed: int, ensemble: EnsembleDetector):
        """Initialize a simulated scenario run."""
        orch = Orchestrator(seed=seed)
        ctx = orch.init_problem(scenario_id)
        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)
        self.orchestrator = orch
        self.agent = agent
        return ctx

    def _reset_live(self, ensemble: EnsembleDetector):
        """Initialize a live Prometheus monitoring run."""
        from environments.prometheus_env import PrometheusEnvironment
        from environments.metric_map import load_metric_map

        prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        map_path = os.getenv("PROMETHEUS_METRIC_MAP", "config/metric_map.yaml")
        scrape_interval = int(os.getenv("PROMETHEUS_SCRAPE_INTERVAL", "15"))

        metric_map = load_metric_map(map_path)
        env = PrometheusEnvironment(prometheus_url, metric_map, scrape_interval)

        # Attach calibrator and action executor
        warmup_steps = int(os.getenv("AIOPS_WARMUP_STEPS", "180"))
        from environments.calibration import OnlineCalibrator
        calibrator = OnlineCalibrator(ensemble, warmup_steps=warmup_steps)
        env.attach_calibrator(calibrator)

        from environments.action_executor_real import RealActionExecutor
        executor = RealActionExecutor.from_env()
        env.attach_action_executor(executor)

        orch = Orchestrator(env=env)
        ctx = orch.init_live()

        agent = AIOpsAgent()
        agent.set_ensemble(ensemble)
        self.orchestrator = orch
        self.agent = agent
        return ctx

    def step_once(self) -> dict[str, Any] | None:
        """Advance one step. Returns step data or None if scenario ended."""
        orch = self.orchestrator
        agent = self.agent
        if orch is None or agent is None:
            return None

        obs = orch.env.step()
        if obs is None:
            self.running = False
            return None

        t0 = time.perf_counter()
        action = agent.get_action(obs)
        self._last_decision_latency_ms = (time.perf_counter() - t0) * 1000

        snapshot = obs.metrics
        self.metrics_history.append(snapshot)

        per_svc_scores = {}
        per_svc_features = {}
        if agent.ensemble is not None and hasattr(agent, '_metrics_history'):
            for svc in snapshot:
                hist = agent._metrics_history.get(svc, [])
                if len(hist) >= 6:
                    df = pd.DataFrame(hist)
                    features = extract_features(df, window_size=6)
                    result = agent.ensemble.detect(features)
                    per_svc_scores[svc] = result.score
                    per_svc_features[svc] = features

        self.anomaly_timeline.append({
            "step": obs.step_index,
            "scores": {s: round(v, 4) for s, v in per_svc_scores.items()},
        })

        resolved = False
        if action.action != "continue_monitoring":
            if self.detection_step < 0:
                self.detection_step = obs.step_index

            orch._agent_detected = True
            orch._agent_localized = action.target
            orch._agent_diagnosed = action.details.get("diagnosis", "")
            orch.env.execute_action(action.action, action.target, **action.details)
            orch._agent_mitigated = orch.env.is_resolved()
            resolved = orch.env.is_resolved()
            if resolved:
                self.running = False

            orch.history.append({
                "action": action,
                "step": orch.env.current_step,
                "incident_id": self.incident_id,
                "run_id": self.run_id,
            })

        return {
            "step": obs.step_index,
            "action": action.action,
            "target": action.target,
            "explanation": action.explanation,
            "confidence": action.confidence,
            "diagnosis": action.details.get("diagnosis", ""),
            "metrics": snapshot,
            "anomaly_scores": per_svc_scores,
            "resolved": resolved,
            "incident_id": self.incident_id,
            "run_id": self.run_id,
        }

    def get_shap_values(self, service: str | None = None) -> dict[str, Any]:
        """Get real SHAP explanations for the most anomalous service."""
        agent = self.agent
        if agent is None or agent.ensemble is None or self.explainer is None:
            return {"features": [], "service": ""}

        target_svc = service
        if target_svc is None and agent._metrics_history:
            best_score = -1.0
            for svc, hist in agent._metrics_history.items():
                if len(hist) >= 6:
                    df = pd.DataFrame(hist)
                    features = extract_features(df, window_size=6)
                    result = agent.ensemble.detect(features)
                    if result.score > best_score:
                        best_score = result.score
                        target_svc = svc

        if target_svc is None or target_svc not in agent._metrics_history:
            return {"features": [], "service": ""}

        hist = agent._metrics_history[target_svc]
        if len(hist) < 6:
            return {"features": [], "service": target_svc}

        df = pd.DataFrame(hist)
        features = extract_features(df, window_size=6)
        explanation = self.explainer.explain(features, top_k=10)

        fnames = get_feature_names()
        raw_values = {fnames[i]: float(features[i]) for i in range(min(len(fnames), len(features)))}

        shap_features = [
            {
                "name": name,
                "value": round(val, 4),
                "human_label": humanize_feature_name(name, raw_values.get(name)),
            }
            for name, val in explanation.top_features
        ]
        all_values = {k: round(v, 4) for k, v in explanation.all_values.items()}

        return {
            "service": target_svc,
            "features": shap_features,
            "all_values": all_values,
        }

    def get_full_history(self) -> list[dict]:
        """Return full metrics history for time-series display."""
        result = []
        for i, snapshot in enumerate(self.metrics_history):
            entry = {"step": i}
            for svc, metrics in snapshot.items():
                for metric, val in metrics.items():
                    entry[f"{svc}__{metric}"] = round(val, 2)
            result.append(entry)
        return result


class RunRegistry:
    """In-memory registry of run sessions. run_id -> RunSession."""

    def __init__(self):
        self._sessions: dict[str, RunSession] = {}

    def get_default(self) -> RunSession:
        """Return the default/legacy session, creating it if needed."""
        if DEFAULT_RUN_ID not in self._sessions:
            self._sessions[DEFAULT_RUN_ID] = RunSession(
                run_id=DEFAULT_RUN_ID,
                incident_id=str(uuid.uuid4()),
            )
        return self._sessions[DEFAULT_RUN_ID]

    def create_run(self) -> tuple[str, str, RunSession]:
        """Create a new run session. Returns (run_id, incident_id, session)."""
        run_id = str(uuid.uuid4())
        incident_id = str(uuid.uuid4())
        session = RunSession(run_id=run_id, incident_id=incident_id)
        self._sessions[run_id] = session
        return run_id, incident_id, session

    def get_session(self, run_id: str) -> RunSession | None:
        """Return session by run_id, or None if not found."""
        return self._sessions.get(run_id)


run_registry = RunRegistry()
state = run_registry.get_default()
