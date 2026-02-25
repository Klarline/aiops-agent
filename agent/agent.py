"""AIOps autonomous agent implementing observe -> think -> act loop.

Follows the AIOpsLab iterative agent pattern where each step:
1. OBSERVE: Extract features from current metrics
2. THINK: Detect anomalies, explain, localize, diagnose
3. ACT: Choose and execute remediation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from simulator.environment import Observation
from simulator.service_topology import get_blast_radius
from orchestrator.problem_context import AgentAction
from features.feature_extractor import extract_features, get_feature_names
from detection.ensemble import EnsembleDetector, AnomalyResult
from detection.explainer import ShapExplainer
from diagnosis.localizer import ServiceLocalizer
from diagnosis.diagnoser import FaultDiagnoser, Diagnosis
from decision.rule_policy import RuleBasedPolicy
from decision.uncertainty_gate import UncertaintyGate
from explanation.summarizer import generate_summary


class AIOpsAgent:
    """Autonomous AIOps agent with observe-think-act loop."""

    def __init__(self):
        self.ensemble: EnsembleDetector | None = None
        self.explainer: ShapExplainer | None = None
        self.localizer = ServiceLocalizer()
        self.diagnoser = FaultDiagnoser()
        self.policy = RuleBasedPolicy()
        self.uncertainty_gate = UncertaintyGate()
        self.rl_policy = None  # Set later if RL converges

        self._metrics_history: dict[str, list[dict]] = {}
        self._reasoning_log: list[dict[str, Any]] = []
        self._feature_names = get_feature_names()
        self._warmup_steps = 15
        self._consecutive_anomalies: int = 0
        self._required_confirmations: int = 5
        self._pending_scores: dict[str, float] = {}
        self._acted = False

    def set_ensemble(self, ensemble: EnsembleDetector) -> None:
        """Inject trained ensemble detector."""
        self.ensemble = ensemble
        self.explainer = ShapExplainer(
            ensemble.iso_detector.model, self._feature_names
        )

    def get_action(self, observation: Observation) -> AgentAction:
        """Main agent loop: observe -> think -> act."""
        try:
            return self._process_observation(observation)
        except Exception as e:
            self._reasoning_log.append({
                "step": observation.step_index,
                "error": str(e),
                "fallback": "alert_human",
            })
            return AgentAction(
                action="alert_human",
                target="",
                explanation=f"Agent error, escalating: {e}",
            )

    def _process_observation(self, obs: Observation) -> AgentAction:
        if obs.step_index < self._warmup_steps:
            return AgentAction(action="continue_monitoring")

        if self.ensemble is None:
            return AgentAction(action="continue_monitoring",
                             explanation="No trained model available")

        # --- OBSERVE ---
        per_service_scores: dict[str, float] = {}
        per_service_results: dict[str, AnomalyResult] = {}
        per_service_features: dict[str, np.ndarray] = {}

        for svc, metrics_dict in obs.metrics.items():
            history = self._get_service_history(svc, metrics_dict)
            if len(history) < 6:
                per_service_scores[svc] = 0.0
                continue

            df = pd.DataFrame(history)
            features = extract_features(df, window_size=6)
            per_service_features[svc] = features

            result = self.ensemble.detect(features)
            per_service_scores[svc] = result.score
            per_service_results[svc] = result

        # --- THINK: Detect ---
        max_score = max(per_service_scores.values()) if per_service_scores else 0
        any_anomalous = any(
            r.is_anomalous for r in per_service_results.values()
        )

        if not any_anomalous:
            self._consecutive_anomalies = 0
            self._pending_scores = {}
            return AgentAction(action="continue_monitoring")

        self._consecutive_anomalies += 1
        for svc, score in per_service_scores.items():
            self._pending_scores[svc] = self._pending_scores.get(svc, 0) * 0.7 + score * 0.3

        if self._acted:
            return AgentAction(action="continue_monitoring")

        if self._consecutive_anomalies < self._required_confirmations:
            return AgentAction(action="continue_monitoring")

        if max_score < 0.3:
            return AgentAction(action="continue_monitoring")

        # --- THINK: Localize ---
        scores_for_localization = {
            svc: self._pending_scores.get(svc, per_service_scores.get(svc, 0))
            for svc in per_service_scores
        }
        root_service = self.localizer.localize(scores_for_localization, obs.topology)

        # --- THINK: Explain ---
        shap_explanation = None
        if root_service in per_service_features and self.explainer:
            shap_explanation = self.explainer.explain(
                per_service_features[root_service]
            )

        # --- THINK: Diagnose ---
        history_df = None
        if root_service in self._metrics_history and len(self._metrics_history[root_service]) >= 6:
            history_df = pd.DataFrame(self._metrics_history[root_service])

        diagnosis = self.diagnoser.diagnose(
            obs.metrics, history_df, root_service,
            per_service_scores.get(root_service, max_score),
        )

        # --- THINK: Uncertainty check ---
        root_result = per_service_results.get(root_service)
        if root_result:
            blast = get_blast_radius(obs.topology, root_service)
            gate = self.uncertainty_gate.check(root_result, diagnosis.severity, blast)
            if gate.should_escalate:
                summary = generate_summary(
                    diagnosis.fault_type, root_service, "alert_human",
                    diagnosis.severity,
                    shap_explanation.top_features if shap_explanation else None,
                    {"reason": gate.reason},
                )
                self._log_reasoning(obs, diagnosis, "alert_human", summary, gate.reason)
                return AgentAction(
                    action="alert_human",
                    target=root_service,
                    explanation=summary,
                    confidence=diagnosis.confidence,
                    details={"diagnosis": diagnosis.fault_type, "gate": gate.reason},
                )

        # --- ACT: Choose remediation ---
        if self.rl_policy and hasattr(self.rl_policy, "converged") and self.rl_policy.converged:
            action = self.rl_policy.decide(diagnosis)
        else:
            action = self.policy.decide(diagnosis)

        summary = generate_summary(
            diagnosis.fault_type, root_service, action,
            diagnosis.severity,
            shap_explanation.top_features if shap_explanation else None,
            self._build_context(obs, root_service, diagnosis),
        )

        self._log_reasoning(obs, diagnosis, action, summary)
        self._acted = True

        return AgentAction(
            action=action,
            target=root_service,
            explanation=summary,
            confidence=diagnosis.confidence,
            details={"diagnosis": diagnosis.fault_type},
        )

    def _get_service_history(
        self, service: str, current_metrics: dict[str, float]
    ) -> list[dict[str, float]]:
        """Maintain sliding window of metrics history per service."""
        if service not in self._metrics_history:
            self._metrics_history[service] = []
        self._metrics_history[service].append(current_metrics)
        if len(self._metrics_history[service]) > 60:
            self._metrics_history[service] = self._metrics_history[service][-60:]
        return self._metrics_history[service]

    def _build_context(
        self, obs: Observation, service: str, diagnosis: Diagnosis
    ) -> dict[str, Any]:
        """Build context dict for NL summary template."""
        metrics = obs.metrics.get(service, {})
        return {
            "timestamp": str(obs.timestamp),
            "cpu": metrics.get("cpu_percent", 0),
            "memory": metrics.get("memory_percent", 0),
            "tpm": metrics.get("transactions_per_minute", 0),
            "tpm_baseline": 850,
            "rate": 0.5,
            "zscore": 3.0,
            "error_count": int(metrics.get("request_rate", 0) * metrics.get("error_rate", 0)),
            "source_ip": "10.0.0.99",
            "window": "30 seconds",
            "affected_count": sum(
                1 for s, m in obs.metrics.items()
                if s != service and m.get("error_rate", 0) > 0.05
            ),
            "latency_mult": 2.0,
            "latency_increase": 50,
            "error_increase": 0.1,
            "request_mult": metrics.get("request_rate", 500) / 500.0,
        }

    def _log_reasoning(
        self,
        obs: Observation,
        diagnosis: Diagnosis,
        action: str,
        summary: str,
        note: str = "",
    ) -> None:
        self._reasoning_log.append({
            "step": obs.step_index,
            "timestamp": str(obs.timestamp),
            "diagnosis": diagnosis.fault_type,
            "service": diagnosis.localized_service,
            "confidence": diagnosis.confidence,
            "action": action,
            "summary": summary,
            "note": note,
        })

    @property
    def reasoning_log(self) -> list[dict[str, Any]]:
        return list(self._reasoning_log)
