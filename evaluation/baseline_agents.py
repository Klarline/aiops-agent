"""Baseline and ablation agents for benchmark comparison.

Original baselines:
  StaticThresholdAgent: Simple threshold rules — CPU > 80% → restart, error > 0.1 → alert.
  RandomAgent: Picks random actions to establish a lower bound.

Ablation agents (Phase B2):
  RuleOnlyAgent:    Threshold detection + pattern diagnosis + rule policy (no ML).
  MLRuleAgent:      ML ensemble detection + pattern diagnosis + rule policy (no LLM).
  LLMToolsAgent:    ML detection + LLM ReAct loop (requires LLM, falls back to MLRule).
  FallbackOnlyAgent: Always alert_human after any detection (safe-only baseline).
"""

from __future__ import annotations

import random

import pandas as pd

from simulator.environment import Observation
from orchestrator.problem_context import AgentAction
from diagnosis.localizer import ServiceLocalizer
from diagnosis.diagnoser import FaultDiagnoser
from decision.rule_policy import RuleBasedPolicy


class StaticThresholdAgent:
    """Naive baseline: fixed per-metric thresholds, always restarts."""

    def __init__(self):
        self._acted = False
        self._warmup = 5

    def set_ensemble(self, ensemble):
        pass

    def get_action(self, observation: Observation) -> AgentAction:
        if observation.step_index < self._warmup:
            return AgentAction(action="continue_monitoring")

        if self._acted:
            return AgentAction(action="continue_monitoring")

        for svc, metrics in observation.metrics.items():
            cpu = metrics.get("cpu_percent", 0)
            mem = metrics.get("memory_percent", 0)
            err = metrics.get("error_rate", 0)
            lat = metrics.get("latency_p50_ms", 0)

            if cpu > 80:
                self._acted = True
                return AgentAction(
                    action="restart_service",
                    target=svc,
                    details={"diagnosis": "cpu_saturation"},
                )
            if mem > 75:
                self._acted = True
                return AgentAction(
                    action="restart_service",
                    target=svc,
                    details={"diagnosis": "memory_leak"},
                )
            if err > 0.1:
                self._acted = True
                return AgentAction(
                    action="alert_human",
                    target=svc,
                    details={"diagnosis": "unknown"},
                )
            if lat > 100:
                self._acted = True
                return AgentAction(
                    action="restart_service",
                    target=svc,
                    details={"diagnosis": "deployment_regression"},
                )

        return AgentAction(action="continue_monitoring")


class RandomAgent:
    """Lower bound: takes random actions after detecting any threshold breach."""

    ACTIONS = ["restart_service", "scale_out", "rollback", "block_ip", "rate_limit", "alert_human"]
    FAULT_TYPES = ["cpu_saturation", "memory_leak", "brute_force", "deployment_regression", "ddos", "transaction_stall"]

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._acted = False

    def set_ensemble(self, ensemble):
        pass

    def get_action(self, observation: Observation) -> AgentAction:
        if observation.step_index < 10:
            return AgentAction(action="continue_monitoring")
        if self._acted:
            return AgentAction(action="continue_monitoring")

        for svc, metrics in observation.metrics.items():
            if metrics.get("cpu_percent", 0) > 60 or metrics.get("error_rate", 0) > 0.05:
                self._acted = True
                services = list(observation.metrics.keys())
                return AgentAction(
                    action=self._rng.choice(self.ACTIONS),
                    target=self._rng.choice(services),
                    details={"diagnosis": self._rng.choice(self.FAULT_TYPES)},
                )

        return AgentAction(action="continue_monitoring")


# ---------------------------------------------------------------------------
# Ablation agents (Phase B2)
# ---------------------------------------------------------------------------


class RuleOnlyAgent:
    """Threshold detection + pattern diagnosis + rule policy.

    No ML model — detects anomalies using fixed metric thresholds,
    then runs the full diagnosis/decision pipeline. Isolates the
    contribution of the ML ensemble.
    """

    THRESHOLDS = {
        "cpu_percent": 70,
        "memory_percent": 65,
        "error_rate": 0.06,
        "latency_p50_ms": 80,
        "request_rate_low": 50,
        "transactions_per_minute_low": 100,
    }

    def __init__(self):
        self._localizer = ServiceLocalizer()
        self._diagnoser = FaultDiagnoser()
        self._policy = RuleBasedPolicy()
        self._metrics_history: dict[str, list[dict]] = {}
        self._acted = False
        self._warmup = 15
        self._consecutive = 0
        self._confirm_steps = 5

    def set_ensemble(self, ensemble):
        pass

    def get_action(self, observation: Observation) -> AgentAction:
        if observation.step_index < self._warmup:
            self._update_history(observation)
            return AgentAction(action="continue_monitoring")

        if self._acted:
            return AgentAction(action="continue_monitoring")

        self._update_history(observation)

        anomaly_scores = self._threshold_detect(observation)
        if not any(s > 0 for s in anomaly_scores.values()):
            self._consecutive = 0
            return AgentAction(action="continue_monitoring")

        self._consecutive += 1
        if self._consecutive < self._confirm_steps:
            return AgentAction(action="continue_monitoring")

        root = self._localizer.localize(anomaly_scores, observation.topology)
        history_df = self._get_history_df(root)
        max_score = max(anomaly_scores.values())

        diagnosis = self._diagnoser.diagnose(
            observation.metrics,
            history_df,
            root,
            max_score,
        )
        action = self._policy.decide(diagnosis)

        self._acted = True
        return AgentAction(
            action=action,
            target=root,
            confidence=diagnosis.confidence,
            details={"diagnosis": diagnosis.fault_type},
        )

    def _threshold_detect(self, obs: Observation) -> dict[str, float]:
        scores: dict[str, float] = {}
        for svc, m in obs.metrics.items():
            score = 0.0
            if m.get("cpu_percent", 0) > self.THRESHOLDS["cpu_percent"]:
                score = max(score, 0.7)
            if m.get("memory_percent", 0) > self.THRESHOLDS["memory_percent"]:
                score = max(score, 0.6)
            if m.get("error_rate", 0) > self.THRESHOLDS["error_rate"]:
                score = max(score, 0.7)
            if m.get("latency_p50_ms", 0) > self.THRESHOLDS["latency_p50_ms"]:
                score = max(score, 0.5)
            if m.get("request_rate", 999) < self.THRESHOLDS["request_rate_low"]:
                score = max(score, 0.4)
            tpm = m.get("transactions_per_minute", 999)
            if tpm < self.THRESHOLDS["transactions_per_minute_low"]:
                score = max(score, 0.8)
            scores[svc] = score
        return scores

    def _update_history(self, obs: Observation) -> None:
        for svc, m in obs.metrics.items():
            if svc not in self._metrics_history:
                self._metrics_history[svc] = []
            self._metrics_history[svc].append(m)
            if len(self._metrics_history[svc]) > 60:
                self._metrics_history[svc] = self._metrics_history[svc][-60:]

    def _get_history_df(self, svc: str) -> pd.DataFrame | None:
        h = self._metrics_history.get(svc, [])
        return pd.DataFrame(h) if len(h) >= 6 else None


class MLRuleAgent:
    """ML ensemble detection + rule-based diagnosis/policy.

    Identical to the main AIOpsAgent with LLM disabled. This is
    the default benchmark path, exposed explicitly for ablation.
    """

    def __init__(self):
        from agent.agent import AIOpsAgent

        self._inner = AIOpsAgent(llm_client=None)

    def set_ensemble(self, ensemble):
        self._inner.set_ensemble(ensemble)

    def get_action(self, observation: Observation) -> AgentAction:
        return self._inner.get_action(observation)


class LLMToolsAgent:
    """ML detection + LLM ReAct investigation.

    Requires an LLM client. When unavailable, transparently falls
    back to MLRuleAgent and marks results accordingly.
    """

    def __init__(self, llm_client=None):
        from agent.agent import AIOpsAgent

        self._inner = AIOpsAgent(llm_client=llm_client)
        self.llm_available = llm_client is not None and getattr(llm_client, "is_available", False)

    def set_ensemble(self, ensemble):
        self._inner.set_ensemble(ensemble)

    def get_action(self, observation: Observation) -> AgentAction:
        return self._inner.get_action(observation)


class FallbackOnlyAgent:
    """Always escalates to alert_human on any detected anomaly.

    Represents the safest possible policy — never takes autonomous
    remediation, always defers to a human. Useful as an upper
    bound on safety and lower bound on autonomous value.
    """

    def __init__(self):
        self._acted = False
        self._warmup = 15
        self._consecutive = 0
        self._confirm_steps = 5

    def set_ensemble(self, ensemble):
        pass

    def get_action(self, observation: Observation) -> AgentAction:
        if observation.step_index < self._warmup:
            return AgentAction(action="continue_monitoring")

        if self._acted:
            return AgentAction(action="continue_monitoring")

        any_hot = False
        for svc, m in observation.metrics.items():
            if (
                m.get("cpu_percent", 0) > 70
                or m.get("error_rate", 0) > 0.05
                or m.get("memory_percent", 0) > 65
                or m.get("transactions_per_minute", 999) < 100
            ):
                any_hot = True
                break

        if not any_hot:
            self._consecutive = 0
            return AgentAction(action="continue_monitoring")

        self._consecutive += 1
        if self._consecutive < self._confirm_steps:
            return AgentAction(action="continue_monitoring")

        self._acted = True
        return AgentAction(
            action="alert_human",
            target="",
            details={"diagnosis": "unknown"},
        )
