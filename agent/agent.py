"""AIOps autonomous agent with LLM ReAct reasoning and rule-based fallback.

Architecture:
- LLM agent (when available): ReAct loop calling ML modules as tools
- Rule-based fallback (always available): hardcoded observe → think → act pipeline
- ML detection triggers both paths; the LLM decides what to investigate and when to act
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from simulator.environment import Observation
from simulator.service_topology import get_blast_radius, METRIC_NAMES
from orchestrator.problem_context import AgentAction
from features.feature_extractor import extract_features, get_feature_names
from detection.ensemble import EnsembleDetector, AnomalyResult
from detection.explainer import ShapExplainer
from diagnosis.localizer import ServiceLocalizer
from diagnosis.diagnoser import FaultDiagnoser, Diagnosis
from decision.rule_policy import RuleBasedPolicy
from decision.uncertainty_gate import UncertaintyGate
from explanation.summarizer import generate_summary
from agent.tools import ToolRegistry
from agent.prompts import format_system_prompt, parse_react_response


class AIOpsAgent:
    """Autonomous AIOps agent with LLM ReAct loop and rule-based fallback."""

    MAX_REACT_STEPS = 10

    def __init__(self, llm_client=None):
        self.ensemble: EnsembleDetector | None = None
        self.explainer: ShapExplainer | None = None
        self.localizer = ServiceLocalizer()
        self.diagnoser = FaultDiagnoser()
        self.policy = RuleBasedPolicy()
        self.uncertainty_gate = UncertaintyGate()
        self.rl_policy = None

        self._llm = llm_client
        self._tool_registry: ToolRegistry | None = None
        self._environment = None

        self._metrics_history: dict[str, list[dict]] = {}
        self._reasoning_log: list[dict[str, Any]] = []
        self._reasoning_chain: list[dict[str, Any]] = []
        self._feature_names = get_feature_names()
        self._warmup_steps = 15
        self._consecutive_anomalies: int = 0
        self._required_confirmations: int = 5
        self._followup_confirmations: int = 2
        self._pending_scores: dict[str, float] = {}

        self._action_count: int = 0
        self._max_actions: int = int(os.environ.get("AIOPS_MAX_ACTIONS", "3"))
        budget_action = os.environ.get("AIOPS_BUDGET_EXHAUSTED_ACTION", "alert_human").strip().lower()
        self._budget_exhausted_action: str = (
            budget_action if budget_action in {"continue_monitoring", "alert_human"} else "alert_human"
        )

        self._budget_escalated: bool = False

        self._terminal_attempts_by_type: dict[str, int] = {}
        self._total_terminal_attempts: int = 0
        self._max_terminal_per_type: int = 2
        self._max_terminal_total: int = 3

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_ensemble(self, ensemble: EnsembleDetector) -> None:
        """Inject trained ensemble detector."""
        self.ensemble = ensemble
        self.explainer = ShapExplainer(ensemble.iso_detector.model, self._feature_names)
        self._maybe_build_tools()

    def set_environment(self, env) -> None:
        """Inject environment reference for LLM agent tools."""
        self._environment = env
        self._maybe_build_tools()

    def _maybe_build_tools(self) -> None:
        """Build tool registry when all dependencies are available."""
        if (
            self._environment is not None
            and self.ensemble is not None
            and self._llm is not None
            and self._llm.is_available
        ):
            self._tool_registry = ToolRegistry(
                env=self._environment,
                ensemble=self.ensemble,
                explainer=self.explainer,
                localizer=self.localizer,
                diagnoser=self.diagnoser,
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_action(self, observation: Observation) -> AgentAction:
        """Main agent loop: observe -> think -> act."""
        try:
            if self._tool_registry:
                self._tool_registry.update_metrics_history(observation)
            return self._process_observation(observation)
        except Exception as e:
            self._reasoning_log.append(
                {
                    "step": observation.step_index,
                    "error": str(e),
                    "fallback": "alert_human",
                }
            )
            return AgentAction(
                action="alert_human",
                target="",
                explanation=f"Agent error, escalating: {e}",
            )

    # ------------------------------------------------------------------
    # Observation processing (shared detection, then branch to ReAct or rules)
    # ------------------------------------------------------------------

    def _process_observation(self, obs: Observation) -> AgentAction:
        if obs.step_index < self._warmup_steps:
            return AgentAction(action="continue_monitoring")

        if self.ensemble is None:
            return AgentAction(action="continue_monitoring", explanation="No trained model available")

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
        any_anomalous = any(r.is_anomalous for r in per_service_results.values())

        if not any_anomalous:
            self._consecutive_anomalies = 0
            self._pending_scores = {}
            return AgentAction(action="continue_monitoring")

        self._consecutive_anomalies += 1
        for svc, score in per_service_scores.items():
            self._pending_scores[svc] = self._pending_scores.get(svc, 0) * 0.7 + score * 0.3

        if self._action_count >= self._max_actions:
            return self._handle_budget_exhaustion(obs, max_score)

        required = self._followup_confirmations if self._action_count > 0 else self._required_confirmations
        if self._consecutive_anomalies < required:
            return AgentAction(action="continue_monitoring")

        if max_score < 0.3:
            return AgentAction(action="continue_monitoring")

        # --- Branch: LLM ReAct or rule-based pipeline ---
        if self._should_use_react():
            try:
                return self._react_loop(obs, per_service_scores)
            except Exception as e:
                self._reasoning_chain.append({"fallback": f"LLM error: {e}"})

        return self._rule_based_pipeline(
            obs,
            per_service_scores,
            per_service_results,
            per_service_features,
            max_score,
        )

    def _should_use_react(self) -> bool:
        return (
            self._tool_registry is not None
            and self._llm is not None
            and self._llm.is_available
            and self._action_count < self._max_actions
        )

    # ------------------------------------------------------------------
    # LLM ReAct loop
    # ------------------------------------------------------------------

    def _react_loop(
        self,
        obs: Observation,
        per_service_scores: dict[str, float],
    ) -> AgentAction:
        """LLM-driven investigation: reason → call tool → observe → repeat."""
        self._reasoning_chain = []

        system = format_system_prompt(
            service_list=list(obs.topology.nodes()),
            topology_edges=list(obs.topology.edges()),
            tool_descriptions=self._tool_registry.descriptions_text(),
        )

        top_anomalies = sorted(per_service_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        anomaly_summary = ", ".join(f"{svc} (score={s:.2f})" for svc, s in top_anomalies)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Anomaly detected at step {obs.step_index}. "
                    f"Top anomalous services: {anomaly_summary}. "
                    "Investigate the root cause and take appropriate remediation."
                ),
            }
        ]

        for step in range(self.MAX_REACT_STEPS):
            response = self._llm.generate(system, messages)

            thought, tool_name, tool_args = parse_react_response(response)
            self._reasoning_chain.append(
                {
                    "step": step,
                    "thought": thought,
                    "tool": tool_name,
                    "args": tool_args,
                }
            )

            if tool_name is None:
                break

            result = self._tool_registry.call(tool_name, tool_args)
            self._reasoning_chain.append(
                {
                    "step": step,
                    "observation": result.data,
                }
            )

            if self._tool_registry.is_terminal(tool_name):
                if result.success:
                    self._action_count += 1
                    self._consecutive_anomalies = 0
                    action, target = self._map_terminal_action(tool_name, tool_args)

                    self._log_react_reasoning(obs, thought, action, target)
                    return AgentAction(
                        action=action,
                        target=target,
                        explanation=self._react_summary(),
                        confidence=0.8,
                        details={"reasoning_chain": self._reasoning_chain},
                    )

                self._terminal_attempts_by_type[tool_name] = self._terminal_attempts_by_type.get(tool_name, 0) + 1
                self._total_terminal_attempts += 1

                if self._should_force_escalation(tool_name):
                    self._action_count += 1
                    self._consecutive_anomalies = 0
                    self._log_react_reasoning(
                        obs,
                        f"Forced escalation after terminal failures: {tool_name}",
                        "alert_human",
                        "",
                    )
                    return AgentAction(
                        action="alert_human",
                        target="",
                        explanation=(
                            f"Escalating: {tool_name} failed "
                            f"({self._total_terminal_attempts} terminal attempts). " + self._react_summary()
                        ),
                        confidence=0.5,
                        details={"reasoning_chain": self._reasoning_chain},
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool result: ACTION FAILED — {result.summary()}. "
                            "The action did not succeed. Choose an alternative "
                            "approach or escalate with alert_human."
                        ),
                    }
                )
                continue

            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result: {result.summary()}",
                }
            )

        raise RuntimeError("ReAct loop exhausted without terminal action")

    def _should_force_escalation(self, tool_name: str) -> bool:
        """Check if terminal failure guardrails require forced escalation."""
        if self._total_terminal_attempts >= self._max_terminal_total:
            return True
        if self._terminal_attempts_by_type.get(tool_name, 0) >= self._max_terminal_per_type:
            return True
        return False

    def _handle_budget_exhaustion(self, obs: Observation, max_score: float) -> AgentAction:
        """Handle incidents observed after action budget is exhausted.

        Escalates to alert_human exactly once (safe default), then returns
        continue_monitoring to avoid spamming the escalation channel.
        """
        if self._budget_exhausted_action == "alert_human" and not self._budget_escalated:
            self._budget_escalated = True
            explanation = (
                f"Action budget exhausted ({self._action_count}/{self._max_actions}) "
                "while anomaly persists. Escalating to human."
            )
            self._reasoning_log.append(
                {
                    "step": obs.step_index,
                    "timestamp": str(obs.timestamp),
                    "mode": "budget_guardrail",
                    "action": "alert_human",
                    "summary": explanation,
                }
            )
            return AgentAction(
                action="alert_human",
                target="",
                explanation=explanation,
                confidence=0.5,
                details={
                    "reason": "action_budget_exhausted",
                    "max_actions": self._max_actions,
                    "anomaly_score": max_score,
                },
            )
        return AgentAction(
            action="continue_monitoring",
            explanation=(f"Action budget exhausted ({self._action_count}/{self._max_actions}); continuing to monitor."),
        )

    @staticmethod
    def _map_terminal_action(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        """Map tool name to (action_string, target)."""
        mapping = {
            "restart_service": ("restart_service", args.get("service", "")),
            "scale_service": ("scale_out", args.get("service", "")),
            "rollback_deploy": ("rollback", args.get("service", "")),
            "block_ip": ("block_ip", args.get("ip", "")),
            "set_rate_limit": ("rate_limit", args.get("service", "")),
            "alert_human": ("alert_human", ""),
        }
        return mapping.get(tool_name, ("alert_human", ""))

    def _react_summary(self) -> str:
        """Build a natural-language summary from the reasoning chain."""
        thoughts = [s["thought"] for s in self._reasoning_chain if "thought" in s and s["thought"]]
        if thoughts:
            return " → ".join(thoughts)
        return "LLM agent completed investigation."

    def _log_react_reasoning(self, obs: Observation, thought: str, action: str, target: str) -> None:
        self._reasoning_log.append(
            {
                "step": obs.step_index,
                "timestamp": str(obs.timestamp),
                "mode": "react",
                "thought": thought,
                "action": action,
                "target": target,
                "chain_length": len(self._reasoning_chain),
            }
        )

    # ------------------------------------------------------------------
    # Rule-based fallback pipeline (original logic, always available)
    # ------------------------------------------------------------------

    def _rule_based_pipeline(
        self,
        obs: Observation,
        per_service_scores: dict[str, float],
        per_service_results: dict[str, AnomalyResult],
        per_service_features: dict[str, np.ndarray],
        max_score: float,
    ) -> AgentAction:
        """Hardcoded detect → localize → diagnose → decide pipeline."""

        # --- Localize ---
        scores_for_localization = {
            svc: self._pending_scores.get(svc, per_service_scores.get(svc, 0)) for svc in per_service_scores
        }
        root_service = self.localizer.localize(scores_for_localization, obs.topology)

        # --- Explain ---
        shap_explanation = None
        if root_service in per_service_features and self.explainer:
            shap_explanation = self.explainer.explain(per_service_features[root_service])

        # --- Diagnose ---
        history_df = None
        if root_service in self._metrics_history and len(self._metrics_history[root_service]) >= 6:
            history_df = pd.DataFrame(self._metrics_history[root_service])

        diagnosis = self.diagnoser.diagnose(
            obs.metrics,
            history_df,
            root_service,
            per_service_scores.get(root_service, max_score),
        )

        # --- Uncertainty check ---
        root_result = per_service_results.get(root_service)
        if root_result:
            blast = get_blast_radius(obs.topology, root_service)
            gate = self.uncertainty_gate.check(root_result, diagnosis.severity, blast)
            if gate.should_escalate:
                ctx = self._build_context(obs, root_service, diagnosis)
                ctx["reason"] = gate.reason
                summary = generate_summary(
                    diagnosis.fault_type,
                    root_service,
                    "alert_human",
                    diagnosis.severity,
                    shap_explanation.top_features if shap_explanation else None,
                    ctx,
                )
                self._log_reasoning(obs, diagnosis, "alert_human", summary, gate.reason)
                self._action_count += 1
                self._consecutive_anomalies = 0
                return AgentAction(
                    action="alert_human",
                    target=root_service,
                    explanation=summary,
                    confidence=diagnosis.confidence,
                    details={"diagnosis": diagnosis.fault_type, "gate": gate.reason},
                )

        # --- Decide ---
        if self.rl_policy and hasattr(self.rl_policy, "converged") and self.rl_policy.converged:
            action = self.rl_policy.decide(diagnosis)
        else:
            action = self.policy.decide(diagnosis)

        summary = generate_summary(
            diagnosis.fault_type,
            root_service,
            action,
            diagnosis.severity,
            shap_explanation.top_features if shap_explanation else None,
            self._build_context(obs, root_service, diagnosis),
        )

        self._log_reasoning(obs, diagnosis, action, summary)
        self._action_count += 1
        self._consecutive_anomalies = 0

        return AgentAction(
            action=action,
            target=root_service,
            explanation=summary,
            confidence=diagnosis.confidence,
            details={"diagnosis": diagnosis.fault_type},
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_service_history(self, service: str, current_metrics: dict[str, float]) -> list[dict[str, float]]:
        if service not in self._metrics_history:
            self._metrics_history[service] = []
        self._metrics_history[service].append(current_metrics)
        if len(self._metrics_history[service]) > 60:
            self._metrics_history[service] = self._metrics_history[service][-60:]
        return self._metrics_history[service]

    def _compute_baseline(
        self,
        history: list[dict],
        fallback: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute per-metric baselines from the warmup window of history.

        Uses the first ``_warmup_steps`` observations (expected to be
        pre-fault normal data).  Falls back to topology-defined baselines
        when history is too short.
        """
        if len(history) >= self._warmup_steps:
            baseline_window = history[: self._warmup_steps]
        elif history:
            baseline_window = history
        else:
            return dict(fallback) if fallback else {}

        baseline_df = pd.DataFrame(baseline_window)
        return {col: float(baseline_df[col].mean()) for col in baseline_df.columns}

    def _build_context(self, obs: Observation, service: str, diagnosis: Diagnosis) -> dict[str, Any]:
        metrics = obs.metrics.get(service, {})
        history = self._metrics_history.get(service, [])

        node_data = obs.topology.nodes.get(service, {})
        topo_baseline = node_data.get("baseline", {})
        baseline = self._compute_baseline(history, fallback=topo_baseline)

        # Use metrics first; if missing/zero, use baseline from history, then topology
        cpu_raw = metrics.get("cpu_percent")
        memory_raw = metrics.get("memory_percent")
        cpu = (
            cpu_raw
            if (cpu_raw is not None and isinstance(cpu_raw, (int, float)) and cpu_raw >= 1)
            else baseline.get("cpu_percent") or topo_baseline.get("cpu_percent")
        )
        memory = (
            memory_raw
            if (memory_raw is not None and isinstance(memory_raw, (int, float)) and memory_raw >= 1)
            else baseline.get("memory_percent") or topo_baseline.get("memory_percent")
        )
        tpm = metrics.get("transactions_per_minute", 0)
        error_rate = metrics.get("error_rate", 0)
        request_rate = metrics.get("request_rate", 0)
        latency_p99 = metrics.get("latency_p99_ms", 0)

        tpm_baseline = baseline.get("transactions_per_minute", 850)
        error_baseline = baseline.get("error_rate", 0.01)
        request_baseline = baseline.get("request_rate", 200)
        latency_baseline = baseline.get("latency_p99_ms", 200)

        mem_rate = 0.0
        if len(history) >= 2:
            recent = pd.DataFrame(history[-6:])
            diffs = recent["memory_percent"].diff().dropna()
            if len(diffs) > 0:
                mem_rate = float(diffs.mean()) * 6  # per-step → per-minute

        peak_zscore = 0.0
        if len(history) >= 6:
            recent_df = pd.DataFrame(history[-6:])
            for metric_name in METRIC_NAMES:
                if metric_name not in recent_df.columns:
                    continue
                series = recent_df[metric_name]
                r_mean = series.mean()
                r_std = series.std()
                if r_std > 1e-8:
                    z = abs(float(series.iloc[-1]) - r_mean) / r_std
                    peak_zscore = max(peak_zscore, z)

        latency_mult = latency_p99 / latency_baseline if latency_baseline > 0 else 1.0
        latency_increase = (latency_p99 - latency_baseline) / latency_baseline * 100 if latency_baseline > 0 else 0
        error_increase = error_rate - error_baseline
        request_mult = request_rate / request_baseline if request_baseline > 0 else 1.0
        error_count = int(request_rate * error_rate)
        observation_window = f"{self._required_confirmations * 10} seconds"

        context: dict[str, Any] = {
            "timestamp": str(obs.timestamp),
            "cpu": cpu if cpu is not None else 0,
            "memory": memory if memory is not None else 0,
            "tpm": tpm,
            "tpm_baseline": tpm_baseline,
            "rate": mem_rate,
            "zscore": peak_zscore,
            "error_count": error_count,
            "window": observation_window,
            "affected_count": sum(1 for s, m in obs.metrics.items() if s != service and m.get("error_rate", 0) > 0.05),
            "latency_mult": latency_mult,
            "latency_increase": latency_increase,
            "error_increase": error_increase,
            "request_mult": request_mult,
        }

        source_ip = ""
        if (
            self._environment is not None
            and hasattr(self._environment, "scenario")
            and self._environment.scenario is not None
        ):
            source_ip = self._environment.scenario.metadata.get("source_ip", "")
        if source_ip:
            context["source_ip"] = source_ip

        return context

    def _log_reasoning(
        self,
        obs: Observation,
        diagnosis: Diagnosis,
        action: str,
        summary: str,
        note: str = "",
    ) -> None:
        self._reasoning_log.append(
            {
                "step": obs.step_index,
                "timestamp": str(obs.timestamp),
                "mode": "rule-based",
                "diagnosis": diagnosis.fault_type,
                "service": diagnosis.localized_service,
                "confidence": diagnosis.confidence,
                "action": action,
                "summary": summary,
                "note": note,
            }
        )

    @property
    def reasoning_log(self) -> list[dict[str, Any]]:
        return list(self._reasoning_log)

    @property
    def reasoning_chain(self) -> list[dict[str, Any]]:
        """Full ReAct reasoning chain from the last LLM investigation."""
        return list(self._reasoning_chain)
