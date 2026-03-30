"""Live environment that scrapes real metrics from a Prometheus server.

Implements BaseEnvironment so the agent and orchestrator work with it
identically to SimulatedEnvironment. No ground truth, no fault injection —
the agent monitors and responds to whatever is actually happening.

Actions are logged but not executed by default (AIOPS_DRY_RUN=true).
Wire in OnlineCalibrator and RealActionExecutor for calibration and live execution.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd
import httpx

from environments.base import BaseEnvironment
from environments.types import ActionResult, Observation
from environments.metric_map import MetricMap

logger = logging.getLogger(__name__)

# Canonical metric names the feature extractor and diagnoser expect.
# Any metric not present in the map gets a 0.0 fallback.
CANONICAL_METRICS = [
    "cpu_percent",
    "memory_percent",
    "latency_p50_ms",
    "latency_p99_ms",
    "error_rate",
    "request_rate",
    "disk_io_mbps",
    "transactions_per_minute",
]

# How many steps of in-memory history to retain per service (~1 hour at 10s intervals)
MAX_HISTORY = 360


class PrometheusEnvironment(BaseEnvironment):
    """Live environment backed by a Prometheus metrics server.

    On each step() call it fires instant queries for every configured metric
    on every configured service and returns an Observation with the same
    structure as SimulatedEnvironment.

    History is accumulated in-memory so the feature extractor has a lookback
    window. On construction, the last PREFETCH_STEPS of data are fetched from
    Prometheus range API to make the agent operational immediately.
    """

    PREFETCH_STEPS = 30  # steps prefetched at startup (~5 min at 10s intervals)

    def __init__(
        self,
        prometheus_url: str,
        metric_map: MetricMap,
        scrape_interval_seconds: int = 15,
        http_timeout: float = 5.0,
    ) -> None:
        self._prometheus_url = prometheus_url.rstrip("/")
        self._metric_map = metric_map
        self._scrape_interval = scrape_interval_seconds
        self._http_timeout = http_timeout

        self._current_step: int = 0
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._history: dict[str, list[dict[str, float]]] = {
            svc: [] for svc in metric_map.services
        }
        self._graph: nx.DiGraph = self._build_topology()
        self.topology: nx.DiGraph = self._graph  # direct attribute, same as SimulatedEnvironment

        # Public attrs tools.py may access
        self.scenario = None
        self.blocked_ips: set[str] = set()
        self.audit_log: list[dict] = []
        self.actions_taken: list[dict] = []

        # Optional hooks (wired in when calibrator/executor are attached)
        self._calibrator: Any = None
        self._action_executor: Any = None

        self._prefetch_history()

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def n_steps(self) -> float:
        return float("inf")  # runs until stopped externally

    @property
    def metrics_data(self) -> dict:
        # Non-None signals to API routes that the environment is active.
        # Return the history dict (may be empty at startup).
        return self._history

    def step(self) -> Observation:
        metrics = self._scrape_current()
        self._metrics_cache = metrics

        for svc, m in metrics.items():
            buf = self._history.setdefault(svc, [])
            buf.append(m)
            if len(buf) > MAX_HISTORY:
                self._history[svc] = buf[-MAX_HISTORY:]

        obs = Observation(
            timestamp=pd.Timestamp.now(),
            metrics=metrics,
            topology=self._graph,
            step_index=self._current_step,
        )
        self._current_step += 1

        # Feed calibrator if attached
        if self._calibrator is not None:
            self._feed_calibrator(metrics)

        return obs

    def get_current_metrics(self) -> dict[str, dict[str, float]]:
        if not self._metrics_cache:
            self._metrics_cache = self._scrape_current()
        return self._metrics_cache

    def get_metrics_history(self, service: str, lookback_steps: int = 30) -> pd.DataFrame:
        buf = self._history.get(service, [])
        window = buf[-lookback_steps:] if buf else []
        if not window:
            return pd.DataFrame(columns=CANONICAL_METRICS)
        return pd.DataFrame(window)

    def execute_action(self, action: str, target: str, **kwargs: Any) -> ActionResult:
        record = {"step": self._current_step, "action": action, "target": target, "kwargs": kwargs}
        self.actions_taken.append(record)

        # Execute via RealActionExecutor if attached, otherwise dry-run
        if self._action_executor is not None:
            result = self._action_executor.execute(action, target, **kwargs)
        else:
            result = ActionResult(
                True, action, target,
                f"[live/dry-run] {action} on {target} logged — set AIOPS_DRY_RUN=false and attach executor to act",
            )

        if action == "block_ip":
            ip = kwargs.get("ip", "unknown")
            self.blocked_ips.add(ip)
            self.audit_log.append({
                "timestamp": str(pd.Timestamp.now()),
                "event": "ip_blocked",
                "ip": ip,
                "service": target,
                "reason": f"Automated block via live agent",
            })

        logger.info("action=%s target=%s success=%s msg=%s",
                    action, target, result.success, result.message)
        return result

    def get_topology(self) -> nx.DiGraph:
        return self._graph

    def get_ground_truth(self) -> None:
        return None  # no injected faults in live mode

    def is_resolved(self) -> bool:
        return False  # runs indefinitely; caller decides when to stop

    # ------------------------------------------------------------------
    # Calibration and action executor wiring
    # ------------------------------------------------------------------

    def attach_calibrator(self, calibrator: Any) -> None:
        """Attach an OnlineCalibrator to re-anchor score normalization to real baselines."""
        self._calibrator = calibrator

    def attach_action_executor(self, executor: Any) -> None:
        """Attach a RealActionExecutor to dispatch actions to real infrastructure."""
        self._action_executor = executor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_topology(self) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()
        for svc in self._metric_map.services:
            g.add_node(svc)
        for src, dst in self._metric_map.topology_edges:
            g.add_edge(src, dst)
        return g

    def _scrape_current(self) -> dict[str, dict[str, float]]:
        """Fire instant Prometheus queries for all services and metrics."""
        results: dict[str, dict[str, float]] = {}
        for svc in self._metric_map.services:
            results[svc] = self._scrape_service(svc)
        return results

    def _scrape_service(self, service: str) -> dict[str, float]:
        """Scrape all configured metrics for one service."""
        metrics: dict[str, float] = {m: 0.0 for m in CANONICAL_METRICS}
        for canonical_name, cfg in self._metric_map.metrics.items():
            query = cfg.query.replace("{service}", service)
            try:
                value = self._instant_query(query)
                if value is not None:
                    metrics[canonical_name] = value
                elif not cfg.optional:
                    logger.warning("required metric %s returned no data for %s", canonical_name, service)
            except Exception as exc:
                if not cfg.optional:
                    logger.warning("failed to scrape %s for %s: %s", canonical_name, service, exc)
        return metrics

    def _instant_query(self, query: str) -> float | None:
        """Execute a Prometheus instant query and return a scalar value."""
        resp = httpx.get(
            f"{self._prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=self._http_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return None

    def _range_query(self, query: str, duration_seconds: int) -> list[tuple[float, float]]:
        """Execute a Prometheus range query and return (timestamp, value) pairs."""
        import time
        end = time.time()
        start = end - duration_seconds
        resp = httpx.get(
            f"{self._prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": self._scrape_interval,
            },
            timeout=self._http_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "success" and data["data"]["result"]:
            return [(float(ts), float(v)) for ts, v in data["data"]["result"][0]["values"]]
        return []

    def _prefetch_history(self) -> None:
        """Pre-populate in-memory history from Prometheus range API at startup.

        This makes the agent operational immediately rather than waiting for
        PREFETCH_STEPS scrape cycles before the feature extractor has enough data.
        """
        duration = self.PREFETCH_STEPS * self._scrape_interval
        for svc in self._metric_map.services:
            step_snapshots: list[dict[str, float]] = []
            for canonical_name, cfg in self._metric_map.metrics.items():
                query = cfg.query.replace("{service}", svc)
                try:
                    points = self._range_query(query, duration)
                    for i, (_, value) in enumerate(points):
                        if i >= len(step_snapshots):
                            step_snapshots.append({m: 0.0 for m in CANONICAL_METRICS})
                        step_snapshots[i][canonical_name] = value
                except Exception as exc:
                    logger.debug("prefetch skipped for %s/%s: %s", svc, canonical_name, exc)

            if step_snapshots:
                self._history[svc] = step_snapshots[-MAX_HISTORY:]
                logger.info("prefetched %d steps for %s", len(step_snapshots), svc)

    def _feed_calibrator(self, metrics: dict[str, dict[str, float]]) -> None:
        """Feed calibrator with feature vectors extracted from current history."""
        try:
            from features.feature_extractor import extract_features

            for svc, history in self._history.items():
                if len(history) >= 6:
                    df = pd.DataFrame(history[-6:])
                    features = extract_features(df, window_size=6)
                    self._calibrator.ingest(features)
        except Exception as exc:
            logger.debug("calibrator feed error: %s", exc)
