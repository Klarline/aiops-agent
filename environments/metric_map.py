"""Metric map config: maps Prometheus queries to internal canonical metric names.

Load a YAML config file that tells PrometheusEnvironment:
  - which services to monitor
  - how to query each canonical metric from Prometheus
  - the service dependency graph

Example config: config/metric_map.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class MetricConfig:
    """Prometheus query template for one canonical metric."""

    query: str
    optional: bool = True  # if True, missing values fall back to 0.0 silently


@dataclass
class MetricMap:
    """Full mapping of services, metrics, and topology for one deployment."""

    services: list[str]
    metrics: dict[str, MetricConfig]
    topology_edges: list[tuple[str, str]] = field(default_factory=list)


def load_metric_map(path: str) -> MetricMap:
    """Load a MetricMap from a YAML file.

    The YAML format:
        services:
          - api-gateway
          - auth-service

        metrics:
          cpu_percent:
            query: 'rate(container_cpu_usage_seconds_total{container="{service}"}[1m]) * 100'
            optional: false
          transactions_per_minute:
            query: 'rate(business_transactions_total{service="{service}"}[1m]) * 60'
            optional: true

        topology_edges:
          - [api-gateway, auth-service]
          - [api-gateway, order-service]
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    services: list[str] = data.get("services", [])

    metrics: dict[str, MetricConfig] = {}
    for name, cfg in data.get("metrics", {}).items():
        metrics[name] = MetricConfig(
            query=cfg["query"],
            optional=cfg.get("optional", True),
        )

    topology_edges: list[tuple[str, str]] = [
        (str(edge[0]), str(edge[1])) for edge in data.get("topology_edges", [])
    ]

    return MetricMap(services=services, metrics=metrics, topology_edges=topology_edges)
