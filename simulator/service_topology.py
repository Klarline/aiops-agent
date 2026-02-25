"""Service topology for simulated microservice environment.

Defines 5 microservices with dependency graph:
    api-gateway -> auth-service -> user-db
    api-gateway -> order-service -> order-db
"""

from __future__ import annotations

import networkx as nx

SERVICE_DEFINITIONS: dict[str, dict] = {
    "api-gateway": {
        "service_type": "web",
        "baseline": {
            "cpu_percent": 35.0,
            "memory_percent": 40.0,
            "latency_p50_ms": 20.0,
            "latency_p99_ms": 80.0,
            "error_rate": 0.01,
            "request_rate": 500.0,
            "disk_io_percent": 10.0,
            "transactions_per_minute": 850.0,
        },
    },
    "auth-service": {
        "service_type": "auth",
        "baseline": {
            "cpu_percent": 25.0,
            "memory_percent": 35.0,
            "latency_p50_ms": 15.0,
            "latency_p99_ms": 60.0,
            "error_rate": 0.02,
            "request_rate": 300.0,
            "disk_io_percent": 5.0,
            "transactions_per_minute": 600.0,
        },
    },
    "order-service": {
        "service_type": "compute",
        "baseline": {
            "cpu_percent": 40.0,
            "memory_percent": 50.0,
            "latency_p50_ms": 50.0,
            "latency_p99_ms": 200.0,
            "error_rate": 0.01,
            "request_rate": 200.0,
            "disk_io_percent": 15.0,
            "transactions_per_minute": 850.0,
        },
    },
    "user-db": {
        "service_type": "database",
        "baseline": {
            "cpu_percent": 30.0,
            "memory_percent": 55.0,
            "latency_p50_ms": 5.0,
            "latency_p99_ms": 25.0,
            "error_rate": 0.005,
            "request_rate": 400.0,
            "disk_io_percent": 40.0,
            "transactions_per_minute": 700.0,
        },
    },
    "order-db": {
        "service_type": "database",
        "baseline": {
            "cpu_percent": 35.0,
            "memory_percent": 60.0,
            "latency_p50_ms": 8.0,
            "latency_p99_ms": 35.0,
            "error_rate": 0.005,
            "request_rate": 250.0,
            "disk_io_percent": 45.0,
            "transactions_per_minute": 850.0,
        },
    },
}

EDGES = [
    ("api-gateway", "auth-service"),
    ("api-gateway", "order-service"),
    ("auth-service", "user-db"),
    ("order-service", "order-db"),
]

METRIC_NAMES = [
    "cpu_percent",
    "memory_percent",
    "latency_p50_ms",
    "latency_p99_ms",
    "error_rate",
    "request_rate",
    "disk_io_percent",
    "transactions_per_minute",
]


def build_topology() -> nx.DiGraph:
    """Build the microservice dependency graph."""
    graph = nx.DiGraph()
    for name, attrs in SERVICE_DEFINITIONS.items():
        graph.add_node(name, **attrs)
    graph.add_edges_from(EDGES)
    return graph


def get_upstream(graph: nx.DiGraph, service: str) -> list[str]:
    """Return services that *depend on* this service (predecessors in the DAG)."""
    return list(graph.predecessors(service))


def get_downstream(graph: nx.DiGraph, service: str) -> list[str]:
    """Return services this service depends on (successors in the DAG)."""
    return list(graph.successors(service))


def get_blast_radius(graph: nx.DiGraph, service: str) -> int:
    """Count of all downstream dependents (transitive)."""
    return len(nx.descendants(graph, service))
