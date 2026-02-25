"""Root cause service localization.

AIOpsLab Localization task: Given per-service anomaly scores and the
service dependency graph, identify which service is the root cause.
"""

from __future__ import annotations

from collections import deque

import networkx as nx


class ServiceLocalizer:
    """Localize the root cause service using anomaly scores and topology."""

    def __init__(self, anomaly_threshold: float = 0.3):
        self.anomaly_threshold = anomaly_threshold

    def localize(
        self,
        per_service_scores: dict[str, float],
        topology: nx.DiGraph,
    ) -> str:
        """Return the root cause service name.

        Strategy:
        1. Find all anomalous services (score > threshold).
        2. Among anomalous services, find the most upstream in the
           dependency graph — the one with fewest predecessors that
           are also anomalous (BFS from source nodes).
        3. Tie-break by highest anomaly score.
        """
        anomalous = {
            svc: score
            for svc, score in per_service_scores.items()
            if score > self.anomaly_threshold
        }

        if not anomalous:
            return max(per_service_scores, key=per_service_scores.get)

        if len(anomalous) == 1:
            return next(iter(anomalous))

        best = self._find_most_upstream(anomalous, topology)
        return best

    def _find_most_upstream(
        self,
        anomalous: dict[str, float],
        topology: nx.DiGraph,
    ) -> str:
        """Find root cause using combined score and topology position.

        Prioritize: highest anomaly score with a bonus for upstream position.
        """
        max_score = max(anomalous.values())
        top_service = max(anomalous, key=anomalous.get)

        close_threshold = max_score * 0.7
        candidates = {s: v for s, v in anomalous.items() if v >= close_threshold}

        if len(candidates) == 1:
            return next(iter(candidates))

        sources = [n for n in topology.nodes if topology.in_degree(n) == 0]
        visited = set()
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in candidates:
                return node
            for successor in topology.successors(node):
                if successor not in visited:
                    queue.append(successor)

        return top_service
