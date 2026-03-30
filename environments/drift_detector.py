"""Feature drift detector using Population Stability Index (PSI).

Compares the distribution of incoming live feature vectors
against the reference distribution from training. A PSI > 0.1 is a warning;
PSI > 0.2 (industry standard threshold) indicates significant drift and should
trigger re-calibration or retraining.

PSI = sum((live_pct - ref_pct) * ln(live_pct / ref_pct))

Interpretation:
  PSI < 0.1   — no significant change
  0.1–0.2     — moderate drift, monitor closely
  > 0.2       — significant drift, recalibrate or retrain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

PSI_WARNING = 0.1
PSI_CRITICAL = 0.2
N_BINS = 10
# Small epsilon added to bin counts to avoid log(0) and division by zero
EPSILON = 1e-6


@dataclass
class DriftReport:
    """PSI scores for each feature and overall drift assessment."""

    psi_scores: dict[str, float]
    drifted_features: list[str]
    max_psi: float
    mean_psi: float

    @property
    def is_warning(self) -> bool:
        return self.max_psi > PSI_WARNING

    @property
    def is_critical(self) -> bool:
        return self.max_psi > PSI_CRITICAL

    @property
    def severity(self) -> str:
        if self.is_critical:
            return "critical"
        if self.is_warning:
            return "warning"
        return "ok"


class MetricDriftMonitor:
    """Monitors feature distribution drift between training and live data.

    Usage:
        monitor = MetricDriftMonitor(feature_names)
        monitor.fit(training_features)          # establish reference
        report = monitor.score(live_features)   # compare live window
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self._ref_distributions: dict[str, np.ndarray] = {}
        self._bin_edges: dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, baseline_features: np.ndarray) -> None:
        """Establish reference distributions from baseline (training) data."""
        for i, name in enumerate(self.feature_names):
            col = baseline_features[:, i]
            hist, edges = np.histogram(col, bins=N_BINS)
            # Normalize with epsilon to avoid division by zero
            self._ref_distributions[name] = (hist + EPSILON) / (hist.sum() + EPSILON * N_BINS)
            self._bin_edges[name] = edges
        self._fitted = True
        logger.info("drift monitor fitted on %d samples, %d features", len(baseline_features), len(self.feature_names))

    def score(self, live_features: np.ndarray) -> DriftReport:
        """Compute PSI for each feature against the reference distribution."""
        if not self._fitted:
            return DriftReport({}, [], 0.0, 0.0)

        psi_scores: dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            if name not in self._ref_distributions:
                continue
            col = live_features[:, i]
            ref_dist = self._ref_distributions[name]
            edges = self._bin_edges[name]

            # Use the same bin edges as the reference to ensure comparable bins
            live_hist, _ = np.histogram(col, bins=edges)
            live_dist = (live_hist + EPSILON) / (live_hist.sum() + EPSILON * N_BINS)

            psi = float(np.sum((live_dist - ref_dist) * np.log(live_dist / ref_dist)))
            psi_scores[name] = round(max(psi, 0.0), 4)

        drifted = [n for n, p in psi_scores.items() if p > PSI_CRITICAL]
        max_psi = max(psi_scores.values(), default=0.0)
        mean_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0

        report = DriftReport(
            psi_scores=psi_scores,
            drifted_features=drifted,
            max_psi=round(max_psi, 4),
            mean_psi=round(mean_psi, 4),
        )

        if report.is_critical:
            logger.warning(
                "feature drift detected (PSI=%.3f > %.1f): drifted=%s",
                max_psi, PSI_CRITICAL, drifted,
            )
        elif report.is_warning:
            logger.info("moderate drift (PSI=%.3f), monitoring closely", max_psi)

        return report
