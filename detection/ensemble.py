"""Ensemble anomaly detector combining Isolation Forest and Statistical methods.

Produces calibrated anomaly scores with uncertainty estimates from
model disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detection.isolation_forest import IsolationForestDetector
from detection.statistical_detector import StatisticalDetector


@dataclass
class AnomalyResult:
    """Result of ensemble anomaly detection."""

    is_anomalous: bool
    score: float
    uncertainty: float
    individual_scores: dict[str, float] = field(default_factory=dict)


class EnsembleDetector:
    """Two-model ensemble: Isolation Forest (60%) + Statistical (40%)."""

    def __init__(
        self,
        iso_weight: float = 0.6,
        stat_weight: float = 0.4,
        threshold: float = 0.5,
    ):
        self.iso_detector = IsolationForestDetector()
        self.stat_detector = StatisticalDetector()
        self.iso_weight = iso_weight
        self.stat_weight = stat_weight
        self.threshold = threshold
        self._iso_median: float = 0.0
        self._iso_max: float = 1.0
        self._stat_median: float = 0.0
        self._stat_max: float = 1.0

    def fit(self, normal_features: np.ndarray) -> None:
        """Fit both detectors on normal data and calibrate score normalization."""
        self.iso_detector.fit(normal_features)
        self.stat_detector.fit(normal_features)

        iso_scores = self.iso_detector.predict(normal_features)
        self._iso_median = float(np.median(iso_scores))
        self._iso_max = max(
            float(np.percentile(iso_scores, 99)) - self._iso_median, 1e-8
        )

        stat_scores = np.array([
            self.stat_detector.detect(f) for f in normal_features
        ])
        self._stat_median = float(np.median(stat_scores))
        self._stat_max = max(
            float(np.percentile(stat_scores, 99)) - self._stat_median, 1e-8
        )

    def detect(self, features: np.ndarray) -> AnomalyResult:
        """Run both detectors, combine scores, compute uncertainty."""
        iso_raw = self.iso_detector.detect(features)
        stat_raw = self.stat_detector.detect(features)

        iso_norm = max((iso_raw - self._iso_median) / self._iso_max, 0.0)
        stat_norm = max((stat_raw - self._stat_median) / self._stat_max, 0.0)

        combined = self.iso_weight * iso_norm + self.stat_weight * stat_norm

        # Cap scores before computing uncertainty so two detectors that both
        # say "very anomalous" (e.g. 3.0 vs 60.0) don't produce fake
        # disagreement.  Uncertainty reflects directional disagreement
        # (anomalous vs normal), not magnitude differences.  Scores above
        # 1.0 mean "above p99 of training data" — i.e. definitely anomalous.
        iso_capped = min(iso_norm, 1.0)
        stat_capped = min(stat_norm, 1.0)
        uncertainty = abs(iso_capped - stat_capped)

        return AnomalyResult(
            is_anomalous=combined > self.threshold,
            score=combined,
            uncertainty=uncertainty,
            individual_scores={
                "isolation_forest": iso_norm,
                "statistical": stat_norm,
            },
        )
