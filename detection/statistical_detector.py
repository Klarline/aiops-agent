"""Statistical anomaly detector with z-score and CUSUM drift detection.

Complements the Isolation Forest with interpretable statistical methods
that are particularly effective for gradual drift (memory leaks).
"""

from __future__ import annotations

import numpy as np


class StatisticalDetector:
    """Z-score and CUSUM based anomaly detector."""

    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold
        self.baseline_mean: np.ndarray | None = None
        self.baseline_std: np.ndarray | None = None
        self._fitted = False

    def fit(self, normal_features: np.ndarray) -> None:
        """Compute per-feature mean and std from normal data."""
        self.baseline_mean = np.mean(normal_features, axis=0)
        self.baseline_std = np.std(normal_features, axis=0)
        self.baseline_std = np.where(self.baseline_std < 1e-8, 1e-8, self.baseline_std)
        self._fitted = True

    def detect(self, features: np.ndarray) -> float:
        """Return max absolute z-score across features. Higher = more anomalous."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        sample = features.flatten()
        zscores = np.abs((sample - self.baseline_mean) / self.baseline_std)
        return float(np.max(zscores))

    def detect_drift(self, feature_history: list[np.ndarray], drift_threshold: float = 5.0) -> float:
        """CUSUM for detecting gradual drift (e.g., memory leaks).

        Returns maximum CUSUM statistic across features.
        """
        if not self._fitted or len(feature_history) < 2:
            return 0.0

        history = np.array(feature_history)
        zscores = (history - self.baseline_mean) / self.baseline_std

        cusum_pos = np.zeros(zscores.shape[1])
        max_cusum = np.zeros(zscores.shape[1])

        for t in range(len(zscores)):
            cusum_pos = np.maximum(0, cusum_pos + zscores[t] - 0.5)
            max_cusum = np.maximum(max_cusum, cusum_pos)

        return float(np.max(max_cusum))

    @property
    def n_features(self) -> int:
        return len(self.baseline_mean) if self.baseline_mean is not None else 0
