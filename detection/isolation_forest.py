"""Isolation Forest anomaly detector.

Trained on normal operational data, detects out-of-distribution
feature vectors as anomalous.
"""

from __future__ import annotations

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Wrapper around scikit-learn IsolationForest for anomaly scoring."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self._fitted = False

    def fit(self, normal_features: np.ndarray) -> None:
        """Train on normal (non-anomalous) feature vectors."""
        self.model.fit(normal_features)
        self._fitted = True

    def detect(self, features: np.ndarray) -> float:
        """Return anomaly score. Higher = more anomalous (flipped from sklearn)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        sample = features.reshape(1, -1) if features.ndim == 1 else features
        return float(-self.model.score_samples(sample)[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction. Returns array of scores (higher = more anomalous)."""
        return -self.model.score_samples(features)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self._fitted = True
