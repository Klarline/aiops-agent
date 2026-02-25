"""SHAP-based explainability for anomaly detections.

Generates feature-attribution explanations for detected anomalies,
identifying which metrics contributed most to the detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import shap
except ImportError:
    shap = None


@dataclass
class ShapExplanation:
    """Feature-attribution explanation for an anomaly detection."""

    top_features: list[tuple[str, float]]
    all_values: dict[str, float] = field(default_factory=dict)


class ShapExplainer:
    """Wraps SHAP TreeExplainer for Isolation Forest model."""

    def __init__(self, model, feature_names: list[str]):
        self.feature_names = feature_names
        self._explainer = None
        if shap is not None:
            try:
                self._explainer = shap.TreeExplainer(model)
            except Exception:
                self._explainer = None

    def explain(self, features: np.ndarray, top_k: int = 3) -> ShapExplanation:
        """Generate SHAP explanation for the given feature vector."""
        sample = features.reshape(1, -1) if features.ndim == 1 else features

        if self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(sample)
                vals = shap_values[0] if shap_values.ndim > 1 else shap_values
                return self._build_explanation(vals, top_k)
            except Exception:
                pass

        return self._fallback_explain(features, top_k)

    def _build_explanation(
        self, shap_values: np.ndarray, top_k: int
    ) -> ShapExplanation:
        abs_vals = np.abs(shap_values)
        top_indices = np.argsort(abs_vals)[-top_k:][::-1]
        top_features = [
            (self.feature_names[i], float(shap_values[i]))
            for i in top_indices
        ]
        all_values = {
            self.feature_names[i]: float(shap_values[i])
            for i in range(len(shap_values))
        }
        return ShapExplanation(top_features=top_features, all_values=all_values)

    def _fallback_explain(
        self, features: np.ndarray, top_k: int
    ) -> ShapExplanation:
        """Z-score based fallback when SHAP is unavailable."""
        abs_features = np.abs(features.flatten())
        top_indices = np.argsort(abs_features)[-top_k:][::-1]
        top_features = [
            (self.feature_names[min(i, len(self.feature_names) - 1)], float(abs_features[i]))
            for i in top_indices
        ]
        return ShapExplanation(top_features=top_features)
