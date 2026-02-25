"""Tests for SHAP explainability."""

from __future__ import annotations

import numpy as np

from features.feature_extractor import extract_features_batch, get_feature_names
from detection.explainer import ShapExplainer
from detection.isolation_forest import IsolationForestDetector


class TestShapExplainer:
    def test_explanation_has_top_features(self, normal_metrics):
        df = normal_metrics["api-gateway"]
        feats = extract_features_batch(df, window_size=6)
        feats = feats[~np.isnan(feats).any(axis=1)]

        detector = IsolationForestDetector()
        detector.fit(feats)

        names = get_feature_names()
        explainer = ShapExplainer(detector.model, names)

        sample = feats[-1]
        explanation = explainer.explain(sample)
        assert len(explanation.top_features) == 3
        for name, val in explanation.top_features:
            assert isinstance(name, str)
            assert isinstance(val, float)

    def test_explanation_names_in_feature_names(self, normal_metrics):
        df = normal_metrics["order-service"]
        feats = extract_features_batch(df, window_size=6)
        feats = feats[~np.isnan(feats).any(axis=1)]

        detector = IsolationForestDetector()
        detector.fit(feats)

        names = get_feature_names()
        explainer = ShapExplainer(detector.model, names)

        explanation = explainer.explain(feats[-1])
        for name, _ in explanation.top_features:
            assert name in names
