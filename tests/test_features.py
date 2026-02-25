"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np

from features.feature_extractor import (
    extract_features,
    extract_features_batch,
    get_feature_names,
    FEATURE_SUFFIXES,
)
from simulator.service_topology import METRIC_NAMES


class TestFeatureExtractor:
    def test_feature_vector_shape(self, normal_metrics):
        df = normal_metrics["api-gateway"]
        features = extract_features(df, window_size=6)
        expected = len(METRIC_NAMES) * len(FEATURE_SUFFIXES)
        assert features.shape == (expected,)

    def test_no_nan_in_features(self, normal_metrics):
        for svc, df in normal_metrics.items():
            features = extract_features(df, window_size=6)
            assert not np.any(np.isnan(features)), f"NaN in {svc} features"

    def test_batch_shape(self, normal_metrics):
        df = normal_metrics["order-service"]
        batch = extract_features_batch(df, window_size=6)
        n_expected_rows = len(df) - 6
        n_expected_cols = len(METRIC_NAMES) * len(FEATURE_SUFFIXES)
        assert batch.shape == (n_expected_rows, n_expected_cols)

    def test_feature_names_count(self):
        names = get_feature_names()
        assert len(names) == len(METRIC_NAMES) * len(FEATURE_SUFFIXES)

    def test_features_are_finite(self, normal_metrics):
        df = normal_metrics["auth-service"]
        features = extract_features(df, window_size=6)
        assert np.all(np.isfinite(features))
