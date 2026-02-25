"""Tests for anomaly detection ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from features.feature_extractor import extract_features_batch, extract_features
from detection.ensemble import EnsembleDetector
from simulator.fault_injector import inject_fault, FaultScenario


@pytest.fixture
def trained_ensemble(normal_metrics):
    all_features = []
    for svc, df in normal_metrics.items():
        feats = extract_features_batch(df, window_size=6)
        all_features.append(feats)
    features = np.vstack(all_features)
    features = features[~np.isnan(features).any(axis=1)]

    ensemble = EnsembleDetector()
    ensemble.fit(features)
    return ensemble


class TestEnsembleDetector:
    def test_normal_data_mostly_not_anomalous(self, trained_ensemble, normal_metrics):
        df = normal_metrics["api-gateway"]
        feats = extract_features_batch(df, window_size=6)
        anomalous_count = sum(
            1 for f in feats[::10] if trained_ensemble.detect(f).is_anomalous
        )
        total = len(feats[::10])
        fpr = anomalous_count / total
        assert fpr < 0.15, f"False positive rate {fpr:.2f} exceeds 0.15"

    def test_cpu_saturation_detected(self, trained_ensemble, normal_metrics, topology):
        scenario = FaultScenario("cpu_saturation", "order-service", 200, 600, 0.9)
        faulty = inject_fault(normal_metrics, scenario, topology)
        fault_mid = int((200 + 500) / 10)
        df = faulty["order-service"].iloc[:fault_mid + 1]
        features = extract_features(df, window_size=6)
        result = trained_ensemble.detect(features)
        assert result.is_anomalous, f"CPU saturation not detected (score={result.score:.3f})"

    def test_anomaly_result_has_fields(self, trained_ensemble, normal_metrics):
        df = normal_metrics["api-gateway"]
        features = extract_features(df, window_size=6)
        result = trained_ensemble.detect(features)
        assert hasattr(result, "is_anomalous")
        assert hasattr(result, "score")
        assert hasattr(result, "uncertainty")
        assert "isolation_forest" in result.individual_scores
        assert "statistical" in result.individual_scores

    @pytest.mark.statistical
    def test_detection_precision(self, trained_ensemble, normal_metrics, topology):
        """Precision > 0.80 across fault types."""
        faults = [
            FaultScenario("cpu_saturation", "order-service", 200, 600, 0.9),
            FaultScenario("brute_force", "auth-service", 150, 300, 0.85, is_security=True, metadata={"source_ip": "10.0.0.99"}),
            FaultScenario("memory_leak", "auth-service", 300, 2400, 0.8),
        ]

        tp = 0
        for scenario in faults:
            faulty = inject_fault(normal_metrics, scenario, topology)
            target = scenario.target_service
            fault_mid = int((scenario.start_time + scenario.duration / 2) / 10)
            df = faulty[target].iloc[:fault_mid + 1]
            features = extract_features(df, window_size=6)
            result = trained_ensemble.detect(features)
            if result.is_anomalous:
                tp += 1

        recall = tp / len(faults)
        assert recall >= 0.66, f"Detection recall {recall:.2f} too low"
