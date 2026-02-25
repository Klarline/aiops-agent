"""Train detection models on normal simulated data.

Generates 24h of normal metrics, extracts features, fits ensemble,
saves model artifacts, and runs quick evaluation.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics
from simulator.fault_injector import inject_fault, FaultScenario
from features.feature_extractor import extract_features_batch
from detection.ensemble import EnsembleDetector


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "detection", "models")


def train() -> EnsembleDetector:
    """Train the ensemble detector on normal data."""
    print("Generating 24h of normal metrics (seed=42)...")
    topology = build_topology()
    metrics = generate_metrics(topology, duration_seconds=86400, interval_seconds=10, seed=42)

    print("Extracting features...")
    all_features = []
    for svc, df in metrics.items():
        feats = extract_features_batch(df, window_size=6)
        all_features.append(feats)
    normal_features = np.vstack(all_features)
    print(f"  Feature matrix: {normal_features.shape}")

    nan_mask = np.isnan(normal_features).any(axis=1)
    normal_features = normal_features[~nan_mask]
    print(f"  After NaN removal: {normal_features.shape}")

    print("Fitting ensemble detector...")
    ensemble = EnsembleDetector()
    ensemble.fit(normal_features)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    ensemble.iso_detector.save(model_path)
    print(f"  Saved IsolationForest to {model_path}")

    print("\nQuick evaluation on fault scenarios...")
    _evaluate(ensemble, topology, normal_features)

    return ensemble


def _evaluate(ensemble: EnsembleDetector, topology, normal_features: np.ndarray) -> None:
    """Quick eval: check detection on known faults."""
    test_faults = [
        FaultScenario("memory_leak", "auth-service", 300, 2400, 0.8),
        FaultScenario("cpu_saturation", "order-service", 200, 600, 0.9),
        FaultScenario("brute_force", "auth-service", 150, 300, 0.85, is_security=True, metadata={"source_ip": "10.0.0.99"}),
        FaultScenario("transaction_stall", "order-service", 250, 1200, 1.0),
    ]

    normal_metrics = generate_metrics(topology, 3600, 10, seed=42)

    tp, fp, fn = 0, 0, 0

    for scenario in test_faults:
        faulty_metrics = inject_fault(normal_metrics, scenario, topology)
        fault_start = int(scenario.start_time / 10)
        fault_end = int((scenario.start_time + scenario.duration) / 10)

        detected = False
        for svc, df in faulty_metrics.items():
            if fault_start + 6 >= len(df):
                continue
            mid = min(fault_start + (fault_end - fault_start) // 2, len(df) - 1)
            window = df.iloc[:mid + 1]
            feats = extract_features_batch(window, window_size=6)
            if len(feats) == 0:
                continue
            result = ensemble.detect(feats[-1])
            if result.is_anomalous:
                detected = True
                break

        status = "DETECTED" if detected else "MISSED"
        print(f"  {scenario.fault_type:25s} → {status}")
        if detected:
            tp += 1
        else:
            fn += 1

    n_normal_samples = min(100, len(normal_features))
    indices = np.random.choice(len(normal_features), n_normal_samples, replace=False)
    for idx in indices:
        result = ensemble.detect(normal_features[idx])
        if result.is_anomalous:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1:        {f1:.2f}")
    print(f"  FPR:       {fp}/{n_normal_samples} = {fp/n_normal_samples:.3f}")


if __name__ == "__main__":
    train()
