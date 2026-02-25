"""Tests for the transaction stall scenario — the 'silent failure' demo highlight.

This tests the critical path: infrastructure appears healthy but
transaction throughput drops to zero.
"""

from __future__ import annotations

import numpy as np

from simulator.environment import SimulatedEnvironment
from simulator.fault_injector import FaultScenario
from features.feature_extractor import extract_features, extract_features_batch
from detection.ensemble import EnsembleDetector
from diagnosis.diagnoser import FaultDiagnoser
from simulator.metrics_generator import generate_metrics
from simulator.service_topology import build_topology


class TestTransactionStall:
    def test_stall_detected_by_ensemble(self, normal_metrics, topology):
        all_feats = []
        for svc, df in normal_metrics.items():
            feats = extract_features_batch(df, window_size=6)
            all_feats.append(feats)
        train_features = np.vstack(all_feats)
        train_features = train_features[~np.isnan(train_features).any(axis=1)]

        ensemble = EnsembleDetector()
        ensemble.fit(train_features)

        env = SimulatedEnvironment(seed=42)
        scenario = FaultScenario(
            fault_type="transaction_stall",
            target_service="order-service",
            start_time=250,
            duration=1200,
            severity=1.0,
        )
        env.reset(scenario)

        detected = False
        for step in range(35, 100):
            obs = env.step()
            if obs is None:
                break
            if step < 30:
                continue

            for svc in obs.metrics:
                history = []
                for s in range(max(0, step - 10), step + 1):
                    snap = env.metrics_data[svc].iloc[s].to_dict()
                    history.append(snap)
                if len(history) < 6:
                    continue
                import pandas as pd
                df = pd.DataFrame(history)
                features = extract_features(df, window_size=6)
                result = ensemble.detect(features)
                if result.is_anomalous and svc == "order-service":
                    detected = True
                    break
            if detected:
                break

        assert detected, "Transaction stall not detected by ensemble"

    def test_stall_diagnosed_correctly(self):
        diagnoser = FaultDiagnoser()
        snapshot = {
            "order-service": {
                "cpu_percent": 38.0,
                "memory_percent": 48.0,
                "latency_p50_ms": 52.0,
                "latency_p99_ms": 195.0,
                "error_rate": 0.01,
                "request_rate": 200.0,
                "disk_io_percent": 14.0,
                "transactions_per_minute": 3.0,
            }
        }
        result = diagnoser.diagnose(snapshot, None, "order-service", 0.9)
        assert result.fault_type == "transaction_stall"

    def test_stall_summary_mentions_healthy_infra(self):
        from explanation.summarizer import generate_summary
        summary = generate_summary(
            "transaction_stall", "order-service", "alert_human",
            severity=1.0,
            context={"cpu": 38, "memory": 48, "tpm": 3, "tpm_baseline": 850},
        )
        assert "healthy" in summary.lower()
        assert "transaction" in summary.lower()
        assert "order-service" in summary
