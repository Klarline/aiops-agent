"""Tests for fault type diagnosis."""

from __future__ import annotations

import pandas as pd

from diagnosis.diagnoser import FaultDiagnoser


class TestFaultDiagnoser:
    def test_diagnose_cpu_saturation(self):
        diagnoser = FaultDiagnoser()
        snapshot = {
            "order-service": {
                "cpu_percent": 95.0,
                "memory_percent": 50.0,
                "latency_p50_ms": 60.0,
                "latency_p99_ms": 200.0,
                "error_rate": 0.02,
                "request_rate": 200.0,
                "disk_io_percent": 15.0,
                "transactions_per_minute": 800.0,
            }
        }
        result = diagnoser.diagnose(snapshot, None, "order-service", 0.8)
        assert result.fault_type == "cpu_saturation"

    def test_diagnose_transaction_stall(self):
        diagnoser = FaultDiagnoser()
        snapshot = {
            "order-service": {
                "cpu_percent": 35.0,
                "memory_percent": 50.0,
                "latency_p50_ms": 50.0,
                "latency_p99_ms": 200.0,
                "error_rate": 0.01,
                "request_rate": 200.0,
                "disk_io_percent": 15.0,
                "transactions_per_minute": 2.0,
            }
        }
        result = diagnoser.diagnose(snapshot, None, "order-service", 0.9)
        assert result.fault_type == "transaction_stall"

    def test_diagnose_brute_force(self):
        diagnoser = FaultDiagnoser()
        snapshot = {
            "auth-service": {
                "cpu_percent": 30.0,
                "memory_percent": 40.0,
                "latency_p50_ms": 20.0,
                "latency_p99_ms": 80.0,
                "error_rate": 0.6,
                "request_rate": 500.0,
                "disk_io_percent": 5.0,
                "transactions_per_minute": 600.0,
            }
        }
        result = diagnoser.diagnose(snapshot, None, "auth-service", 0.85)
        assert result.fault_type == "brute_force"
        assert result.is_security

    def test_diagnose_memory_leak(self):
        diagnoser = FaultDiagnoser()
        history_data = {
            "memory_percent": [40, 42, 44, 46, 48, 55, 58, 62, 66, 70, 75, 80],
            "cpu_percent": [30] * 12,
            "latency_p50_ms": [15] * 12,
            "latency_p99_ms": [60] * 12,
            "error_rate": [0.02] * 12,
            "request_rate": [300] * 12,
            "disk_io_percent": [5] * 12,
            "transactions_per_minute": [600] * 12,
        }
        history = pd.DataFrame(history_data)
        snapshot = {
            "auth-service": {
                "cpu_percent": 30.0,
                "memory_percent": 80.0,
                "latency_p50_ms": 15.0,
                "latency_p99_ms": 60.0,
                "error_rate": 0.02,
                "request_rate": 300.0,
                "disk_io_percent": 5.0,
                "transactions_per_minute": 600.0,
            }
        }
        result = diagnoser.diagnose(snapshot, history, "auth-service", 0.7)
        assert result.fault_type == "memory_leak"
