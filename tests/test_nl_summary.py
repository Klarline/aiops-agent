"""Tests for natural language alert summaries."""

from __future__ import annotations

from explanation.summarizer import generate_summary


class TestNLSummary:
    def test_memory_leak_summary_has_service(self):
        summary = generate_summary(
            "memory_leak", "auth-service", "restart_service",
            severity=0.8,
            context={"memory": 85, "rate": 0.5},
        )
        assert "auth-service" in summary
        assert "memory" in summary.lower()

    def test_transaction_stall_summary(self):
        summary = generate_summary(
            "transaction_stall", "order-service", "alert_human",
            severity=1.0,
            context={"cpu": 32, "memory": 45, "tpm": 0, "tpm_baseline": 850},
        )
        assert "order-service" in summary
        assert "transaction" in summary.lower()
        assert "healthy" in summary.lower()

    def test_brute_force_summary_has_action(self):
        summary = generate_summary(
            "brute_force", "auth-service", "block_ip",
            severity=0.85,
            context={"error_count": 247, "source_ip": "10.0.0.99", "window": "30 seconds"},
        )
        assert "blocked" in summary.lower() or "block" in summary.lower()
        assert "10.0.0.99" in summary

    def test_summary_includes_shap(self):
        shap_top = [("cpu_percent_zscore", 0.42), ("memory_percent_mean", 0.31)]
        summary = generate_summary(
            "cpu_saturation", "order-service", "scale_out",
            severity=0.9,
            shap_top=shap_top,
            context={"cpu": 96, "zscore": 3.2},
        )
        assert "CPU usage" in summary
        assert "Key drivers" in summary

    def test_unknown_fault_still_produces_summary(self):
        summary = generate_summary("weird_fault", "some-service", "alert_human")
        assert "some-service" in summary
        assert len(summary) > 10
