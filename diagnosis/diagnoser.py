"""Fault type diagnosis from feature patterns.

AIOpsLab Diagnosis task: Given features and the localized service,
classify the fault type using pattern matching on metric signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Diagnosis:
    """Result of fault diagnosis."""

    fault_type: str
    confidence: float
    localized_service: str
    severity: float
    is_security: bool


class FaultDiagnoser:
    """Classify fault type from feature and metric patterns."""

    def diagnose(
        self,
        metrics_snapshot: dict[str, dict[str, float]],
        metrics_history: pd.DataFrame | None,
        localized_service: str,
        anomaly_score: float,
    ) -> Diagnosis:
        """Pattern-match to classify the fault type.

        Args:
            metrics_snapshot: Current metrics for all services.
            metrics_history: Recent history for the localized service.
            localized_service: Which service was identified as root cause.
            anomaly_score: Overall anomaly score from the ensemble.
        """
        current = metrics_snapshot.get(localized_service, {})

        checks = [
            self._check_transaction_stall,
            self._check_brute_force,
            self._check_ddos,
            self._check_cascading_failure,
            self._check_cpu_saturation,
            self._check_deployment_regression,
            self._check_anomalous_access,
            self._check_memory_leak,
        ]

        for check in checks:
            result = check(current, metrics_history, localized_service, metrics_snapshot)
            if result is not None:
                result.severity = max(min(anomaly_score, 1.0), 0.1)
                return result

        return Diagnosis(
            fault_type="unknown",
            confidence=0.3,
            localized_service=localized_service,
            severity=min(anomaly_score, 1.0),
            is_security=False,
        )

    def _check_transaction_stall(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        tpm = current.get("transactions_per_minute", 999)
        cpu = current.get("cpu_percent", 50)
        mem = current.get("memory_percent", 50)
        lat = current.get("latency_p50_ms", 50)

        if tpm < 100 and cpu < 75 and mem < 80 and lat < 150:
            return Diagnosis("transaction_stall", 0.9, service, 0.0, False)
        return None

    def _check_brute_force(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        error = current.get("error_rate", 0)
        req = current.get("request_rate", 0)

        if error > 0.25 and "auth" in service.lower():
            return Diagnosis("brute_force", 0.85, service, 0.0, True)
        if error > 0.15 and req > 400 and "auth" in service.lower():
            return Diagnosis("brute_force", 0.75, service, 0.0, True)
        return None

    def _check_ddos(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        req = current.get("request_rate", 0)
        lat = current.get("latency_p50_ms", 0)

        if req > 1500 and ("gateway" in service or "api" in service):
            return Diagnosis("ddos", 0.85, service, 0.0, True)
        if req > 1000 and lat > 40 and ("gateway" in service or "api" in service):
            return Diagnosis("ddos", 0.75, service, 0.0, True)
        return None

    def _check_cpu_saturation(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        cpu = current.get("cpu_percent", 0)
        if cpu > 75:
            return Diagnosis("cpu_saturation", 0.9, service, 0.0, False)
        return None

    def _check_memory_leak(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        mem = current.get("memory_percent", 0)
        err = current.get("error_rate", 0)
        lat = current.get("latency_p50_ms", 0)
        if err > 0.06 or lat > 80:
            return None
        if history is not None and len(history) >= 6:
            mem_series = history["memory_percent"]
            early = mem_series.iloc[:3].mean()
            late = mem_series.iloc[-3:].mean()
            trend = late - early
            if trend > 5.0:
                return Diagnosis("memory_leak", 0.85, service, 0.0, False)
            if trend > 3.0 and late > 42:
                return Diagnosis("memory_leak", 0.75, service, 0.0, False)
        if mem > 65:
            return Diagnosis("memory_leak", 0.6, service, 0.0, False)
        return None

    def _check_deployment_regression(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        if history is not None and len(history) >= 6:
            lat = history["latency_p50_ms"]
            err = history["error_rate"]
            early_lat = lat.iloc[:3].mean()
            late_lat = lat.iloc[-3:].mean()
            late_err = err.iloc[-3:].mean()
            if late_lat > early_lat * 1.3 and late_err > 0.03:
                return Diagnosis("deployment_regression", 0.75, service, 0.0, False)
        lat_now = current.get("latency_p50_ms", 0)
        err_now = current.get("error_rate", 0)
        if lat_now > 80 and err_now > 0.05:
            return Diagnosis("deployment_regression", 0.6, service, 0.0, False)
        return None

    def _check_anomalous_access(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        error = current.get("error_rate", 0)
        lat_p99 = current.get("latency_p99_ms", 0)
        req = current.get("request_rate", 0)
        if "db" in service.lower():
            if error > 0.03 and lat_p99 > 28:
                return Diagnosis("anomalous_access", 0.65, service, 0.0, True)
            if error > 0.02 and lat_p99 > 25 and req > 300:
                return Diagnosis("anomalous_access", 0.55, service, 0.0, True)
        return None

    def _check_cascading_failure(
        self, current, history, service, all_metrics
    ) -> Diagnosis | None:
        anomalous_count = 0
        high_latency_count = 0
        for svc, metrics in all_metrics.items():
            err = metrics.get("error_rate", 0)
            lat = metrics.get("latency_p50_ms", 0)
            if err > 0.03 or lat > 60:
                anomalous_count += 1
            if lat > 40:
                high_latency_count += 1
        if anomalous_count >= 3 or high_latency_count >= 3:
            return Diagnosis("cascading_failure", 0.7, service, 0.0, False)
        return None
