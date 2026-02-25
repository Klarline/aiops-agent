"""Tests for root cause service localization."""

from __future__ import annotations

from diagnosis.localizer import ServiceLocalizer


class TestServiceLocalizer:
    def test_single_anomalous_service(self, topology):
        localizer = ServiceLocalizer()
        scores = {
            "api-gateway": 0.1,
            "auth-service": 0.8,
            "order-service": 0.1,
            "user-db": 0.05,
            "order-db": 0.05,
        }
        result = localizer.localize(scores, topology)
        assert result == "auth-service"

    def test_upstream_preferred(self, topology):
        localizer = ServiceLocalizer()
        scores = {
            "api-gateway": 0.7,
            "auth-service": 0.6,
            "order-service": 0.5,
            "user-db": 0.4,
            "order-db": 0.3,
        }
        result = localizer.localize(scores, topology)
        assert result == "api-gateway"

    def test_only_downstream_anomalous(self, topology):
        localizer = ServiceLocalizer()
        scores = {
            "api-gateway": 0.1,
            "auth-service": 0.1,
            "order-service": 0.1,
            "user-db": 0.8,
            "order-db": 0.1,
        }
        result = localizer.localize(scores, topology)
        assert result == "user-db"

    def test_no_anomalous_returns_highest(self, topology):
        localizer = ServiceLocalizer()
        scores = {
            "api-gateway": 0.2,
            "auth-service": 0.1,
            "order-service": 0.25,
            "user-db": 0.05,
            "order-db": 0.05,
        }
        result = localizer.localize(scores, topology)
        assert result == "order-service"
