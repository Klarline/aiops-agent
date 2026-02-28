"""Tests for FastAPI backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestAPI:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "AIOps" in response.json()["name"]

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "actions_taken" in data
        assert "mode" in data
        assert "uptime_seconds" in data
        assert "running" in data
        assert "detection_step" in data
        assert "decision_latency_ms" in data

    def test_agent_health(self):
        """Agent self-observability endpoint returns full payload."""
        response = client.get("/agent/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "actions_taken" in data
        assert "mode" in data
        assert "uptime_seconds" in data
        assert "running" in data
        assert "detection_step" in data
        assert "decision_latency_ms" in data

    def test_list_scenarios(self):
        response = client.get("/agent/scenarios")
        assert response.status_code == 200
        scenarios = response.json()["scenarios"]
        assert len(scenarios) >= 4

    def test_topology(self):
        response = client.get("/metrics/topology")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 5
