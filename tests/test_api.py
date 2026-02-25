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
        assert response.json()["status"] == "healthy"

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
