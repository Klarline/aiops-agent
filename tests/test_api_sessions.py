"""Phase C: Run/session isolation and incident ID traceability."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestSessionizedRuns:
    """Sessionized endpoints: two runs execute independently without state collisions."""

    def test_create_run_returns_run_id_and_incident_id(self):
        resp = client.post(
            "/agent/runs",
            json={"scenario_id": "cpu_saturation_order", "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "incident_id" in data
        assert data["status"] == "started"
        assert data["scenario"] == "cpu_saturation_order"
        assert len(data["run_id"]) == 36  # UUID format
        assert len(data["incident_id"]) == 36

    def test_two_runs_isolated(self):
        resp1 = client.post(
            "/agent/runs",
            json={"scenario_id": "cpu_saturation_order", "seed": 42},
        )
        resp2 = client.post(
            "/agent/runs",
            json={"scenario_id": "memory_leak_auth", "seed": 99},
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        run_id1 = resp1.json()["run_id"]
        run_id2 = resp2.json()["run_id"]
        assert run_id1 != run_id2

        # Advance run1 a few steps
        client.post(f"/agent/runs/{run_id1}/step", params={"n_steps": 5})
        status1 = client.get(f"/agent/runs/{run_id1}/status").json()
        status2 = client.get(f"/agent/runs/{run_id2}/status").json()

        # Both should have their own state
        assert status1["run_id"] == run_id1
        assert status2["run_id"] == run_id2
        assert status1["incident_id"] != status2["incident_id"]

    def test_step_includes_incident_id_and_run_id(self):
        resp = client.post(
            "/agent/runs",
            json={"scenario_id": "cpu_saturation_order", "seed": 42},
        )
        run_id = resp.json()["run_id"]
        step_resp = client.post(f"/agent/runs/{run_id}/step", params={"n_steps": 3})
        assert step_resp.status_code == 200
        data = step_resp.json()
        assert data["run_id"] == run_id
        assert "incident_id" in data
        if data["steps"]:
            for s in data["steps"]:
                assert "incident_id" in s
                assert "run_id" in s

    def test_unknown_run_id_404(self):
        resp = client.get("/agent/runs/00000000-0000-0000-0000-000000000000/status")
        assert resp.status_code == 404

    def test_metrics_accept_run_id(self):
        resp = client.post(
            "/agent/runs",
            json={"scenario_id": "cpu_saturation_order", "seed": 42},
        )
        run_id = resp.json()["run_id"]
        client.post(f"/agent/runs/{run_id}/step", params={"n_steps": 2})

        # With run_id
        metrics_resp = client.get("/metrics/current", params={"run_id": run_id})
        assert metrics_resp.status_code == 200

        # Without run_id uses default session (may be empty)
        default_resp = client.get("/metrics/current")
        assert default_resp.status_code == 200

    def test_metrics_unknown_run_id_404(self):
        resp = client.get(
            "/metrics/current",
            params={"run_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404


class TestLegacyBackwardCompat:
    """Legacy endpoints still work with default session."""

    def test_legacy_start_and_step(self):
        resp = client.post(
            "/agent/scenarios/start",
            json={"scenario_id": "cpu_saturation_order", "seed": 42},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

        step_resp = client.post("/agent/step", params={"n_steps": 2})
        assert step_resp.status_code == 200
        assert "steps" in step_resp.json()

    def test_legacy_run_scenario(self):
        resp = client.post(
            "/agent/run-scenario",
            json={"scenario_id": "cpu_saturation_order", "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "steps" in data
        assert "evaluation" in data


class TestAPIKeyProtection:
    """When AIOPS_API_KEY is set, mutating endpoints require valid key."""

    def test_mutating_rejected_without_key_when_configured(self):
        os.environ["AIOPS_API_KEY"] = "secret123"
        try:
            resp = client.post(
                "/agent/scenarios/start",
                json={"scenario_id": "cpu_saturation_order"},
            )
            assert resp.status_code == 401
        finally:
            os.environ.pop("AIOPS_API_KEY", None)

    def test_mutating_accepted_with_x_api_key(self):
        os.environ["AIOPS_API_KEY"] = "secret123"
        try:
            resp = client.post(
                "/agent/scenarios/start",
                json={"scenario_id": "cpu_saturation_order"},
                headers={"X-API-Key": "secret123"},
            )
            assert resp.status_code == 200
        finally:
            os.environ.pop("AIOPS_API_KEY", None)

    def test_mutating_accepted_with_bearer_token(self):
        os.environ["AIOPS_API_KEY"] = "secret123"
        try:
            resp = client.post(
                "/agent/scenarios/start",
                json={"scenario_id": "cpu_saturation_order"},
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200
        finally:
            os.environ.pop("AIOPS_API_KEY", None)
