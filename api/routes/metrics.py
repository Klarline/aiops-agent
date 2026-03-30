"""Metrics API routes — uses the shared simulation state."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.shared_state import run_registry, state
from simulator.service_topology import build_topology

router = APIRouter()


def _get_session(run_id: str | None):
    """Return session for run_id, or default session when omitted."""
    if run_id is None:
        return state
    session = run_registry.get_session(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return session


@router.get("/current")
async def get_current_metrics(run_id: str | None = None):
    """Return current metrics. Optional run_id for sessionized runs."""
    session = _get_session(run_id)
    orch = session.orchestrator
    if orch is None or orch.env.metrics_data is None:
        return {}
    return orch.env.get_current_metrics()


@router.get("/history/{service}")
async def get_metrics_history(service: str, lookback: int = 60, run_id: str | None = None):
    """Return metrics history for a service. Optional run_id for sessionized runs."""
    session = _get_session(run_id)
    orch = session.orchestrator
    if orch is None or orch.env.metrics_data is None:
        return {"error": "No scenario running."}
    try:
        df = orch.env.get_metrics_history(service, lookback_steps=lookback)
        return {
            "service": service,
            "timestamps": [str(t) for t in df.index],
            "metrics": df.to_dict(orient="list"),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/topology")
async def get_topology(run_id: str | None = None):
    """Return service topology. In live mode, returns the Prometheus-discovered graph."""
    session = _get_session(run_id)
    orch = session.orchestrator
    if orch is not None:
        graph = orch.env.get_topology()
    else:
        graph = build_topology()
    nodes = [{"id": n, **graph.nodes[n]} for n in graph.nodes]
    edges = [{"source": u, "target": v} for u, v in graph.edges]
    return {"nodes": nodes, "edges": edges}


@router.get("/drift")
async def get_drift_report(run_id: str | None = None):
    """Feature distribution drift vs training baseline.

    Returns PSI scores per feature and overall drift severity.
    Only meaningful in live mode with sufficient history.
    """
    import os
    import numpy as np

    aiops_mode = os.getenv("AIOPS_MODE", "simulated").lower()
    if aiops_mode != "live":
        return {"mode": "simulated", "message": "Drift monitoring only applies in live mode"}

    session = _get_session(run_id)
    orch = session.orchestrator
    if orch is None:
        return {"error": "No active session"}

    env = orch.env
    history = getattr(env, "_history", {})
    if not history:
        return {"error": "No live history collected yet"}

    try:
        import pandas as pd
        from features.feature_extractor import extract_features_batch, get_feature_names
        from environments.drift_detector import MetricDriftMonitor
        from simulator.service_topology import build_topology
        from simulator.metrics_generator import generate_metrics

        feature_names = get_feature_names()
        monitor = MetricDriftMonitor(feature_names)

        # Fit reference from simulated baseline
        topology = build_topology()
        sim_metrics = generate_metrics(topology, 3600, 10, seed=42)
        sim_features = []
        for _, df in sim_metrics.items():
            feats = extract_features_batch(df, window_size=6)
            sim_features.append(feats)
        ref_features = np.vstack(sim_features)
        ref_features = ref_features[~np.isnan(ref_features).any(axis=1)]
        monitor.fit(ref_features)

        # Score live data
        live_features = []
        for svc, snapshots in history.items():
            if len(snapshots) >= 6:
                df = pd.DataFrame(snapshots)
                feats = extract_features_batch(df, window_size=6)
                valid = feats[~np.isnan(feats).any(axis=1)]
                if len(valid):
                    live_features.append(valid)

        if not live_features:
            return {"message": "Not enough live history for drift analysis (need ≥6 steps)"}

        live_array = np.vstack(live_features)
        report = monitor.score(live_array)

        return {
            "severity": report.severity,
            "max_psi": report.max_psi,
            "mean_psi": report.mean_psi,
            "drifted_features": report.drifted_features,
            "psi_scores": report.psi_scores,
            "thresholds": {"warning": 0.1, "critical": 0.2},
        }
    except Exception as exc:
        return {"error": str(exc)}
