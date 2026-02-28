"""Structured incident report generation.

Produces post-incident markdown artifacts suitable for managers
and compliance officers — maps to JD "business communication" requirements.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from orchestrator.orchestrator import Orchestrator
from orchestrator.scenario_registry import get_scenario


def generate_incident_report(
    orch: Orchestrator,
    scenario_id: str,
    episode: int,
    incident_id: str,
) -> str:
    """Generate a structured markdown incident report from an episode.

    Args:
        orch: Orchestrator after run_episode (has history, env).
        scenario_id: Scenario identifier.
        episode: Episode index.
        incident_id: Unique incident ID (e.g. INC-2024-0042).

    Returns:
        Markdown string.
    """
    scenario = get_scenario(scenario_id)
    fault_type = scenario.fault_type
    target = scenario.target_service
    source_ip = scenario.metadata.get("source_ip", "unknown")

    # Detection time (simplified: fault starts at scenario.start_time)
    detected_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Actions taken
    actions_taken: list[str] = []
    for entry in orch.history:
        action = entry.get("action")
        if action is None:
            continue
        act = getattr(action, "action", str(action))
        tgt = getattr(action, "target", "")
        expl = getattr(action, "explanation", "")[:80]
        if act == "block_ip":
            ip = getattr(action, "details", {}).get("ip", source_ip)
            actions_taken.append(f"Blocked IP {ip} on {tgt}")
        elif act == "rate_limit":
            actions_taken.append(f"Rate limiting applied on {tgt}")
        elif act == "restart_service":
            actions_taken.append(f"Restarted {tgt}")
        elif act == "scale_out":
            actions_taken.append(f"Scaled out {tgt}")
        elif act == "rollback":
            actions_taken.append(f"Rolled back {tgt}")
        elif act == "alert_human":
            actions_taken.append(f"Escalated to human operator: {expl}...")
        else:
            actions_taken.append(f"{act} on {tgt}")

    # MTTR
    mttr_seconds = 0
    if orch.history:
        first_step = orch.history[0].get("step", 0)
        mttr_seconds = first_step * 10  # 10s per step

    # Severity from last action
    severity = "High"
    confidence = 0.85
    if orch.history:
        last_action = orch.history[-1].get("action")
        if last_action:
            conf = getattr(last_action, "confidence", 0.85)
            confidence = conf
            if conf >= 0.9:
                severity = "Critical"
            elif conf >= 0.7:
                severity = "High"
            elif conf >= 0.5:
                severity = "Medium"
            else:
                severity = "Low"

    # Root cause description
    if "brute_force" in fault_type:
        root_cause = f"Brute force attack on {target} from {source_ip}"
    elif "transaction_stall" in fault_type:
        root_cause = f"Transaction stall on {target} — TPM collapsed, infra healthy"
    elif "memory_leak" in fault_type:
        root_cause = f"Memory leak on {target}"
    elif "cpu_saturation" in fault_type:
        root_cause = f"CPU saturation on {target}"
    elif "ddos" in fault_type:
        root_cause = f"DDoS attack on {target}"
    elif "deployment_regression" in fault_type:
        root_cause = f"Post-deployment regression on {target}"
    elif "cascading_failure" in fault_type:
        root_cause = f"Cascading failure originating from {target}"
    elif "anomalous_access" in fault_type:
        root_cause = f"Anomalous access pattern on {target}"
    else:
        root_cause = f"{fault_type} on {target}"

    # Metrics table (from env if available)
    metrics_rows: list[str] = []
    if orch.env.scenario and hasattr(orch.env, "metrics_data") and orch.env.metrics_data:
        # Use last known metrics from target service
        for svc, df in orch.env.metrics_data.items():
            if svc == target and df is not None and len(df) > 0:
                row = df.iloc[-1]
                for col in ["error_rate", "request_rate", "cpu_percent", "transactions_per_minute"]:
                    if col in row:
                        val = row[col]
                        if col == "error_rate":
                            val_str = f"{float(val):.2%}"
                        elif col in ("cpu_percent", "memory_percent", "disk_io_percent"):
                            val_str = f"{float(val):.0f}%"
                        else:
                            val_str = f"{float(val):.0f}"
                        metrics_rows.append(f"| {col} | {val_str} | — | — |")
                break
    if not metrics_rows:
        metrics_rows = [
            "| error_rate | — | 0.01 | — |",
            "| request_rate | — | 200/s | — |",
            "| cpu_percent | — | 35% | — |",
        ]

    # Audit entries
    audit_refs: list[str] = []
    if hasattr(orch.env, "audit_log") and orch.env.audit_log:
        for i, entry in enumerate(orch.env.audit_log):
            ev = entry.get("event", "action")
            if ev == "ip_blocked":
                audit_refs.append(f"IP {entry.get('ip', '?')} blocked (AUD-{incident_id[-4:]}-{i+1:03d})")
            elif ev == "rate_limit_applied":
                audit_refs.append(f"Rate limit applied (AUD-{incident_id[-4:]}-{i+1:03d})")

    actions_section = "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions_taken))
    if audit_refs:
        actions_section += "\n" + "\n".join(f"   - {r}" for r in audit_refs)

    report = f"""## Incident Report — {incident_id}

**Detected:** {detected_ts}
**Root Cause:** {root_cause}
**Severity:** {severity} | **Confidence:** {confidence:.0%}
**MTTR:** {mttr_seconds} seconds

### Detection

Scenario: {scenario_id}. Agent detected anomaly on {target}.

### Actions Taken

{actions_section}

### Metrics at Detection

| Metric | Value | Baseline | Status |
|--------|-------|----------|--------|
{chr(10).join(metrics_rows)}

### Recommendation

Review in operations dashboard. Re-run `python scripts/demo_security.py` or `demo_transaction_stall.py` to reproduce.
"""
    return report


def save_incident_report(
    report: str,
    output_dir: Path | str,
    incident_id: str,
) -> Path:
    """Save report to evaluation/results/incidents/."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{incident_id}.md"
    path.write_text(report, encoding="utf-8")
    return path
