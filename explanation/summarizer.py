"""Natural language alert summarizer.

Template-based NL generation — no LLM required. Produces human-readable
summaries for each fault type with detection context and recommended actions.
"""

from __future__ import annotations

from typing import Any


_TEMPLATES = {
    "memory_leak": (
        "Detected gradual memory increase on {service}, currently at {memory:.0f}% "
        "(rising ~{rate:.1f}%/min). {shap_context} "
        "Action: {action}. "
        "Severity: {severity}."
    ),
    "cpu_saturation": (
        "Detected sudden CPU spike on {service} at {timestamp}, reaching {cpu:.0f}% "
        "({zscore:.1f}\u03c3 above baseline). {shap_context} "
        "Action: {action}. "
        "Status: {status}."
    ),
    "brute_force": (
        "Security alert: Detected {error_count} failed auth attempts"
        "{ip_context} in {window}. {shap_context} "
        "Action: {action}. Audit entry created."
    ),
    "transaction_stall": (
        "\u26a0 Silent failure detected: {service} infrastructure appears healthy "
        "(CPU: {cpu:.0f}%, Memory: {memory:.0f}%) but transaction throughput "
        "dropped to {tpm:.0f}/min (normal: {tpm_baseline:.0f}/min). {shap_context} "
        "Action: {action}."
    ),
    "cascading_failure": (
        "Cascading failure detected originating from {service}. "
        "{affected_count} downstream services affected. "
        "Latency increased {latency_mult:.1f}x. {shap_context} "
        "Action: {action}."
    ),
    "deployment_regression": (
        "Post-deployment regression on {service}: latency +{latency_increase:.0f}%, "
        "error rate +{error_increase:.1f}%. {shap_context} "
        "Action: {action}."
    ),
    "anomalous_access": (
        "Security alert: Anomalous access pattern detected on {service}. "
        "{pattern_desc} {shap_context} "
        "Action: {action}."
    ),
    "ddos": (
        "DDoS attack detected on {service}: request rate {request_mult:.0f}x normal. "
        "Latency degraded {latency_mult:.1f}x. {shap_context} "
        "Action: {action}."
    ),
}

_ACTION_LABELS = {
    "restart_service": "Restart service",
    "scale_out": "Scale to additional replicas",
    "rollback": "Rollback to previous version",
    "block_ip": "IP blocked",
    "rate_limit": "Rate limiting applied",
    "alert_human": "High-priority alert to operations team",
    "continue_monitoring": "Continue monitoring",
}

_SEVERITY_LABELS = {
    (0.0, 0.3): "Low",
    (0.3, 0.6): "Medium",
    (0.6, 0.8): "High",
    (0.8, 1.01): "Critical",
}


def _severity_label(severity: float) -> str:
    for (lo, hi), label in _SEVERITY_LABELS.items():
        if lo <= severity < hi:
            return label
    return "Unknown"


def _shap_context(shap_top: list[tuple[str, float]] | None) -> str:
    if not shap_top:
        return ""
    parts = [f"{name} ({val:+.2f})" for name, val in shap_top[:3]]
    return "Key drivers: " + ", ".join(parts) + "."


def generate_summary(
    fault_type: str,
    service: str,
    action: str,
    severity: float = 0.5,
    shap_top: list[tuple[str, float]] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Generate a natural-language alert summary.

    Args:
        fault_type: Diagnosed fault type string.
        service: Localized service name.
        action: Remediation action taken.
        severity: Fault severity 0-1.
        shap_top: Top SHAP feature attributions [(name, value), ...].
        context: Additional context values for template rendering.

    Returns:
        Human-readable summary string.
    """
    ctx = context or {}
    template = _TEMPLATES.get(fault_type)
    if template is None:
        return (
            f"Anomaly detected on {service} (type: {fault_type}). "
            f"{_shap_context(shap_top)} Action: {_ACTION_LABELS.get(action, action)}."
        )

    defaults = {
        "service": service,
        "action": _ACTION_LABELS.get(action, action),
        "severity": _severity_label(severity),
        "shap_context": _shap_context(shap_top),
        "timestamp": ctx.get("timestamp", "now"),
        "cpu": ctx.get("cpu", 0),
        "memory": ctx.get("memory", 0),
        "rate": ctx.get("rate", 0.5),
        "zscore": ctx.get("zscore", 3.0),
        "status": ctx.get("status", "Metrics recovering"),
        "error_count": ctx.get("error_count", 0),
        "ip_context": f" from {ctx['source_ip']}" if "source_ip" in ctx else "",
        "window": ctx.get("window", "30 seconds"),
        "tpm": ctx.get("tpm", 0),
        "tpm_baseline": ctx.get("tpm_baseline", 850),
        "affected_count": ctx.get("affected_count", 0),
        "latency_mult": ctx.get("latency_mult", 2.0),
        "latency_increase": ctx.get("latency_increase", 50),
        "error_increase": ctx.get("error_increase", 0.1),
        "pattern_desc": ctx.get("pattern_desc", "Unusual query volume detected."),
        "request_mult": ctx.get("request_mult", 10),
    }

    try:
        return template.format(**defaults)
    except KeyError:
        return (
            f"Anomaly detected on {service} (type: {fault_type}). "
            f"{_shap_context(shap_top)} Action: {_ACTION_LABELS.get(action, action)}."
        )
