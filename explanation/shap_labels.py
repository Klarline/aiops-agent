"""Human-readable labels for SHAP feature names.

Translates raw feature names (e.g. error_rate_zscore) into operator-friendly
descriptions with actual metric values when available.
"""

from __future__ import annotations

_METRIC_LABELS: dict[str, str] = {
    "cpu_percent": "CPU usage",
    "memory_percent": "Memory usage",
    "latency_p50_ms": "Latency p50 (ms)",
    "latency_p99_ms": "Latency p99 (ms)",
    "error_rate": "Login failure rate",
    "request_rate": "Request volume",
    "disk_io_percent": "Disk I/O",
    "transactions_per_minute": "Transactions per minute",
}

_SUFFIX_LABELS: dict[str, str] = {
    "_mean": "rolling mean",
    "_std": "volatility",
    "_roc": "rate of change",
    "_zscore": "z-score",
}


def humanize_feature_name(name: str, raw_value: float | None = None) -> str:
    """Convert raw feature name to human-readable label, optionally with value.

    Examples:
        error_rate_zscore, 4.2 -> "Login failure rate (4.2σ above normal)"
        tpm_mean, 3.1 -> "Transactions per minute (3/min, normal ~850)"
        cpu_percent_mean, 94 -> "CPU usage (94%)"
    """
    base = name
    suffix = ""
    for suf in ["_zscore", "_roc", "_std", "_mean"]:
        if name.endswith(suf):
            base = name[: -len(suf)]
            suffix = suf
            break

    metric_label = _METRIC_LABELS.get(base, base.replace("_", " "))
    suffix_label = _SUFFIX_LABELS.get(suffix, suffix.replace("_", " ") if suffix else "")

    if raw_value is not None and suffix_label:
        if suffix == "_zscore":
            return f"{metric_label} ({raw_value:+.1f}σ from normal)"
        if suffix == "_mean":
            if base == "transactions_per_minute":
                return f"{metric_label} ({raw_value:.0f}/min, normal ~850)"
            if base == "error_rate":
                return f"{metric_label} ({raw_value:.1%})"
            if "percent" in base or "cpu" in base or "memory" in base or "disk" in base:
                return f"{metric_label} ({raw_value:.0f}%)"
            if "latency" in base:
                return f"{metric_label} ({raw_value:.0f} ms)"
            if "request" in base or "rate" in base:
                return f"{metric_label} ({raw_value:.0f}/s)"
            if "error" in base:
                return f"{metric_label} ({raw_value:.1%})"
        if suffix == "_roc":
            return f"{metric_label} (change: {raw_value:+.1f}/step)"

    if suffix_label:
        return f"{metric_label} ({suffix_label})"
    return metric_label
