"""Report generator for benchmark results.

Produces JSON results, plots (confusion matrix, SHAP waterfall, MTTR),
and markdown summaries. Supports multi-agent leaderboard.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from evaluation.metrics_calculator import (
    format_leaderboard,
    compute_per_scenario_breakdown,
    compute_precision_recall,
    compute_calibration_buckets,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def generate_report(benchmark_results: dict[str, Any]) -> dict[str, Any]:
    """Generate full evaluation report from benchmark results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    serializable = _make_serializable(benchmark_results)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    leaderboard = _build_leaderboard(benchmark_results)
    breakdown = compute_per_scenario_breakdown(benchmark_results["raw_results"])

    try:
        _generate_plots(benchmark_results, breakdown)
    except Exception:
        pass

    markdown = _generate_markdown(benchmark_results, leaderboard, breakdown)

    return {
        "results_path": results_path,
        "leaderboard": leaderboard,
        "markdown": markdown,
    }


def generate_leaderboard_report(
    leaderboard_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate comparative report from multi-agent leaderboard run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    agent_aggregates = {name: data["aggregate"] for name, data in leaderboard_data.items()}
    leaderboard_text = format_leaderboard(agent_aggregates)

    calibration_data: dict[str, Any] = {}
    for name, data in leaderboard_data.items():
        raw = data.get("raw_results", [])
        calibration_data[name] = {
            "precision_recall": compute_precision_recall(raw),
            "calibration_buckets": compute_calibration_buckets(raw),
        }

    results_path = os.path.join(RESULTS_DIR, "leaderboard_results.json")
    serializable = _make_serializable(
        {
            name: {
                "aggregate": data["aggregate"],
                "per_scenario": data["per_scenario"],
                "calibration": calibration_data.get(name, {}),
            }
            for name, data in leaderboard_data.items()
        }
    )
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    try:
        _generate_leaderboard_plots(leaderboard_data)
    except Exception as e:
        print(f"  Warning: plot generation failed: {e}")

    return {
        "results_path": results_path,
        "leaderboard": leaderboard_text,
        "calibration": calibration_data,
    }


def save_expected_results(benchmark_results: dict[str, Any]) -> str:
    """Save expected results for reproducibility testing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "expected_results.json")
    expected = {
        "seed": benchmark_results["seed"],
        "aggregate": benchmark_results["aggregate"],
        "per_scenario": benchmark_results["per_scenario"],
    }
    with open(path, "w") as f:
        json.dump(expected, f, indent=2)
    return path


def _build_leaderboard(results: dict) -> str:
    agents = {
        "Expert Policy": results["aggregate"],
    }
    return format_leaderboard(agents)


def _generate_plots(results: dict, breakdown: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    _plot_scenario_heatmap(breakdown, plt, sns)
    _plot_mttr_chart(results, plt)


def _generate_leaderboard_plots(leaderboard_data: dict[str, dict]) -> None:
    """Generate all comparative plots for the multi-agent leaderboard."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    ml_data = leaderboard_data.get("ML Ensemble Agent", {})
    if ml_data:
        breakdown = compute_per_scenario_breakdown(ml_data.get("raw_results", []))
        _plot_scenario_heatmap(breakdown, plt, sns)
        _plot_mttr_chart(ml_data, plt)

    _plot_leaderboard_comparison(leaderboard_data, plt)
    _plot_per_task_comparison(leaderboard_data, plt)

    try:
        _plot_shap_waterfall(plt)
    except Exception:
        pass


def _plot_scenario_heatmap(breakdown: dict, plt, sns) -> None:
    """Per-scenario × per-task accuracy heatmap."""
    scenarios = list(breakdown.keys())
    tasks = ["detection", "localization", "diagnosis", "mitigation"]
    data = np.array([[breakdown[s][t] for t in tasks] for s in scenarios])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        xticklabels=[t.capitalize() for t in tasks],
        yticklabels=[s.replace("_", " ").title() for s in scenarios],
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("Per-Scenario Accuracy (AIOpsLab Format)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_mttr_chart(results: dict, plt) -> None:
    """Mean Time To Remediation bar chart."""
    mttr = results.get("mttr_by_scenario", {})
    if not mttr:
        return

    scenarios = list(mttr.keys())
    times = [mttr[s] for s in scenarios]
    labels = [s.replace("_", " ").title() for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(scenarios)))
    bars = ax.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Time to Remediation (seconds)", fontsize=11)
    ax.set_title("MTTR by Scenario", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f"{val:.0f}s", va="center", fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mttr_by_scenario.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_leaderboard_comparison(leaderboard_data: dict, plt) -> None:
    """Side-by-side bar chart of agent aggregate scores."""
    agents = list(leaderboard_data.keys())
    tasks = ["detection", "localization", "diagnosis", "mitigation"]
    x = np.arange(len(tasks))
    width = 0.8 / len(agents)
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, agent in enumerate(agents):
        agg = leaderboard_data[agent]["aggregate"]
        vals = [agg[t] for t in tasks]
        offset = (i - len(agents) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, label=agent, color=colors[i % len(colors)], edgecolor="white", linewidth=0.5
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Agent Comparison — AIOpsLab Leaderboard", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "leaderboard_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_per_task_comparison(leaderboard_data: dict, plt) -> None:
    """Per-scenario comparison between ML agent and baselines."""
    ml_data = leaderboard_data.get("ML Ensemble Agent", {})
    static_data = leaderboard_data.get("Static Threshold", {})
    if not ml_data or not static_data:
        return

    ml_scenarios = ml_data.get("per_scenario", {})
    st_scenarios = static_data.get("per_scenario", {})
    scenarios = list(ml_scenarios.keys())
    if not scenarios:
        return

    ml_avgs = [ml_scenarios[s]["average"] for s in scenarios]
    st_avgs = [st_scenarios.get(s, {}).get("average", 0) for s in scenarios]
    labels = [s.replace("_", " ").title() for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.35

    ax.barh(x - width / 2, ml_avgs, width, label="ML Ensemble Agent", color="#2ecc71", edgecolor="white")
    ax.barh(x + width / 2, st_avgs, width, label="Static Threshold", color="#e74c3c", edgecolor="white")

    ax.set_xlabel("Average Score", fontsize=11)
    ax.set_title("Per-Scenario: ML Agent vs Static Threshold", fontsize=14, fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_scenario_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_shap_waterfall(plt) -> None:
    """Generate SHAP waterfall plots for 3 representative incidents.

    Runs a short simulation for memory_leak, cpu_saturation, and brute_force,
    extracts features at the fault peak, and generates SHAP explanations.
    """
    try:
        import shap
    except ImportError:
        return

    from simulator.service_topology import build_topology
    from simulator.metrics_generator import generate_metrics
    from simulator.fault_injector import FaultScenario, inject_fault
    from features.feature_extractor import extract_features, get_feature_names
    from detection.isolation_forest import IsolationForestDetector
    from features.feature_extractor import extract_features_batch

    topology = build_topology()
    normal = generate_metrics(topology, 3600, 10, seed=42)

    all_feats = []
    for svc, df in normal.items():
        feats = extract_features_batch(df, window_size=6)
        all_feats.append(feats)
    training_features = np.vstack(all_feats)
    training_features = training_features[~np.isnan(training_features).any(axis=1)]

    model = IsolationForestDetector()
    model.fit(training_features)
    explainer = shap.TreeExplainer(model.model)

    feature_names = get_feature_names()

    representative_incidents = [
        (
            "Memory Leak",
            FaultScenario(
                fault_type="memory_leak",
                target_service="auth-service",
                start_time=600,
                duration=1200,
                severity=0.8,
            ),
            "auth-service",
        ),
        (
            "CPU Saturation",
            FaultScenario(
                fault_type="cpu_saturation",
                target_service="order-service",
                start_time=600,
                duration=1200,
                severity=0.8,
            ),
            "order-service",
        ),
        (
            "Brute Force",
            FaultScenario(
                fault_type="brute_force",
                target_service="auth-service",
                start_time=600,
                duration=1200,
                severity=0.8,
            ),
            "auth-service",
        ),
    ]

    for title, scenario, target_service in representative_incidents:
        faulty = inject_fault(
            {s: df.copy() for s, df in normal.items()},
            scenario,
            topology,
        )

        peak_step = int((scenario.start_time + scenario.duration * 0.7) / 10)
        target_df = faulty[target_service]
        if peak_step >= len(target_df):
            continue

        window_start = max(0, peak_step - 6)
        window = target_df.iloc[window_start : peak_step + 1]
        features = extract_features(window, window_size=6)

        shap_values = explainer.shap_values(features.reshape(1, -1))
        vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        abs_vals = np.abs(vals)
        top_k = 10
        top_indices = np.argsort(abs_vals)[-top_k:][::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        names = [feature_names[i] for i in top_indices]
        contributions = [vals[i] for i in top_indices]
        colors = ["#e74c3c" if c > 0 else "#3498db" for c in contributions]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, contributions, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("SHAP Value (impact on anomaly score)", fontsize=10)
        ax.set_title(f"SHAP Waterfall — {title} ({target_service})", fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        fname = f"shap_waterfall_{scenario.fault_type}.png"
        plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()


def _generate_markdown(results: dict, leaderboard: str, breakdown: dict) -> str:
    lines = [
        "# Evaluation Results",
        "",
        "## Aggregate Scores (AIOpsLab Leaderboard Format)",
        "",
        "```",
        leaderboard,
        "```",
        "",
        "## Per-Scenario Breakdown",
        "",
    ]
    for sid, scores in breakdown.items():
        det = scores["detection"]
        loc = scores["localization"]
        diag = scores["diagnosis"]
        mit = scores["mitigation"]
        lines.append(f"  {sid:35s} Det={det:.0%} Loc={loc:.0%} Diag={diag:.0%} Mit={mit:.0%}")

    lines.extend(
        [
            "",
            "## Configuration",
            f"- Seed: {results['seed']}",
            f"- Episodes per scenario: {results['episodes_per_scenario']}",
            f"- Total episodes: {results['total_episodes']}",
        ]
    )

    return "\n".join(lines)


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
