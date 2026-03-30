# Adaptive AIOps Agent

[![CI](https://github.com/Klarline/aiops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Klarline/aiops-agent/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A security-aware autonomous operations agent for protecting and stabilizing business-critical transaction systems.

> CPU: 32%. Memory: 45%. Latency: normal. Error rate: 0.1%.
> Every dashboard says healthy. But the transaction pipeline has silently stopped.
>
> This agent catches that.

![Transaction stall detection](docs/screenshot_transaction_stall.png)

**67-point gap** vs static threshold monitoring. ML Ensemble Agent: **97%** average across Detection, Localization, Diagnosis, Mitigation. See [EVALUATION.md](EVALUATION.md).

## What It Does

Protects transaction processing pipelines from silent failures, security threats, and cascading infrastructure issues — with full audit trails and explainable AI. This agent detects what threshold monitoring misses.

## Use Case

In any business where transactions drive revenue — payments, insurance, e-commerce — a silent pipeline stall means missed SLAs and regulatory exposure while every dashboard says healthy. The security scenarios (brute force detection, IP blocking, audit trails) address the compliance requirements common in regulated financial and legal data environments. The system's security-first design enables fast automated response with full traceability for compliance review.

## Quick Start

```bash
# Docker (recommended)
docker-compose up -d
open http://localhost:3000

# Local
pip install -r requirements.txt
python scripts/run_benchmark.py --fast
python -m uvicorn api.main:app --reload
open http://localhost:8000/docs
```

## Inspired by AIOpsLab

This project is inspired by [Microsoft Research's AIOpsLab framework](https://github.com/microsoft/AIOpsLab), which provides a holistic framework for evaluating AI agents for autonomous cloud operations.

**What we adopted:**
- **Task taxonomy**: Detection → Localization → Diagnosis → Mitigation
- **Agent-Cloud Interface (ACI)**: Clean separation between agent and environment through an orchestrator
- **Evaluation methodology**: Scoring each task independently, presented in leaderboard format
- **Iterative agent loop**: Observe → Think → Act pattern

**What we built differently:**
- Lightweight Python simulator instead of Kubernetes deployment (enables rapid experimentation)
- Added security-focused fault scenarios (brute force, DDoS)
- Added business logic faults (transaction stall — the "silent failure")
- Added SHAP explainability and natural language summaries
- Included Q-learning for adaptive remediation

## Architecture

```
Orchestrator (ACI Pattern)
├── LLM Agent (ReAct: Reason → Act → Observe)
│   ├── Tool: get_metrics       → Ensemble anomaly detector
│   ├── Tool: explain_anomaly   → SHAP feature attribution
│   ├── Tool: localize_root_cause → Dependency graph traversal
│   ├── Tool: diagnose          → Pattern-based fault classifier
│   ├── Tool: restart/scale/block/rollback → Remediation actions
│   └── Fallback: Rule-based policy (always available)
└── Environment (BaseEnvironment interface)
    ├── SimulatedEnvironment  — built-in fault simulator (default, for demos/benchmarks)
    └── PrometheusEnvironment — live Prometheus scraping (set AIOPS_MODE=live)
```

The agent and orchestrator are environment-agnostic: they work identically whether the data comes from the simulator or a real Prometheus server. The LLM agent reasons step-by-step, calling ML modules as tools. When the LLM is unavailable or errors, the agent seamlessly falls back to a reliable rule-based pipeline.

## AI Techniques

- **LLM ReAct Agent** — decides *what* to investigate (e.g., check topology before diagnosing). Falls back to rules if unavailable.
- **Isolation Forest + Statistical (CUSUM)** — multivariate anomaly detection; gradual drift; model disagreement = uncertainty.
- **SHAP TreeExplainer** — explains *why* an alert fired: "error_rate_zscore +4.2 contributed +0.38."
- **NetworkX graph traversal** — root cause localization via dependency DAG.

See [DESIGN.md](DESIGN.md) for rationale.

## Fault Scenarios (8 Total)

| Category | Fault | Key Challenge |
|----------|-------|---------------|
| Operational | Memory leak | Gradual drift detection |
| Operational | CPU saturation | Sudden spike + downstream impact |
| Operational | Cascading failure | Multi-service root cause tracing |
| Operational | Deployment regression | Post-deploy metric shift |
| Security | Brute force | Attack detection + IP blocking + audit |
| Security | Anomalous access | Subtle pattern recognition |
| Security | DDoS | Volume spike + rate limiting |
| Business Logic | **Transaction stall** | **Silent failure — infra healthy, business broken** |

## Results (Reproducible)

```
AGENT                 DETECTION  LOCALIZATION  DIAGNOSIS  MITIGATION  AVERAGE
ML Ensemble Agent         100%          94%       94%       100%   97.1%
Static Threshold           44%          38%       12%        25%   29.8%
Random Agent               62%          12%       12%        62%   37.5%
```

The **67-point gap** between the ML agent and static threshold comes from the ensemble catching multivariate patterns that no single threshold can express. Run `python scripts/run_benchmark.py --leaderboard` to reproduce. See [EVALUATION.md](EVALUATION.md) for per-scenario analysis.

## Security Features

- Brute force detection → IP blocking + audit log entry
- DDoS detection → automated rate limiting
- Anomalous access → human escalation
- All security actions create audit trail entries

## Testing (4 Layers)

```bash
pytest tests/ -m "not slow" -v          # Fast: unit + integration
pytest tests/ -m "statistical" -v       # ML model quality
pytest tests/ -m "slow" -v              # Full benchmark
```

## Known Limitations

- **Cascading failure localization**: Lag correlation is insufficient when propagation delays vary across services. Distributed tracing span IDs would enable causal ordering and improve accuracy here.
- **Anomalous access**: The signal is intentionally subtle (error_rate +3%, latency_p99 +40%), causing ~38% localization misses. Per-service adaptive thresholds calibrated to historical baselines would reduce this.
- **Transaction stall mitigation**: The agent detects and alerts but does not auto-remediate — business logic faults (deadlocks, broken consumers) cannot be safely fixed without human context. This is by design.
- **RL convergence**: Q-learning improves over episodes but would benefit from more training data or a contextual bandit approach with production reward signals for production-grade reliability.

## Live Mode (Prometheus)

Connect to a real Prometheus server instead of the simulator:

```bash
cp .env.example .env
# Set AIOPS_MODE=live, PROMETHEUS_URL, PROMETHEUS_METRIC_MAP in .env
# Edit config/metric_map.yaml to match your service names and metric queries
python -m uvicorn api.main:app --reload
```

The agent runs identically in live mode — same detection models, same reasoning loop, same API. Key differences:

- Metrics are scraped from Prometheus on each step instead of generated
- No ground-truth fault labels — evaluation switches to operational metrics (MTTR, escalation rate)
- On startup, the detector re-anchors its score normalization to 30 min of real baseline data (`OnlineCalibrator`). Actions are dry-run by default (`AIOPS_DRY_RUN=true`); set to `false` after validation to enable real `kubectl` and webhook execution.
- `GET /metrics/drift` reports feature distribution shift (PSI) vs the training baseline. `python scripts/retrain_live.py` blends real normal periods with sim data to retrain the ensemble.

See `config/metric_map.yaml` for the Prometheus query format.

## Tech Stack

Python 3.11 · OpenAI · FastAPI · scikit-learn · NumPy · Pandas · SHAP · NetworkX · PyYAML · React TypeScript · Recharts · Docker · GitHub Actions · Pytest

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                    # Add your OPENAI_API_KEY (optional)
python scripts/train_models.py          # Train detection models
python scripts/train_rl_agent.py        # Train RL agent
python scripts/run_benchmark.py --fast  # Quick benchmark
python -m uvicorn api.main:app --reload # Start API
```

## Configuration

**Sim mode (default):**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | LLM reasoning (optional — falls back to rule-based) |
| `AIOPS_MAX_ACTIONS` | `3` | Max autonomous actions per episode |
| `AIOPS_BUDGET_EXHAUSTED_ACTION` | `continue_monitoring` | Behavior when action budget runs out (`continue_monitoring` or `alert_human`) |
| `AIOPS_API_KEY` | — | When set, mutating endpoints require this key |

**Live mode (set `AIOPS_MODE=live`):**

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus server |
| `PROMETHEUS_METRIC_MAP` | `config/metric_map.yaml` | Metric name mapping config |
| `PROMETHEUS_SCRAPE_INTERVAL` | `15` | Scrape interval in seconds |
| `AIOPS_WARMUP_STEPS` | `180` | Calibration warmup steps (~30 min at 10s) |
| `AIOPS_DRY_RUN` | `true` | `false` to enable real kubectl/webhook actions |
| `KUBECTL_CONTEXT` | current | kubectl context for Kubernetes actions |
| `KUBECTL_NAMESPACE` | `default` | Kubernetes namespace |
| `BLOCK_IP_WEBHOOK_URL` | — | Webhook for IP blocking actions |
| `RATE_LIMIT_WEBHOOK_URL` | — | Webhook for rate limiting actions |

## API

**Sessionized runs** (concurrent, isolated):

- `POST /agent/runs` — create run, returns `run_id` and `incident_id`
- `POST /agent/runs/{run_id}/step` — advance run
- `GET /agent/runs/{run_id}/status`, `/log`, `/shap`, `/evaluate`, etc.

**Live mode endpoints:**

- `GET /agent/calibration-status` — warmup progress and whether score normalization has been re-anchored to real baselines
- `GET /metrics/drift` — PSI-based feature drift report vs training distribution
- `GET /metrics/topology` — returns the live Prometheus-discovered graph in live mode

Legacy endpoints (`POST /agent/scenarios/start`, `POST /agent/step`) use a default session and remain backward compatible for the dashboard.

## Documentation

- [DESIGN.md](DESIGN.md) — technical rationale and design decisions
- [EVALUATION.md](EVALUATION.md) — benchmark results and per-scenario analysis
- [DEMO_GUIDE.md](DEMO_GUIDE.md) — quick demo guide


