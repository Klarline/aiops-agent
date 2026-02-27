# Adaptive AIOps Agent

An autonomous AI agent for cloud operations that detects anomalies, identifies root causes, and executes remediation — evaluated using Microsoft Research's AIOpsLab framework.

## Evaluate This Project in 5 Minutes

```bash
# Option 1: Docker (recommended)
git clone <repo-url> && cd aiops-agent
docker-compose up -d
open http://localhost:3000

# Option 2: Local
pip install -r requirements.txt
python scripts/run_benchmark.py --fast
python -m uvicorn api.main:app --reload
open http://localhost:8000/docs
```

## What This Does

This system monitors a simulated cloud application with 5 connected services, automatically detecting when something goes wrong — whether it's a server overloading, a security attack, or a silent business failure where everything *looks* healthy but transactions have stopped processing. When it detects a problem, it explains what's happening in plain English, identifies the root cause, and takes corrective action.

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
└── Simulated Environment
    ├── 5 microservices with dependency graph
    ├── 8 metrics per service (including TPM)
    └── 8 fault types (operational, security, business logic)
```

The LLM agent reasons step-by-step, calling ML modules as tools. When the LLM is unavailable or errors, the agent seamlessly falls back to a reliable rule-based pipeline.

## AI Techniques

| Technique | Why This, Not Something Else |
|-----------|------------------------------|
| **LLM ReAct Agent** | A fixed detect→localize→diagnose pipeline can't adapt investigation order. The LLM decides *what* to investigate (e.g., check topology before diagnosing during cascading failures). Falls back to rules if unavailable. |
| **Isolation Forest** | Operates in 32-dimensional feature space — catches multivariate anomalies where no single metric exceeds a threshold. Robust to dimensionality via random partitioning. |
| **Statistical Z-score + CUSUM** | IF has a blind spot for gradual drift. CUSUM accumulates small deviations over time — exactly the memory leak signature IF misses. Model disagreement = calibrated uncertainty. |
| **SHAP TreeExplainer** | Game-theoretic feature attribution with provable properties (local accuracy, consistency). Tells operators *why* an alert fired: "error_rate_zscore +4.2 contributed +0.38 to the anomaly score." |
| **Q-learning RL** | Learns remediation policy from outcomes (reward = correct action). Only replaces rule-based policy if it demonstrates >50% improvement — documented upgrade path, not premature optimization. |
| **NetworkX graph traversal** | Service dependencies form a DAG. Root cause localization uses BFS from source nodes to find the most-upstream anomalous service — structural reasoning, not statistical correlation. |

The LLM decides *what* to investigate and *when* to act. The ML models provide the actual measurements it reasons about. See [DESIGN.md](DESIGN.md) for in-depth rationale behind each choice.

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
ML Ensemble Agent         100%          97%       95%       100%   98.1%
Static Threshold           43%          38%       12%        25%   29.6%
Random Agent               62%          12%       12%        62%   37.5%
```

The **68-point gap** between the ML agent and a static threshold baseline proves the architecture works. Run `python scripts/run_benchmark.py --leaderboard` to reproduce. 6/8 scenarios score 100% across all tasks; the two weaker scenarios (anomalous access 77% diagnosis, memory leak 92% localization) are analyzed honestly in [EVALUATION.md](EVALUATION.md).

See [TUNING_LOG.md](TUNING_LOG.md) for the iteration history from v1 (F1=0.62) through v7 (98% average).

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

- **Cascading failure localization**: Lag correlation insufficient when propagation delays vary significantly
- **Anomalous access**: Higher false positive rate due to subtle signal overlap
- **Transaction stall mitigation**: Agent correctly detects but can only alert (cannot auto-fix business logic)
- **RL convergence**: Q-learning shows learning but may need more episodes for production reliability

## Tech Stack

Python 3.11 · OpenAI · FastAPI · scikit-learn · NumPy · Pandas · SHAP · NetworkX · React TypeScript · Recharts · Docker · GitHub Actions · Pytest

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                    # Add your OPENAI_API_KEY (optional)
python scripts/train_models.py          # Train detection models
python scripts/train_rl_agent.py        # Train RL agent
python scripts/run_benchmark.py --fast  # Quick benchmark
python -m uvicorn api.main:app --reload # Start API
```

## Agent Guardrail Configuration

Optional environment variables:

- `AIOPS_MAX_ACTIONS` (default: `3`): maximum autonomous actions per episode
- `AIOPS_BUDGET_EXHAUSTED_ACTION` (default: `continue_monitoring`): behavior after budget is exhausted while anomaly persists
  - `continue_monitoring`: stop autonomous actions and keep monitoring
  - `alert_human`: immediately escalate to a human operator
- `AIOPS_API_KEY` (optional): when set, mutating endpoints require `X-API-Key` or `Authorization: Bearer`. When unset, all requests are allowed (local/demo).

These settings are useful for tuning safety posture during evaluation and demos.

## API: Sessionized Runs

For API-driven clients, use sessionized endpoints for concurrent independent runs:

- `POST /agent/runs` — create run, returns `run_id` and `incident_id`
- `POST /agent/runs/{run_id}/step` — advance run
- `GET /agent/runs/{run_id}/status`, `/log`, `/shap`, `/evaluate`, etc.

Legacy endpoints (`POST /agent/scenarios/start`, `POST /agent/step`) use a default session and remain backward compatible for the dashboard.
