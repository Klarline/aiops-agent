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
├── Agent (Observe → Think → Act)
│   ├── Detection: Isolation Forest + Statistical ensemble
│   ├── Explanation: SHAP feature attribution + NL summaries
│   ├── Localization: Dependency graph traversal + lag correlation
│   ├── Diagnosis: Pattern-based fault classification
│   └── Decision: Rule-based policy + Q-learning upgrade
└── Simulated Environment
    ├── 5 microservices with dependency graph
    ├── 8 metrics per service (including TPM)
    └── 8 fault types (operational, security, business logic)
```

## AI Techniques

| Technique | Purpose | Why Not Just Use an LLM? |
|-----------|---------|--------------------------|
| Isolation Forest | Anomaly detection | Trained on actual data distribution, not prompt engineering |
| Statistical Z-score + CUSUM | Drift detection | Mathematically rigorous for gradual changes (memory leaks) |
| SHAP | Explainability | Feature-level attribution, not black-box reasoning |
| Q-learning | Adaptive remediation | Learns from outcomes, improves over time |
| NetworkX graph traversal | Root cause localization | Structural reasoning about service dependencies |

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

## Results

Run `python scripts/run_benchmark.py --seed 42` to reproduce.

See [EVALUATION.md](EVALUATION.md) for full results with per-scenario breakdown.

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

Python 3.11 · FastAPI · scikit-learn · NumPy · Pandas · SHAP · NetworkX · React TypeScript · Recharts · Docker · GitHub Actions · Pytest

## Setup

```bash
pip install -r requirements.txt
python scripts/train_models.py          # Train detection models
python scripts/train_rl_agent.py        # Train RL agent
python scripts/run_benchmark.py --fast  # Quick benchmark
python -m uvicorn api.main:app --reload # Start API
```
