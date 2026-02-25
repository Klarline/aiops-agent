# Design Document

## Design Philosophy

**Grounded AI + Security Focus + AIOpsLab Alignment + Clear Communication**


## Key Technical Decisions

### 1. Two-Model Ensemble

**Why two models?** Isolation Forest captures multivariate anomalies in joint feature space. Statistical detector excels at gradual drift (CUSUM) and interpretable z-score thresholds. Together, they cover both sudden spikes and slow leaks, with model disagreement providing calibrated uncertainty.

```
combined_score = 0.6 × IF_score + 0.4 × Stat_score
uncertainty = |IF_score - Stat_score|
```

### 2. Rule-Based Policy Default

**Why not RL-first?** RL is fragile in prototype settings. If the simulator has subtle bugs, the Q-table learns garbage policies. A broken RL demo is worse than no RL at all.

Our approach: reliable rule-based expert policy as the default, with Q-learning as a documented upgrade path. If RL converges (>50% reward improvement), it becomes primary. If not, we document why — either outcome demonstrates engineering maturity.

### 3. Transaction Stall as Business Logic Fault

**Why is this our highlight?** Traditional monitoring checks infrastructure metrics (CPU, memory, latency). When a transaction processing pipeline silently fails, all dashboards show green. Only multivariate ML detection that includes business metrics (TPM) catches this. 

### 4. SHAP for Explainability 

**Why SHAP?** Anomaly scores alone are black boxes. SHAP provides feature-level attribution: "This alert was triggered because auth_error_rate increased by +0.42 and auth_latency_p99 increased by +0.31." This enables human operators to validate AI decisions.

### 5. AIOpsLab Alignment (4-Task Evaluation)

**Why mirror AIOpsLab?** The evaluation framework from Microsoft Research provides a standardized way to measure AIOps agents. By adopting their task taxonomy and leaderboard format, our results are comparable to published benchmarks and demonstrate awareness of the research frontier.

## Expected Utility Calculation

For each candidate action, we compute risk-adjusted expected utility:

```
Incident: DB latency spike (severity=0.7, confidence=0.8)

Action A — Restart DB:
  success_prob = 0.85 × 0.8 = 0.68
  EU = 0.68 × 100 - 30 = 38.0  ← SELECTED

Action B — Scale out:
  success_prob = 0.30 × 0.8 = 0.24
  EU = 0.24 × 80 - 50 = -30.8

Risk check: severity × uncertainty × blast_radius
  = 0.7 × 0.2 × 2 = 0.28 < threshold(0.5)
  → auto-execute ✓
```

## Uncertainty Gate

When ensemble models disagree (high uncertainty) or risk is too high, the agent escalates to humans rather than taking potentially harmful autonomous action. 

## Feature Engineering

Per-metric features (8 metrics × 4 features = 32 total):
- `rolling_mean_60s`: Recent average (smooths noise)
- `rolling_std_60s`: Recent volatility (detects instability)
- `rate_of_change`: First derivative (detects trends)
- `z_score`: Standard deviations from rolling mean (detects spikes)

## Architecture Constraints

- **No real database**: In-memory DataFrames (prototype scope)
- **No Kubernetes**: Simulated environment (same AI concepts, lighter infra)
- **No LLM API calls**: All ML is local `model.fit()` + `model.predict()`
- **No authentication on API**: Out of scope for prototype
