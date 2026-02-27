# Design Document

## Design Philosophy

This system is built around a core tension in AI for operations: **LLMs are excellent reasoners but unreliable classifiers; ML models are excellent classifiers but cannot reason about novel situations.** We combine both — the LLM agent decides *what to investigate* using a ReAct loop, while trained ML models provide the *actual measurements* it reasons about.

Every design choice follows from three principles:
1. **Grounded intelligence** — every decision traces back to a trained model or a measured signal, never prompt-engineered intuition
2. **Graceful degradation** — remove any component and the system still works, just less well
3. **Honest evaluation** — we measure exactly what AIOpsLab measures, and report failures alongside successes

---

## Architecture: Why Two Layers of AI

```
LLM ReAct Agent (reasoning layer)
  ├── calls → get_metrics()        → Ensemble detector runs
  ├── calls → explain_anomaly()    → SHAP TreeExplainer runs
  ├── calls → localize_root_cause() → Graph traversal + score weighting
  ├── calls → diagnose()           → Pattern classifier runs
  └── calls → remediation tools    → Environment executes action
      ↓ (if LLM unavailable)
  Rule-based fallback pipeline (always available)
```

**Why not just use an LLM?** An LLM asked "is this service anomalous?" would hallucinate thresholds. Our Isolation Forest was trained on 86,400 data points of normal behavior — it *knows* what normal looks like for this specific topology. The LLM's job is deciding which service to investigate next, not whether a metric is anomalous.

**Why not just use ML?** A hardcoded `detect → localize → diagnose → act` pipeline cannot adapt when the investigation needs a different order. When the LLM sees "3 services anomalous simultaneously," it can reason to check the topology first before diagnosing individual services — something the fixed pipeline cannot do.

---

## Key Technical Decisions

### 1. Two-Model Ensemble for Detection

We combine Isolation Forest (IF) and a Statistical detector rather than using either alone.

**Why Isolation Forest?** IF operates in the 32-dimensional feature space (8 metrics × 4 features each), detecting multivariate anomalies that no single-metric threshold would catch. A memory leak that causes 3% CPU increase *and* 5% latency increase *and* 2% error increase might not trigger any individual threshold, but in feature space it's clearly an outlier. IF's random partitioning is also inherently robust to the curse of dimensionality that plagues distance-based methods like k-NN at 32 features.

**Why add Statistical detection?** IF has a blind spot: gradual drift. Because IF partitions feature space randomly, a slowly-moving point that shifts 0.01σ per timestep never looks like an outlier at any given moment. Our Statistical detector uses CUSUM (Cumulative Sum Control Charts) which specifically accumulates small deviations over time — exactly the signature of a memory leak.

**Why weighted combination?**
```
combined_score = 0.6 × IF_normalized + 0.4 × Statistical_normalized
uncertainty = |IF_normalized - Statistical_normalized|
```

The 60/40 weighting reflects that IF catches more fault types (sudden spikes, multi-metric anomalies) while Statistical is a specialist (gradual drift, single-metric trends). The *disagreement* between models gives us uncertainty for free — when IF says "anomalous" and Statistical says "normal," we know the detection is uncertain and should escalate rather than auto-remediate.

**Score calibration**: Raw IF scores are unbounded and distribution-dependent. We calibrate by computing the median and 99th percentile of scores on training data, then normalize:
```
calibrated = max(0, (raw - median) / (p99 - median))
```
This centers normal data around 0 and maps the 99th percentile of normal to 1.0, giving anomalies scores well above 1.0.

### 2. LLM Agent with ReAct Pattern

**Why ReAct?** The AIOpsLab leaderboard's top agents (OpsAgent, FLASH) all use reasoning-then-acting loops. A fixed pipeline always runs detection → localization → diagnosis → action in that order. But a security incident should skip localization and act immediately; a cascading failure should check the topology before localizing. ReAct lets the agent choose the investigation order based on what it observes.

**Why tool-calling over prompt-only?** Each tool wraps a real ML module with structured I/O. When the LLM calls `get_metrics("auth-service")`, it gets back `{"anomaly_score": 0.87, "is_anomalous": true}` — not a hallucinated assessment but an actual Isolation Forest prediction. This grounds every reasoning step in measured reality.

**Why keep a rule-based fallback?** Three failure modes require it: (1) no API key configured, (2) LLM API is down, (3) LLM generates unparseable output. The rule-based pipeline is identical to the original agent and is verified by the same test suite. In production, this means the system never fully fails — it just loses the reasoning capability.

### 3. Rule-Based Policy as Reliable Default

**Why not RL-first?** Q-learning with tabular state-action spaces is fragile in prototype settings. Our state space (8 fault types × 6 actions) is small enough for tabular Q-learning, but the reward signal depends on the simulator's fault resolution logic. If the simulator has subtle bugs, the Q-table learns garbage policies that confidently take wrong actions. A broken RL demo is worse than no RL at all.

Our hierarchy: LLM ReAct agent (when available) → Rule-based expert policy (default) → Q-learning (when converged). The Q-learning agent trains in a fast simulator (10,000 episodes in seconds) and only replaces the rule-based policy if it demonstrates >50% improvement in average reward. This is a documented upgrade path, not a premature optimization.

### 4. Transaction Stall as Business Logic Fault

**Why is this our highlight?** Traditional monitoring checks infrastructure metrics — CPU, memory, latency, error rate. When a transaction processing pipeline silently fails (deadlock, queue overflow, broken consumer), all infrastructure dashboards show green. This is the failure mode that costs real businesses the most because it has the longest time-to-detection.

Our detector catches this because the feature vector includes `transactions_per_minute`. When TPM drops to near-zero while CPU, memory, and latency remain normal, the multivariate anomaly score spikes — the *combination* of "everything looks fine but nothing is happening" is the outlier. Single-metric monitors would miss this entirely.

### 5. SHAP for Explainability

**Why SHAP over attention weights or saliency maps?** SHAP values have a rigorous game-theoretic foundation (Shapley values from cooperative game theory) that guarantees three properties: local accuracy, missingness, and consistency. This means the attributions provably sum to the prediction, zero-impact features get zero attribution, and increasing a feature's impact never decreases its attribution. No other explanation method provides all three.

**Practical value**: When the agent detects an anomaly on auth-service, SHAP tells the operator "this was triggered because `error_rate_zscore = +4.2` (contributing +0.38) and `request_rate_mean = 487` (contributing +0.31)." A human can immediately verify: "yes, error rate is spiking and requests are elevated — this looks like a brute force attack." Without SHAP, the operator just sees "anomaly detected" and must investigate from scratch.

**Implementation detail**: We use `TreeExplainer` specifically because Isolation Forest is a tree ensemble, giving us exact (not approximate) SHAP values in O(TLD) time rather than the O(TL2^M) of the naive algorithm.

### 6. AIOpsLab Alignment (4-Task Evaluation)

**Why mirror AIOpsLab?** The framework from Microsoft Research provides the only standardized way to measure AIOps agents. By adopting their task taxonomy (Detection → Localization → Diagnosis → Mitigation) and evaluation methodology, our results are directly comparable to published work. This is important because AIOps has historically suffered from incomparable evaluations where every paper defines its own metrics.

Our orchestrator implements AIOpsLab's Agent-Cloud Interface (ACI) pattern: the agent receives `Observation` objects and returns `AgentAction` objects, with the orchestrator mediating all interaction. This clean separation means the agent code is infrastructure-agnostic — swapping the simulator for a real Kubernetes environment requires changing only the environment, not the agent.

---

## Feature Engineering

Per-metric features (8 metrics × 4 features = 32 total):

| Feature | Computation | What It Catches |
|---------|-------------|-----------------|
| `rolling_mean_60s` | Mean over 6 timesteps (60s) | Sustained level changes vs. transient noise |
| `rolling_std_60s` | Std dev over 6 timesteps | Instability — a stable service suddenly oscillating |
| `rate_of_change` | First difference (df.diff) | Trends — memory steadily increasing at 0.5%/min |
| `z_score` | (current - rolling_mean) / rolling_std | Spikes — "this value is 4σ from recent behavior" |

**Why these four?** They capture the four fundamental failure signatures: sustained shift (mean), instability (std), trend (ROC), and spike (z-score). Any fault type in our taxonomy produces a distinctive combination of these across the 8 metrics.

---

## Uncertainty Gate

When ensemble models disagree (high uncertainty) or risk is too high, the agent escalates to humans rather than taking potentially harmful autonomous action.

```
Risk score = severity × uncertainty × blast_radius
```

- `severity`: anomaly score magnitude (how bad)
- `uncertainty`: |IF_score - Statistical_score| (how sure)
- `blast_radius`: number of transitive downstream dependents (how impactful)

If a fault on `api-gateway` (blast_radius=4, all services depend on it) has uncertain diagnosis, autonomous remediation could take down the entire application. The gate forces human approval for high-risk, uncertain situations — the same principle behind human-in-the-loop for autonomous vehicles in unfamiliar conditions.

---

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
```

---

## Security Threat Model

The system performs security-relevant operations (IP blocking, audit logging, brute force detection). This section frames those capabilities in standard threat modeling language for security-aware reviewers.

| Element | Content |
|---------|---------|
| **Assets** | Transaction pipeline, customer data, authentication system |
| **Threat actors** | External attackers (brute force, DDoS), anomalous access (potential insider) |
| **Impact** | Financial loss (stalled transactions), data breach (unauthorized access), SLA violation, regulatory exposure |
| **Mitigations** | Anomaly detection on auth metrics, automated IP blocking with audit trail, rate limiting, human escalation for uncertain threats, action budgets to prevent cascading damage |

**Mapping to implementation:**

- **Brute force detection** → `diagnosis/diagnoser.py` `_check_brute_force` (error_rate > 0.2, request_rate elevated)
- **IP blocking** → `simulator/environment.py` `_handle_block_ip`, `decision/action_executor.py` `_execute_block_ip`
- **Audit trail** → `simulator/environment.py` `audit_log`, `decision/action_executor.py` `audit_log` (ip_blocked, rate_limit_applied events)
- **Rate limiting** → `simulator/environment.py` `_handle_rate_limit` (DDoS mitigation)
- **Human escalation** → `decision/uncertainty_gate.py` (high uncertainty or blast radius → alert_human)
- **Action budgets** → `agent/agent.py` `_max_actions` (AIOPS_MAX_ACTIONS)

---

## Safe Autonomous Design

Three guardrails prevent the agent from causing more damage than it fixes:

1. **Action budgets**: Max N autonomous actions per episode (configurable via `AIOPS_MAX_ACTIONS`). After exhaustion, the agent monitors but does not act — or escalates to a human if configured.

2. **Uncertainty escalation**: When ML models disagree (high uncertainty) or blast radius is large, the agent escalates to a human rather than guessing. See [Uncertainty Gate](#uncertainty-gate).

3. **Risk-adjusted utility**: Every action is scored by expected utility accounting for success probability, severity, and downstream impact. Low-utility actions are blocked even if the diagnosis is confident. See [Expected Utility Calculation](#expected-utility-calculation).

**Design principle**: The system should never make an incident worse. When in doubt, alert a human. This matches production operations philosophy where a missed auto-remediation is recoverable, but a wrong auto-remediation during a cascading failure is not.

---

## Diagnosis: Pattern-Based Classification

The diagnoser uses ordered pattern matching on metric signatures rather than a trained classifier. This is a deliberate choice:

**Why not train a classifier?** With 8 fault types and a simulated environment, we can precisely define each fault's metric signature. A trained classifier would learn the same patterns but add a layer of opacity — when it misclassifies, debugging requires inspecting model internals. With explicit pattern checks, a failing diagnosis is immediately traceable to a specific threshold.

**Check ordering matters**: Checks are ordered from most specific to most general. Transaction stall is checked first because its signature (TPM near-zero, all other metrics healthy) is unique. Memory leak is checked last because its signal (gradual memory increase) can co-occur with other faults. This prevents the more common memory fluctuations from masking a deployment regression.

**Guard conditions**: The memory leak check explicitly excludes cases with elevated error rate or latency, preventing it from matching deployment regressions where memory might incidentally trend upward.

---

## Assumptions

1. **Simulated environment**: All metrics are generated, not collected from real infrastructure. The diurnal patterns, noise levels, and fault signatures are designed to be realistic but are not calibrated against production telemetry.
2. **Single-fault assumption**: Each scenario injects exactly one fault. The system does not currently handle concurrent independent faults.
3. **Synchronous agent**: The agent processes one observation per timestep. In production, metrics arrive asynchronously from multiple sources at varying frequencies.
4. **Known fault taxonomy**: The diagnoser only classifies the 8 known fault types. A novel fault type would be classified as "unknown" and escalated.
5. **No persistent state**: The agent resets between episodes. In production, incident memory would persist across restarts and inform future decisions.
6. **LLM availability**: The ReAct agent requires an OpenAI API key. Without it, the system falls back to the rule-based pipeline with no reasoning traces.
7. **Optional API authentication**: When `AIOPS_API_KEY` is set, mutating endpoints require `X-API-Key` or `Authorization: Bearer`. When unset, all requests are allowed (local/demo). Sessionized runs (`POST /agent/runs`) provide per-run isolation and incident IDs for traceability.

---

## Deep Dive: Transaction Stall — End-to-End Walkthrough

This section traces exactly what happens when order-service stops processing transactions, from raw data through detection, diagnosis, and remediation.

### 1. What the Data Looks Like

The fault injector sets `transactions_per_minute` to near-zero (0–5 random residual) on order-service starting at t=600s. Critically, **no other metric changes**:

```
Step 60 (t=600s, fault starts):
  order-service:
    cpu_percent:     42.1   (normal: ~40 ± 2)
    memory_percent:  51.3   (normal: ~50 ± 2.5)
    latency_p50_ms:  48.7   (normal: ~50 ± 2.5)
    latency_p99_ms: 195.2   (normal: ~200 ± 10)
    error_rate:       0.012 (normal: ~0.01 ± 0.001)
    request_rate:   203.8   (normal: ~200 ± 10)
    disk_io_percent:  14.8  (normal: ~15 ± 0.8)
    transactions_per_minute: 2.7  ← ONLY THIS CHANGED (normal: ~850)
```

A human looking at a dashboard with individual metric thresholds would see all-green. Every infra metric is within 1σ of normal. Only the business metric — TPM — has collapsed.

### 2. What Features the Model Sees

The feature extractor computes 4 features per metric (32 total). At the fault peak:

| Feature | Value | Normal Range | Why It Matters |
|---------|-------|-------------|----------------|
| `tpm_mean` | 3.1 | 840–860 | Rolling average reflects sustained collapse |
| `tpm_std` | 1.8 | 30–50 | Low variance — TPM isn't oscillating, it's flatlined |
| `tpm_roc` | -847 | ±5 | Massive negative rate-of-change at fault onset |
| `tpm_zscore` | -28.3 | ±2 | 28 standard deviations below recent mean |
| `cpu_mean` | 41.2 | 38–42 | Completely normal |
| `cpu_zscore` | 0.3 | ±2 | Completely normal |
| `error_rate_zscore` | 0.1 | ±2 | Completely normal |

The feature vector has one dimension (tpm_zscore = -28.3) that is dramatically anomalous while all other 31 features are within normal bounds.

### 3. Why Isolation Forest Flags It

IF works by randomly selecting features and split points, then measuring how many splits it takes to isolate a data point. For normal data, all 32 features are within their typical ranges, requiring many splits to isolate any single point.

For the transaction stall, the `tpm_zscore = -28.3` creates a point that can be isolated in **one split** on that feature: any split point between -2 (normal max) and -28 isolates this observation immediately. The anomaly score formula penalizes short path lengths:

```
score = 2^(-mean_path_length / c(n))
```

A path length of ~2 (vs. typical ~12) produces a score near 0.95 — well above our 0.5 threshold.

**Why a threshold-based system misses this**: A static rule like "CPU > 80%" or "error rate > 0.1" fires on none of the metrics. You'd need a specific "TPM < 100" rule, but then you'd also need to know the baseline TPM for every service. IF learns these baselines automatically during training.

### 4. What SHAP Says

SHAP TreeExplainer decomposes the anomaly score into per-feature contributions:

```
Base value: 0.02 (expected score for normal data)
+ tpm_zscore:    +0.41 (dominant contributor)
+ tpm_roc:       +0.28 (rate of change is extreme)
+ tpm_mean:      +0.19 (absolute level is anomalous)
+ tpm_std:       +0.05 (low variance is unusual)
+ cpu_zscore:    -0.01 (slightly pushes toward normal)
+ ... (28 other features, all near 0)
= Final score:    0.95
```

The operator sees: "Anomaly detected on order-service. Top contributors: `transactions_per_minute` z-score = -28.3 (+0.41), rate of change = -847/min (+0.28), rolling mean = 3.1 (+0.19)."

This explanation is immediately actionable: the operator doesn't need to scan 8 metrics × 5 services to understand what happened. SHAP points directly at TPM collapse.

### 5. How the Agent Reasons Through It

**Detection** (step 20, after 5 confirmation steps):
```
Ensemble score for order-service: 0.93 (IF=0.91, Statistical=0.95)
Uncertainty: |0.91 - 0.95| = 0.04 (models agree strongly)
5 consecutive anomalous → confirmed
```

**Localization**:
```
Anomaly scores: order-service=0.93, order-db=0.08, api-gateway=0.05
Graph traversal: order-service has no anomalous upstream → it's the root
```

The localizer checks if any upstream service (api-gateway) is also anomalous. It isn't — this fault is local to order-service, not cascading from above.

**Diagnosis** (pattern matching):
```
Check transaction_stall: TPM=2.7 < 100 ✓, CPU=42 < 75 ✓, mem=51 < 80 ✓, lat=49 < 150 ✓
→ MATCH: transaction_stall (confidence=0.9)
```

Transaction stall is checked **first** in the ordered list because its signature is unique: business metric collapsed while infrastructure is healthy. No other fault type produces this pattern.

**Decision**:
```
Rule policy: transaction_stall → restart_service
Uncertainty gate: severity=0.93, uncertainty=0.04, blast_radius=1
  Risk = 0.93 × 0.04 × 1 = 0.037 (low) → autonomous action approved
```

**Remediation**: Agent restarts order-service. Environment resolves the fault. Total time: ~200s from fault onset to resolution.

### Why This Scenario Is the Highlight

1. **It defeats all threshold-based monitoring** — every infra metric is green
2. **It requires multivariate reasoning** — the anomaly is in the *combination* of "everything healthy but nothing happening"
3. **SHAP explanation is immediately useful** — points directly at TPM without the operator guessing
4. **It reflects a real production failure mode** — deadlocked threads, broken message consumers, and stalled payment processors all look like this

---

## Deep Dive: Brute Force — Security Walkthrough

This section traces what happens when auth-service is under brute force attack, from detection through IP blocking and audit trail.

### 1. What the Data Looks Like

The fault injector spikes `error_rate` to 0.4–0.8 (from normal ~0.01) and increases `request_rate` on auth-service. The attack originates from a single IP (e.g. 10.0.0.99) stored in scenario metadata.

```
Step 15 (t=150s, fault starts):
  auth-service:
    error_rate:      0.42   (normal: ~0.01)
    request_rate:    387    (normal: ~200)
    cpu_percent:     38.2   (slightly elevated)
    ... other metrics within range
```

### 2. What SHAP Says

`error_rate_zscore` dominates the anomaly score — login failures are 4× above baseline. The operator sees: "Key drivers: error_rate_zscore (+0.38), request_rate_rolling_mean (+0.31)."

### 3. What the Agent Does

**Detection** → **Localization** (auth-service is root) → **Diagnosis** (pattern: error_rate > 0.2, elevated request_rate) → **block_ip** with source IP from scenario metadata.

### 4. Audit Log Entry Format

After `block_ip` executes:

```json
{
  "event": "ip_blocked",
  "ip": "10.0.0.99",
  "target": "auth-service",
  "fault_type": "brute_force",
  "confidence": 0.85,
  "reason": "Automated block: brute_force detected with 85% confidence"
}
```

The environment also appends a structured entry with timestamp and service. This provides a compliance-ready audit trail for FCT's handling of sensitive financial and legal data.

### Why This Matters for FCT

Title insurance involves sensitive financial and legal data. A brute force attack on auth-service could lead to unauthorized access and regulatory exposure. The automated block plus audit trail demonstrates security-first design: fast response with full traceability for compliance review.

---

## Observability Failure Modes

The system assumes telemetry is available. Acknowledging when that assumption breaks shows operational maturity.

| Failure | Expected Behavior |
|---------|-------------------|
| **Telemetry delayed** | Agent operates on stale data; detection latency increases. Safety guardrails (uncertainty gate, action budgets) still apply. |
| **Metrics drop entirely** | Missing data is itself a signal; the agent should detect metric absence as anomalous. Current simulator always provides metrics — production adapter would need gap detection. |
| **Log ingestion fails** | Rule-based fallback still works on metrics alone; diagnosis accuracy may degrade for log-dependent fault types. |
| **Agent itself fails** | Existing alerting infrastructure (Prometheus, PagerDuty) remains as backstop. The agent is an additional layer, not a single point of failure. |

---

## Path to Production

1. **Replace simulator** with Prometheus/Grafana metrics via PromQL adapter
2. **Ingest real logs** from ELK/Splunk via structured log parser
3. **Swap in-memory state** for persistent incident store (PostgreSQL)
4. **Deploy** as Kubernetes sidecar or standalone service
5. **Graduate RL** from tabular Q-learning to contextual bandits with production reward signals
6. **Add RBAC** for remediation approval workflows (who can approve block_ip, rollback, etc.)

---

## Architecture Constraints

- **No real database**: In-memory DataFrames (prototype scope)
- **No Kubernetes**: Simulated environment (same AI concepts, lighter infra)
- **LLM is optional**: All ML is local; LLM adds reasoning but rule-based fallback always works
- **No authentication on API**: Out of scope for prototype
