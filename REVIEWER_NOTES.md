# Reviewer Quick Guide

## What Makes This Different

- **Real ML models** trained with `model.fit()`, not LLM API wrappers
- **Catches silent business failures** that threshold monitoring misses (transaction stall)
- **Security scenarios** with IP blocking and audit trails (brute force, DDoS)
- **Every decision explained** via SHAP feature attribution
- **Honest evaluation** with baselines, limitations, and reproducible benchmarks

## Check These 3 Things

### 1. Transaction Stall (Silent Failure)

CPU and memory normal. Latency normal. But TPM dropped to zero. Threshold monitoring misses it.

```bash
python scripts/demo_transaction_stall.py
```

**Expected output (seed=42):**

```
Transaction Stall Demo — Silent failure where every dashboard says healthy

Scenario: order-service stops processing transactions. CPU, memory, latency all normal.
Only TPM (transactions per minute) collapses. Threshold monitoring misses it.

Running... (seed=42)

Result:
  Detection:   True
  Localization: True
  Diagnosis:   True
  Mitigation:  True
  Score:       100%

Agent reasoning (last entry):
  ⚠ Silent failure detected: order-service infrastructure appears healthy (CPU: 40%, Memory: 50%)...

✓ Agent caught the silent failure.
```

---

### 2. Security (Brute Force → IP Block → Audit)

Detects brute force on auth-service, blocks source IP, creates audit entry.

```bash
python scripts/demo_security.py
```

**Expected output (seed=42):**

```
Security Demo — Brute force → IP block → audit trail

Scenario: Brute force login attack on auth-service from 10.0.0.99
Security path: detect → block IP → create audit log entry.

Executing block_ip... (seed=42)

Result:
  Action:      block_ip
  Success:     True

Blocked IPs:
  10.0.0.99

Audit log:
  [ip_blocked] IP 10.0.0.99 blocked on auth-service
    Reason: Automated block: brute_force detected with 85% confidence...

✓ Security threat remediated with audit trail.
```

---

### 3. Benchmark

```bash
python scripts/run_benchmark.py --fast
```

Shows per-scenario scores in AIOpsLab leaderboard format.

## Go Deeper

- [EVALUATION.md](EVALUATION.md) — per-scenario analysis, robustness evidence
- [DESIGN.md](DESIGN.md) — technical rationale, security threat model

## Video Demo

See `demo_video.mp4`. Timestamps: 0:30 Architecture · 2:00 Transaction stall · 3:30 Security · 5:00 SHAP · 6:30 Results · 8:00 Engineering practices

## Quick Setup

```bash
docker-compose up -d --build   # --build ensures latest code; omit if image is fresh
python scripts/run_benchmark.py --seed 42
open http://localhost:3000
```

If the dashboard shows stale data (e.g. CPU/Memory 0% in transaction stall), rebuild: `docker-compose build --no-cache && docker-compose up -d`
