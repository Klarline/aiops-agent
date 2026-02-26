# Reviewer Quick Guide

## TL;DR

This is an autonomous AIOps agent that uses real ML models to detect, explain, localize, diagnose, and remediate cloud infrastructure issues — including security threats and silent business failures.

## Check These 3 Things

### 1. The Transaction Stall Demo
```bash
python -c "
from orchestrator.orchestrator import Orchestrator
from agent.agent import AIOpsAgent
from evaluation.benchmark_runner import train_ensemble

ensemble = train_ensemble()
orch = Orchestrator(seed=42)
orch.init_problem('transaction_stall_order')
agent = AIOpsAgent()
agent.set_ensemble(ensemble)
result = orch.run_episode(agent)
print(f'Detection: {result.detection}')
print(f'Diagnosis: {result.diagnosis}')
print(f'Agent log:')
for entry in agent.reasoning_log:
    print(f'  {entry[\"summary\"][:100]}...')
"
```
CPU and memory are normal. Latency is normal. But TPM dropped to zero. The agent catches this "silent failure."

### 2. Security Remediation
```bash
python -c "
from orchestrator.orchestrator import Orchestrator
from agent.agent import AIOpsAgent
from evaluation.benchmark_runner import train_ensemble

ensemble = train_ensemble()
orch = Orchestrator(seed=42)
orch.init_problem('brute_force_auth')
agent = AIOpsAgent()
agent.set_ensemble(ensemble)
result = orch.run_episode(agent)
print(f'Detection: {result.detection}')
print(f'Audit log: {orch.env.audit_log}')
print(f'Blocked IPs: {orch.env.blocked_ips}')
"
```
Detects brute force → blocks IP → creates audit entry.

### 3. Run the Benchmark
```bash
python scripts/run_benchmark.py --fast
```
Shows per-scenario scores in AIOpsLab leaderboard format.

## Evaluation Results
See [EVALUATION.md](EVALUATION.md)

## Design Decisions
See [DESIGN.md](DESIGN.md)

## Video Demo
See `demo_video.mp4` (timestamps in description):
- 0:30 Architecture + AIOpsLab
- 2:00 Transaction stall demo
- 3:30 Security demo
- 5:00 SHAP + operational demo
- 6:30 Results
- 8:00 Engineering practices

## Quick Reproduce
```bash
docker-compose up -d
python scripts/run_benchmark.py --seed 42
open http://localhost:3000
```
