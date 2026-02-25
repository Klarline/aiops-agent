# Evaluation Results

## Methodology

Evaluated using the AIOpsLab 4-task taxonomy:
- **Detection**: Did the agent detect the anomaly?
- **Localization**: Did it identify the correct root cause service?
- **Diagnosis**: Did it classify the correct fault type?
- **Mitigation**: Did the remediation action resolve the issue?

Benchmark: 8 scenarios × 13 episodes each = 104 total episodes (seed=42 for reproducibility).

## How to Reproduce

```bash
python scripts/run_benchmark.py --seed 42 --episodes 13
# Results saved to: evaluation/results/benchmark_results.json
# Verified against: evaluation/results/expected_results.json
```

## Agent Configuration

| Component | Implementation |
|-----------|---------------|
| Detection | Isolation Forest (60%) + Statistical Z-score/CUSUM (40%) |
| Localization | Dependency graph BFS + anomaly score ranking |
| Diagnosis | Pattern matching on metric signatures |
| Policy | Rule-based expert policy (default) + Q-learning (upgrade) |
| Explainability | SHAP TreeExplainer + template-based NL summaries |

## Known Limitations

1. **Cascading failure localization**: Lag correlation is insufficient when propagation delays vary significantly between services. A more sophisticated approach would use distributed tracing data.

2. **Anomalous access false positives**: Normal and malicious access patterns overlap in feature space. Production systems would incorporate endpoint-level behavioral baselines.

3. **Transaction stall mitigation**: The agent correctly detects and diagnoses but can only alert humans — cannot automatically fix business logic issues. This is an inherent limitation, not a bug.

4. **RL agent over-treatment**: In early training episodes, the Q-learning agent tends to over-treat low-severity incidents. More training episodes and reward shaping would address this.

5. **Single-node simulation**: All metrics are generated synthetically without real distributed system effects like network partitions or clock skew.

## What I Would Improve With More Time

- Distributed tracing for better cascading failure localization
- Endpoint-level behavioral baselines for anomalous access
- Longer RL training with reward shaping
- Real-time dashboard with WebSocket streaming
- Historical incident memory for pattern matching
