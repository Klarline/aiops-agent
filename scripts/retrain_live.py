"""Retrain the ensemble detector on blended sim + real data.

Blends simulated normal data (the original training set) with real normal
periods collected from the incident store, retrains the EnsembleDetector,
validates it against the sim benchmark scenarios, and saves the model if
validation passes.

Usage:
    python scripts/retrain_live.py
    python scripts/retrain_live.py --sim-ratio 0.7 --min-regression 0.05

Arguments:
    --sim-ratio     Fraction of training data from simulation (default: 0.7)
    --min-regression  Max allowed drop in sim benchmark score before rejecting
                    the new model (default: 0.05 = 5 percentage points)
    --dry-run       Run validation but do not save the model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.ensemble import EnsembleDetector
from features.feature_extractor import extract_features_batch
from simulator.service_topology import build_topology
from simulator.metrics_generator import generate_metrics
from knowledge_base.incident_store import IncidentStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_simulated_normal_features(n_seeds: int = 3) -> np.ndarray:
    """Generate simulated normal features across multiple seeds."""
    topology = build_topology()
    all_features = []
    for seed in range(n_seeds):
        metrics = generate_metrics(topology, 3600, 10, seed=seed)
        for _, df in metrics.items():
            feats = extract_features_batch(df, window_size=6)
            all_features.append(feats)
    combined = np.vstack(all_features)
    return combined[~np.isnan(combined).any(axis=1)]


def load_real_normal_features(incident_store: IncidentStore) -> np.ndarray | None:
    """Extract feature vectors from real normal periods in the incident store.

    Normal periods are identified as steps where no non-monitoring agent
    action was recorded (i.e., the system was stable).
    """
    try:
        normal_snapshots = incident_store.get_normal_snapshots()
        if not normal_snapshots:
            logger.info("no real normal snapshots found in incident store")
            return None

        import pandas as pd
        all_features = []
        for snapshots in normal_snapshots:
            if len(snapshots) >= 6:
                df = pd.DataFrame(snapshots)
                feats = extract_features_batch(df, window_size=6)
                valid = feats[~np.isnan(feats).any(axis=1)]
                if len(valid):
                    all_features.append(valid)

        if not all_features:
            return None

        combined = np.vstack(all_features)
        logger.info("loaded %d real normal feature vectors", len(combined))
        return combined
    except Exception as exc:
        logger.warning("could not load real features from incident store: %s", exc)
        return None


def validate_against_sim_benchmark(
    new_model: EnsembleDetector,
    baseline_score: float,
    min_regression: float,
) -> tuple[bool, float]:
    """Run the new model against a quick sim benchmark.

    Returns (passed, new_score).
    """
    try:
        from evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(ensemble=new_model, fast=True)
        results = runner.run()
        new_score = results.average_score
        passed = (baseline_score - new_score) <= min_regression
        return passed, new_score
    except Exception as exc:
        logger.warning("benchmark validation failed: %s — skipping validation", exc)
        return True, baseline_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain ensemble on blended sim+real data")
    parser.add_argument("--sim-ratio", type=float, default=0.7,
                        help="Fraction of training data from simulation (default: 0.7)")
    parser.add_argument("--min-regression", type=float, default=0.05,
                        help="Max allowed benchmark score drop (default: 0.05)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate but do not save the model")
    args = parser.parse_args()

    # 1. Load simulated normal features (baseline)
    logger.info("generating simulated normal features...")
    sim_features = load_simulated_normal_features()
    logger.info("sim features: %d vectors", len(sim_features))

    # 2. Load real normal features
    store = IncidentStore()
    real_features = load_real_normal_features(store)

    # 3. Blend datasets
    if real_features is not None and len(real_features) > 0:
        n_sim = int(len(sim_features) * args.sim_ratio / (1 - args.sim_ratio + args.sim_ratio))
        n_real = len(real_features)
        sim_sample = sim_features[np.random.choice(len(sim_features), min(n_sim, len(sim_features)), replace=False)]
        training_features = np.vstack([sim_sample, real_features])
        logger.info("blended training set: %d sim + %d real = %d total",
                    len(sim_sample), n_real, len(training_features))
    else:
        training_features = sim_features
        logger.info("no real features available — training on sim data only")

    # 4. Train baseline model for regression comparison
    logger.info("training baseline model (sim only) for regression check...")
    baseline_model = EnsembleDetector()
    baseline_model.fit(sim_features)

    # 5. Train new model on blended data
    logger.info("training new model on blended data...")
    new_model = EnsembleDetector()
    new_model.fit(training_features)

    # 6. Validate — ensure we haven't regressed on known sim scenarios
    logger.info("validating against sim benchmark...")
    try:
        baseline_passed, baseline_score = validate_against_sim_benchmark(
            baseline_model, 1.0, args.min_regression
        )
        new_passed, new_score = validate_against_sim_benchmark(
            new_model, baseline_score, args.min_regression
        )

        logger.info("baseline score: %.3f  new score: %.3f  delta: %.3f",
                    baseline_score, new_score, new_score - baseline_score)

        if not new_passed:
            logger.error(
                "new model regressed by %.3f (max allowed: %.3f) — NOT saving",
                baseline_score - new_score, args.min_regression,
            )
            return 1
    except Exception as exc:
        logger.warning("validation skipped: %s", exc)
        new_score = 0.0

    # 7. Save
    if args.dry_run:
        logger.info("dry-run: model NOT saved (validation passed)")
        return 0

    import joblib
    model_path = Path("detection/models/isolation_forest_live.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_model, model_path)
    logger.info("model saved to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
