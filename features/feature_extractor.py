"""Feature engineering pipeline for anomaly detection.

Extracts rolling statistics from raw service metrics to produce
feature vectors for the detection ensemble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.service_topology import METRIC_NAMES

FEATURE_SUFFIXES = ["_mean", "_std", "_roc", "_zscore"]


def get_feature_names() -> list[str]:
    """Return ordered list of feature names."""
    names = []
    for metric in METRIC_NAMES:
        for suffix in FEATURE_SUFFIXES:
            names.append(metric + suffix)
    return names


def extract_features(
    metrics_df: pd.DataFrame,
    window_size: int = 6,
) -> np.ndarray:
    """Extract feature vector from the latest time step of a service's metrics.

    Computes per-metric: rolling mean, rolling std, rate of change, z-score.
    Total features = 8 metrics * 4 features = 32.
    """
    features = []

    for metric in METRIC_NAMES:
        series = metrics_df[metric]
        roll = series.rolling(window=window_size, min_periods=1)

        r_mean = roll.mean().iloc[-1]
        r_std = roll.std().iloc[-1]
        if pd.isna(r_std):
            r_std = 0.0

        current = series.iloc[-1]
        roc = series.diff().iloc[-1] if len(series) > 1 else 0.0
        if pd.isna(roc):
            roc = 0.0

        zscore = (current - r_mean) / r_std if r_std > 1e-8 else 0.0

        features.extend([r_mean, r_std, roc, zscore])

    return np.array(features, dtype=np.float64)


def extract_features_batch(
    metrics_df: pd.DataFrame,
    window_size: int = 6,
) -> np.ndarray:
    """Extract feature vectors for every time step (used for training).

    Returns array of shape (n_timesteps - window_size, n_features).
    """
    rows = []
    for end_idx in range(window_size, len(metrics_df)):
        window = metrics_df.iloc[: end_idx + 1]
        feat = extract_features(window, window_size=window_size)
        rows.append(feat)

    if not rows:
        return np.empty((0, len(METRIC_NAMES) * len(FEATURE_SUFFIXES)))
    return np.vstack(rows)
