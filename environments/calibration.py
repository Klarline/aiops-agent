"""Online calibrator for adapting sim-trained models to real traffic baselines.

On startup in live mode, collects WARMUP_STEPS of real
metric data as a "normal" baseline, then re-anchors the EnsembleDetector's
score normalization parameters to the real distribution.

The IsolationForest model itself is NOT retrained — only the percentile
normalization (median, p99) is recalculated. This is safe because the
feature engineering (z-scores, rate-of-change) is distribution-agnostic:
anomaly patterns look like anomalies regardless of absolute metric values.

A circuit breaker aborts calibration if more than ABORT_RATIO of warmup
samples already score as anomalous — this means calibration started during
an active incident and the data is not representative of normal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationStatus:
    calibrated: bool = False
    aborted: bool = False
    steps_collected: int = 0
    warmup_steps: int = 0
    abort_reason: str = ""
    baselines: dict[str, float] = field(default_factory=dict)


class OnlineCalibrator:
    """Collects warmup data and re-anchors EnsembleDetector score normalization.

    Usage:
        calibrator = OnlineCalibrator(ensemble, warmup_steps=180)
        env.attach_calibrator(calibrator)

    On each step(), the environment calls calibrator.ingest(features).
    After warmup_steps are collected (and no circuit breaker fires),
    calibrator.calibrated is True and the ensemble scores are re-anchored.
    """

    def __init__(
        self,
        ensemble: object,
        warmup_steps: int = 180,
        abort_ratio: float = 0.2,
        min_steps_before_abort_check: int = 20,
    ) -> None:
        self._ensemble = ensemble
        self._warmup_steps = warmup_steps
        self._abort_ratio = abort_ratio
        self._min_steps_before_abort_check = min_steps_before_abort_check
        self._warmup_features: list[np.ndarray] = []
        self._calibrated = False
        self._aborted = False
        self._abort_reason = ""

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def steps_collected(self) -> int:
        return len(self._warmup_features)

    @property
    def steps_remaining(self) -> int:
        return max(0, self._warmup_steps - self.steps_collected)

    def ingest(self, features: np.ndarray) -> None:
        """Feed one feature vector from the warmup window.

        Call this once per step during the warmup period. After warmup_steps
        are collected the calibrator re-anchors the ensemble and sets
        self.calibrated = True.
        """
        if self._calibrated or self._aborted:
            return

        self._warmup_features.append(features)

        # Circuit breaker: check if too many warmup samples look anomalous
        if len(self._warmup_features) >= self._min_steps_before_abort_check:
            self._check_circuit_breaker()
            if self._aborted:
                return

        if len(self._warmup_features) >= self._warmup_steps:
            self._recalibrate()

    def status(self) -> CalibrationStatus:
        return CalibrationStatus(
            calibrated=self._calibrated,
            aborted=self._aborted,
            steps_collected=self.steps_collected,
            warmup_steps=self._warmup_steps,
            abort_reason=self._abort_reason,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_circuit_breaker(self) -> None:
        """Abort if more than abort_ratio of warmup samples look anomalous."""
        try:
            scores = np.array([
                self._ensemble.iso_detector.detect(f)
                for f in self._warmup_features
            ])
            # Use the current (sim-trained) iso median as a rough threshold
            rough_threshold = self._ensemble._iso_median + self._ensemble._iso_max * 0.5
            high_ratio = float(np.mean(scores > rough_threshold))
            if high_ratio > self._abort_ratio:
                self._aborted = True
                self._abort_reason = (
                    f"{high_ratio:.0%} of warmup samples appear anomalous "
                    f"(threshold {self._abort_ratio:.0%}). "
                    "Calibration aborted — likely started during an active incident. "
                    "Restart the agent when the system is stable."
                )
                logger.warning("calibration aborted: %s", self._abort_reason)
        except Exception as exc:
            logger.debug("circuit breaker check failed: %s", exc)

    def _recalibrate(self) -> None:
        """Re-anchor ensemble score normalization to real traffic baselines."""
        try:
            features_array = np.vstack(self._warmup_features)
            ensemble = self._ensemble

            # Re-anchor IsolationForest normalization (percentiles on real data)
            iso_scores = np.array([ensemble.iso_detector.detect(f) for f in features_array])
            ensemble._iso_median = float(np.median(iso_scores))
            ensemble._iso_max = max(
                float(np.percentile(iso_scores, 99)) - ensemble._iso_median, 1e-8
            )

            # Re-anchor statistical detector normalization
            stat_scores = np.array([ensemble.stat_detector.detect(f) for f in features_array])
            ensemble._stat_median = float(np.median(stat_scores))
            ensemble._stat_max = max(
                float(np.percentile(stat_scores, 99)) - ensemble._stat_median, 1e-8
            )

            self._calibrated = True
            logger.info(
                "calibration complete: iso_median=%.4f iso_max=%.4f "
                "stat_median=%.4f stat_max=%.4f (n=%d samples)",
                ensemble._iso_median, ensemble._iso_max,
                ensemble._stat_median, ensemble._stat_max,
                len(features_array),
            )
        except Exception as exc:
            logger.error("calibration failed: %s", exc)
