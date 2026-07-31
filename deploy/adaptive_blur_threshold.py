"""Adaptive Laplacian-variance threshold for blur rejection."""

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ThresholdUpdate:
    """Summary of one completed threshold update."""

    previous_threshold: float
    threshold: float
    lower_quantile: float
    sample_count: int


class AdaptiveBlurThreshold:
    """Update a blur threshold from a time window of raw sharpness values.

    Each observation is an unfiltered Laplacian variance.  The threshold used
    for the current frame remains unchanged; a completed window updates the
    threshold for subsequent frames.
    """

    def __init__(
        self,
        initial_threshold: float = 150.0,
        update_interval: float = 2.0,
        quantile: float = 0.2,
        alpha: float = 0.2,
        lower_scale: float = 0.5,
        upper_scale: float = 1.5,
    ):
        if initial_threshold <= 0:
            raise ValueError("initial_threshold must be positive")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if lower_scale <= 0 or upper_scale < lower_scale:
            raise ValueError("threshold scales must satisfy 0 < lower <= upper")

        self.initial_threshold = float(initial_threshold)
        self.update_interval = float(update_interval)
        self.quantile = float(quantile)
        self.alpha = float(alpha)
        self.minimum_threshold = lower_scale * self.initial_threshold
        self.maximum_threshold = upper_scale * self.initial_threshold

        self.threshold = self.initial_threshold
        self._window_start: Optional[float] = None
        self._variances = []

    def observe(
        self, laplacian_variance: float, timestamp: Optional[float] = None
    ) -> Optional[ThresholdUpdate]:
        """Buffer one raw variance and update after ``update_interval`` seconds."""

        variance = float(laplacian_variance)
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("laplacian_variance must be finite and non-negative")

        now = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(now):
            raise ValueError("timestamp must be finite")

        if self._window_start is None:
            self._window_start = now
        elif now < self._window_start:
            raise ValueError("timestamps must be monotonic")

        self._variances.append(variance)
        if now - self._window_start < self.update_interval:
            return None

        lower_quantile = float(np.quantile(self._variances, self.quantile))
        previous_threshold = self.threshold
        smoothed_threshold = (
            (1.0 - self.alpha) * previous_threshold
            + self.alpha * lower_quantile
        )
        self.threshold = float(
            np.clip(
                smoothed_threshold,
                self.minimum_threshold,
                self.maximum_threshold,
            )
        )

        update = ThresholdUpdate(
            previous_threshold=previous_threshold,
            threshold=self.threshold,
            lower_quantile=lower_quantile,
            sample_count=len(self._variances),
        )
        self._variances.clear()
        self._window_start = now
        return update
