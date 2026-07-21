from __future__ import annotations

from collections import deque
from statistics import mean, pstdev

from drift.base import BaseDriftDetector, DriftEvent


class RunningStatsBaselineDetector(BaseDriftDetector):
    """Simple baseline detector using z-scores against a fixed warmup baseline."""

    def __init__(
        self,
        warmup_size: int = 50,
        threshold_std: float = 3.0,
        consecutive: int = 3,
        min_std: float = 1e-8,
    ) -> None:
        if warmup_size < 2:
            raise ValueError("warmup_size must be >= 2")
        if threshold_std <= 0:
            raise ValueError("threshold_std must be > 0")
        if consecutive < 1:
            raise ValueError("consecutive must be >= 1")
        if min_std <= 0:
            raise ValueError("min_std must be > 0")

        self.warmup_size = warmup_size
        self.threshold_std = threshold_std
        self.consecutive = consecutive
        self.min_std = min_std

        self._baseline_window: deque[float] = deque(maxlen=warmup_size)
        self._baseline_mean: float | None = None
        self._baseline_std: float | None = None
        self._exceedance_streak = 0

    def update(self, value: float) -> DriftEvent:
        value_f = float(value)

        if self._baseline_mean is None or self._baseline_std is None:
            return self._collect_warmup(value_f)

        score = abs(value_f - self._baseline_mean) / max(self._baseline_std, self.min_std)
        exceeds = score >= self.threshold_std
        self._exceedance_streak = self._exceedance_streak + 1 if exceeds else 0
        drift = self._exceedance_streak >= self.consecutive

        if drift:
            self.reset()
            return DriftEvent(drift=True, warning=False, score=score)
        return DriftEvent(drift=False, warning=exceeds, score=score)

    def reset(self) -> None:
        self._baseline_window.clear()
        self._baseline_mean = None
        self._baseline_std = None
        self._exceedance_streak = 0

    def _collect_warmup(self, value: float) -> DriftEvent:
        self._baseline_window.append(value)

        if len(self._baseline_window) < self.warmup_size:
            return DriftEvent(drift=False, warning=False, score=0.0)

        window = list(self._baseline_window)
        self._baseline_mean = mean(window)
        self._baseline_std = pstdev(window)
        return DriftEvent(drift=False, warning=False, score=0.0)

