from __future__ import annotations

from collections.abc import Iterable

from drift.base import BaseDriftDetector, DriftEvent


class OracleDriftDetector(BaseDriftDetector):
    """Privileged detector that fires exactly at the ground-truth drift positions.

    Consumes the stream index (``signal = "step"``) and raises a drift signal
    at ``position + delay`` for every known drift position.  Used as the B3
    upper bound for reset-based adaptation; it uses no labels but has access
    to information no real detector has.
    """

    signal = "step"

    def __init__(self, drift_positions: Iterable[int] = (), delay: int = 0) -> None:
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.delay = int(delay)
        self.set_drift_positions(drift_positions)

    def set_drift_positions(self, drift_positions: Iterable[int]) -> None:
        self.drift_positions = sorted(int(p) for p in drift_positions)
        self._triggers = {p + self.delay for p in self.drift_positions}

    @property
    def requires_labels(self) -> bool:
        return False

    def update(self, value: float) -> DriftEvent:
        drift = int(value) in self._triggers
        return DriftEvent(drift=drift, warning=False, score=1.0 if drift else 0.0)

    def reset(self) -> None:
        pass
