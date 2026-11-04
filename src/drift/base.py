from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DriftEvent:
    drift: bool
    warning: bool
    score: float


class BaseDriftDetector(ABC):
    """Detectors reset their own state when update() reports drift."""

    @abstractmethod
    def update(self, value: float) -> DriftEvent:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
