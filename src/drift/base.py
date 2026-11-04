from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DriftEvent:
    drift: bool
    warning: bool
    score: float


class BaseDriftDetector(ABC):
    """Base interface for all drift detectors.

    ``signal`` declares which per-step stream signal the engine feeds into
    :meth:`update`:

    * ``"error"``      – 0/1 prediction error (requires the true label).
    * ``"loss"``       – per-sample log loss (requires the true label).
    * ``"confidence"`` – max class probability (label-free).
    * ``"set_size"``   – conformal set size (label-free).
    * ``"embedding"``  – model embedding / feature vector (label-free).
    * ``"step"``       – the stream index (used by the oracle detector).
    """

    signal: str = "error"

    @property
    def requires_labels(self) -> bool:
        return self.signal in {"error", "loss"}

    @abstractmethod
    def update(self, value: float) -> DriftEvent:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
