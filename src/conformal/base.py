from abc import ABC, abstractmethod
from typing import Dict

class BaseConformalCalibrator(ABC):
    @abstractmethod
    def predict_set(self, prediction: Dict, x=None) -> Dict:
        pass

    @abstractmethod
    def update(self, prediction: Dict, y_true) -> None:
        pass

    @abstractmethod
    def recalibrate(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass