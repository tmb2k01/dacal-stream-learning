from abc import ABC, abstractmethod


class BaseConformalCalibrator(ABC):
    @abstractmethod
    def predict_set(self, prediction: dict, x=None) -> dict:
        pass

    @abstractmethod
    def update(self, prediction: dict, y_true) -> None:
        pass

    @abstractmethod
    def recalibrate(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
