from abc import ABC, abstractmethod
from typing import Any


class BasePredictor(ABC):
    @abstractmethod
    def predict_step_online(self, x: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def update_online(self, x: Any, y: Any) -> None:
        pass

    @abstractmethod
    def reset_adaptation_state(self) -> None:
        pass
