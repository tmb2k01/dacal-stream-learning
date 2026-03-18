from abc import ABC, abstractmethod


class BaseActivePolicy(ABC):
    @abstractmethod
    def should_query(self, x, prediction: dict, conformal_output: dict, state: dict) -> bool:
        pass

    @abstractmethod
    def update(self, feedback: dict) -> None:
        pass
