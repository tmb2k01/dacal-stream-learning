from abc import ABC, abstractmethod
from typing import Dict

class BaseActivePolicy(ABC):
    @abstractmethod
    def should_query(
        self,
        x,
        prediction: Dict,
        conformal_output: Dict,
        state: Dict
    ) -> bool:
        pass

    @abstractmethod
    def update(self, feedback: Dict) -> None:
        pass