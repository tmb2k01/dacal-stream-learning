from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class StreamBatch:
    x: Any
    y: Optional[Any]
    metadata: Dict[str, Any]

class BaseTask(ABC):
    name: str

    @abstractmethod
    def preprocess(self, raw_item: Dict[str, Any]) -> StreamBatch:
        pass

    @abstractmethod
    def compute_nonconformity(
        self,
        prediction: Any,
        target: Any
    ) -> float:
        pass

    @abstractmethod
    def compute_uncertainty(self, prediction: Any) -> float:
        pass

    @abstractmethod
    def format_prediction_set(
        self,
        prediction: Any,
        threshold: float
    ) -> Any:
        pass

    @abstractmethod
    def compute_monitoring_signal(
        self,
        prediction: Any,
        target: Any
    ) -> float:
        pass