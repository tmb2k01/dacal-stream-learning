from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseActivePolicy(ABC):
    @abstractmethod
    def should_query(self, x, prediction: dict, conformal_output: dict, state: dict) -> bool:
        pass

    @abstractmethod
    def update(self, feedback: dict) -> None:
        pass


class UncertaintySamplingPolicy(BaseActivePolicy):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_query(self, x: Any, prediction: dict, conformal_output: dict, state: dict) -> bool:
        uncertainty = prediction.get("uncertainty")
        if uncertainty is not None:
            return _max_value(uncertainty) >= self.threshold

        probs = prediction.get("probs")
        if probs is None:
            return False
        max_prob = _max_value(probs)
        return 1.0 - max_prob >= self.threshold

    def update(self, feedback: dict) -> None:
        pass


class HybridActivePolicy(BaseActivePolicy):
    def __init__(self, unc_threshold: float, set_size_threshold: int):
        self.unc_threshold = unc_threshold
        self.set_size_threshold = set_size_threshold
        self.uncertainty_policy = UncertaintySamplingPolicy(threshold=unc_threshold)

    def should_query(self, x: Any, prediction: dict, conformal_output: dict, state: dict) -> bool:
        if self.uncertainty_policy.should_query(x, prediction, conformal_output, state):
            return True

        set_size = conformal_output.get("set_size")
        if set_size is None:
            return False
        return _max_value(set_size) >= self.set_size_threshold

    def update(self, feedback: dict) -> None:
        pass


def _max_value(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).max())
