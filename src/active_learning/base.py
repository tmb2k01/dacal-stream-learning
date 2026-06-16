from abc import ABC, abstractmethod


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

    def should_query(self, x, prediction: dict, conformal_output: dict, state: dict) -> bool:
        uncertainty = prediction.get("uncertainty")
        if uncertainty is not None:
            if hasattr(uncertainty, "detach"):
                uncertainty = uncertainty.detach().cpu().numpy()
            return float(getattr(uncertainty, "max", lambda: uncertainty)()) >= self.threshold

        probs = prediction.get("probs")
        if probs is None:
            return False
        max_prob = probs.max().item() if hasattr(probs.max(), "item") else float(probs.max())
        return 1.0 - max_prob >= self.threshold

    def update(self, feedback: dict) -> None:
        pass


class HybridActivePolicy(BaseActivePolicy):
    def __init__(self, unc_threshold: float, set_size_threshold: int):
        self.unc_threshold = unc_threshold
        self.set_size_threshold = set_size_threshold

    def should_query(self, x, prediction: dict, conformal_output: dict, state: dict) -> bool:
        uncertainty_policy = UncertaintySamplingPolicy(threshold=self.unc_threshold)
        if uncertainty_policy.should_query(x, prediction, conformal_output, state):
            return True

        prediction_set = conformal_output.get("prediction_set")
        if prediction_set is None:
            return False
        return len(prediction_set) >= self.set_size_threshold

    def update(self, feedback: dict) -> None:
        pass
