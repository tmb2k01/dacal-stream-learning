from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from engine.state import EngineState
from models import BasePredictor


@dataclass
class SimulationResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    state: EngineState = field(default_factory=EngineState)

    @property
    def predictions(self) -> list[Any]:
        return [record["prediction"] for record in self.records]


class StreamSimulationEngine:
    def __init__(
        self,
        predictor: BasePredictor,
        calibrator=None,
        label_oracle=None,
    ):
        self.predictor = predictor
        self.calibrator = calibrator
        self.label_oracle = label_oracle
        self.state = EngineState()

    @classmethod
    def from_components(
        cls,
        dataset,
        predictor: BasePredictor,
        calibrator=None,
    ) -> StreamSimulationEngine:
        engine = cls(
            predictor=predictor,
            calibrator=calibrator,
            label_oracle=dataset,
        )
        engine.fit_initial(dataset)
        return engine

    def fit_initial(self, dataset) -> None:
        if hasattr(self.predictor, "fit_initial") and hasattr(dataset, "X_train"):
            self.predictor.fit_initial(dataset.X_train, dataset.y_train)

    def run(
        self, stream: Iterable[dict[str, Any]], max_steps: int | None = None
    ) -> SimulationResult:
        records = []
        for idx, raw_item in enumerate(stream):
            if max_steps is not None and idx >= max_steps:
                break
            records.append(self.step(raw_item))
        return SimulationResult(records=records, state=self.state)

    def step(self, raw_item: dict):
        x = raw_item["x"]
        y_true = raw_item.get("y")
        metadata = raw_item.get("metadata", {})

        prediction = self.predictor.predict_step_online(x)
        conformal_output = (
            self.calibrator.predict_set(prediction, x=x) if self.calibrator is not None else {}
        )

        point_prediction = prediction.get("point_prediction")
        correct = self._is_correct(point_prediction, y_true)
        self._update_metrics(correct=correct, y_true=y_true, conformal_output=conformal_output)
        self.state.step += 1

        return {
            "step": self.state.step - 1,
            "metadata": metadata,
            "prediction": prediction,
            "conformal_output": conformal_output,
            "y_true": y_true,
            "correct": correct,
        }

    def _update_metrics(
        self,
        correct: bool | None,
        y_true: Any,
        conformal_output: dict,
    ) -> None:
        if correct is None:
            return
        metrics = self.state.metrics
        metrics["seen"] = metrics.get("seen", 0) + 1
        metrics["correct"] = metrics.get("correct", 0) + int(correct)
        metrics["accuracy"] = metrics["correct"] / metrics["seen"]

        set_size = conformal_output.get("set_size")
        prediction_set = conformal_output.get("prediction_set")
        if set_size is None or prediction_set is None:
            return

        size = int(self._first_value(set_size))
        y_class = int(y_true)
        covered = bool(prediction_set[0, y_class])
        num_classes = int(prediction_set.shape[1])

        metrics["total_set_size"] = metrics.get("total_set_size", 0) + size
        metrics["efficiency"] = metrics["total_set_size"] / metrics["seen"]
        metrics["informativeness"] = 1.0 - (metrics["efficiency"] / num_classes)

        class_counts = metrics.setdefault("class_counts", {})
        class_covered = metrics.setdefault("class_covered", {})
        class_key = str(y_class)
        class_counts[class_key] = class_counts.get(class_key, 0) + 1
        class_covered[class_key] = class_covered.get(class_key, 0) + int(covered)
        metrics["coverage_by_class"] = {
            cls: class_covered[cls] / count for cls, count in class_counts.items()
        }
        target_coverage = 1.0 - conformal_output.get("alpha", 0.1)
        metrics["coverage_gap_by_class"] = {
            cls: coverage - target_coverage
            for cls, coverage in metrics["coverage_by_class"].items()
        }

    @staticmethod
    def _is_correct(point_prediction: Any, y_true: Any) -> bool | None:
        if y_true is None or point_prediction is None:
            return None
        if hasattr(point_prediction, "detach"):
            point_prediction = point_prediction.detach().cpu().numpy()
        point_arr = getattr(point_prediction, "reshape", lambda *_: [point_prediction])(-1)
        return int(point_arr[0]) == int(y_true)

    def _handle_drift(self):
        self.state.drift_mode = True
        if self.calibrator is not None:
            self.calibrator.reset()
        self.predictor.reset_adaptation_state()

    @staticmethod
    def _first_value(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "reshape"):
            return value.reshape(-1)[0]
        if isinstance(value, list | tuple):
            return value[0]
        return value


class DriftAdaptiveActiveLearningEngine(StreamSimulationEngine):
    def __init__(
        self,
        task,
        predictor: BasePredictor,
        calibrator,
        active_policy,
        drift_detector,
        label_oracle,
    ):
        super().__init__(
            predictor=predictor,
            calibrator=calibrator,
            label_oracle=label_oracle,
        )
