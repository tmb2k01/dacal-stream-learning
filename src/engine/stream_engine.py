from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from active_learning import BaseActivePolicy
from conformal import BaseConformalCalibrator
from drift import BaseDriftDetector
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
        calibrator: BaseConformalCalibrator | None = None,
        active_policy: BaseActivePolicy | None = None,
        drift_detector: BaseDriftDetector | None = None,
        label_oracle: Any | None = None,
    ):
        self.predictor: BasePredictor = predictor
        self.calibrator = calibrator
        self.active_policy = active_policy
        self.drift_detector = drift_detector
        self.label_oracle = label_oracle
        self.state = EngineState()

    @classmethod
    def from_components(
        cls,
        dataset,
        predictor: BasePredictor,
        calibrator: BaseConformalCalibrator | None = None,
        active_policy: BaseActivePolicy | None = None,
        drift_detector: BaseDriftDetector | None = None,
    ) -> StreamSimulationEngine:
        engine = cls(
            predictor=predictor,
            calibrator=calibrator,
            active_policy=active_policy,
            drift_detector=drift_detector,
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
        records: list[dict[str, Any]] = []
        for idx, raw_item in enumerate(stream):
            if max_steps is not None and idx >= max_steps:
                break
            records.append(self.step(raw_item))
        return SimulationResult(records=records, state=self.state)

    def step(self, raw_item: dict[str, Any]) -> dict[str, Any]:
        """Run one stream item through the full simulation workflow."""
        step_index = self.state.step
        x = raw_item["x"]
        y_true = raw_item.get("y")
        metadata = raw_item.get("metadata", {})

        # 1. Model prediction: every downstream component receives this dict.
        prediction = self.predictor.predict_step_online(x)

        # 2. Optional conformal prediction: wraps probabilities into a set/p-value view.
        conformal_output = (
            self.calibrator.predict_set(prediction, x=x) if self.calibrator is not None else {}
        )

        # 3. Optional active learning: decide whether this label should be revealed.
        policy_state = self._policy_state(step_index=step_index, metadata=metadata)
        queried = (
            self.active_policy.should_query(x, prediction, conformal_output, policy_state)
            if self.active_policy is not None
            else False
        )
        y_observed = self._query_label(raw_item) if queried else None

        # 4. Optional online update: only queried labels update adaptive components.
        if y_observed is not None:
            self.state.queried_count += 1
            self.state.labeled_count += 1
            self.predictor.update_online(x, y_observed)
            if self.calibrator is not None:
                self.calibrator.update({"x": x, **prediction}, y_observed)
            if self.active_policy is not None:
                self.active_policy.update(
                    {
                        "x": x,
                        "y": y_observed,
                        "prediction": prediction,
                        "conformal_output": conformal_output,
                        "metadata": metadata,
                    }
                )

        # 5. Optional drift detection: by default detectors consume prediction error.
        point_prediction = prediction.get("point_prediction")
        correct = self._is_correct(point_prediction, y_true)
        drift_event = self._update_drift_detector(correct)
        if drift_event is not None and drift_event.drift:
            self._handle_drift()

        # 6. Metrics and record: keep raw outputs visible for analysis notebooks.
        self._update_metrics(correct=correct, y_true=y_true, conformal_output=conformal_output)
        self.state.step += 1

        return {
            "step": step_index,
            "metadata": metadata,
            "prediction": prediction,
            "conformal_output": conformal_output,
            "y_true": y_true,
            "queried": queried,
            "y_observed": y_observed,
            "correct": correct,
            "drift_event": drift_event,
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
        if self.drift_detector is not None:
            self.drift_detector.reset()

    @staticmethod
    def _first_value(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "reshape"):
            return value.reshape(-1)[0]
        if isinstance(value, list | tuple):
            return value[0]
        return value

    def _policy_state(self, step_index: int, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": step_index,
            "metadata": metadata,
            "metrics": self.state.metrics,
            "drift_mode": self.state.drift_mode,
            "queried_count": self.state.queried_count,
            "labeled_count": self.state.labeled_count,
        }

    def _query_label(self, raw_item: dict[str, Any]) -> Any:
        if self.label_oracle is not None and hasattr(self.label_oracle, "query"):
            return self.label_oracle.query(raw_item)
        return raw_item.get("y")

    def _update_drift_detector(self, correct: bool | None):
        if self.drift_detector is None or correct is None:
            return None
        error = 0.0 if correct else 1.0
        return self.drift_detector.update(error)
