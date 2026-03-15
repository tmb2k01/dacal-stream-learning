from engine import EngineState
from tasks import BaseTask
from models import BasePredictor
from conformal import BaseConformalCalibrator
from active_learning import BaseActivePolicy
from drift import BaseDriftDetector

class DriftAdaptiveActiveLearningEngine:
    def __init__(
        self,
        task: BaseTask,
        predictor: BasePredictor,
        calibrator: BaseConformalCalibrator,
        active_policy: BaseActivePolicy,
        drift_detector: BaseDriftDetector,
        label_oracle,
    ):
        self.task = task
        self.predictor = predictor
        self.calibrator = calibrator
        self.active_policy = active_policy
        self.drift_detector = drift_detector
        self.label_oracle = label_oracle
        self.state = EngineState()

    def step(self, raw_item: dict):
        batch = self.task.preprocess(raw_item)
        x = batch.x

        prediction = self.predictor.predict_step_online(x)
        conformal_output = self.calibrator.predict_set(prediction, x=x)

        query = self.active_policy.should_query(
            x=x,
            prediction=prediction,
            conformal_output=conformal_output,
            state={
                "task": self.task,
                "engine_state": self.state,
                "drift_mode": self.state.drift_mode,
            },
        )

        y = None
        if query:
            y = self.label_oracle.query(batch)
            self.state.queried_count += 1

            signal = self.task.compute_monitoring_signal(prediction, y)
            drift_event = self.drift_detector.update(signal)

            if drift_event.drift:
                self._handle_drift()

            self.predictor.update_online(x, y)
            self.calibrator.update(prediction, y)
            self.active_policy.update({
                "x": x,
                "y": y,
                "prediction": prediction,
                "drift": drift_event.drift,
            })

        self.state.step += 1

        return {
            "prediction": prediction,
            "conformal_output": conformal_output,
            "queried": query,
            "y": y,
        }

    def _handle_drift(self):
        self.state.drift_mode = True
        self.calibrator.reset()
        self.predictor.reset_adaptation_state()