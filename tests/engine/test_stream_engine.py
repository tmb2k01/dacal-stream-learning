from drift.base import BaseDriftDetector, DriftEvent
from engine.stream_engine import StreamSimulationEngine


class FixedPredictor:
    def __init__(self):
        self.reset_count = 0
        self.updates = []

    def predict_step_online(self, x):
        return {"point_prediction": 0}

    def update_online(self, x, y):
        self.updates.append(y)

    def reset_adaptation_state(self):
        self.reset_count += 1


class SelectivePolicy:
    def should_query(self, x, prediction, conformal_output, state):
        return x == "queried"

    def update(self, feedback):
        pass


class RecordingDetector(BaseDriftDetector):
    def __init__(self, trigger=False):
        self.values = []
        self.reset_count = 0
        self.trigger = trigger

    def update(self, value):
        self.values.append(value)
        drift = self.trigger
        if drift:
            self.reset()
        return DriftEvent(drift=drift, warning=False, score=value)

    def reset(self):
        self.reset_count += 1


def test_active_detector_only_sees_queried_errors_but_metrics_see_all_labels():
    detector = RecordingDetector()
    predictor = FixedPredictor()
    engine = StreamSimulationEngine(
        predictor=predictor, active_policy=SelectivePolicy(), drift_detector=detector
    )

    records = engine.run([{"x": "unqueried", "y": 1}, {"x": "queried", "y": 1}]).records

    assert detector.values == [1.0]
    assert predictor.updates == [1]
    assert records[0]["correct"] is False
    assert records[0]["drift_event"] is None
    assert records[1]["drift_event"].score == 1.0
    assert engine.state.metrics["seen"] == 2


def test_no_policy_preserves_fully_labeled_detector_updates():
    detector = RecordingDetector()
    engine = StreamSimulationEngine(predictor=FixedPredictor(), drift_detector=detector)

    engine.run([{"x": "first", "y": 0}, {"x": "second", "y": 1}])

    assert detector.values == [0.0, 1.0]


def test_engine_does_not_reset_detector_twice_after_drift():
    detector = RecordingDetector(trigger=True)
    predictor = FixedPredictor()
    engine = StreamSimulationEngine(predictor=predictor, drift_detector=detector)

    record = engine.step({"x": "first", "y": 1})

    assert record["drift_event"].drift
    assert detector.reset_count == 1
    assert predictor.reset_count == 1
