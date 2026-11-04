from __future__ import annotations

import numpy as np

from active_learning import AlwaysQueryPolicy, RandomQueryPolicy
from drift.base import BaseDriftDetector, DriftEvent
from engine.stream_engine import StreamSimulationEngine
from models.base import BasePredictor


class RecordingPredictor(BasePredictor):
    def __init__(self):
        self.updates: list[tuple[float, int]] = []

    def predict_step_online(self, x):
        return {
            "probs": np.array([[0.8, 0.2]]),
            "point_prediction": np.array([0]),
            "uncertainty": np.array([0.2]),
            "embedding": np.array([[1.0, 2.0]]),
        }

    def update_online(self, x, y):
        self.updates.append((float(x[0]), int(y)))

    def reset_adaptation_state(self):
        pass


class RecordingDetector(BaseDriftDetector):
    def __init__(self, signal: str):
        self.signal = signal
        self.seen: list[float] = []

    def update(self, value):
        self.seen.append(value)
        return DriftEvent(drift=False, warning=False, score=0.0)

    def reset(self):
        pass


def _stream(n: int):
    return [{"x": np.array([float(i)]), "y": i % 2, "metadata": {"index": i}} for i in range(n)]


def test_model_is_updated_on_every_queried_step_with_the_true_label():
    predictor = RecordingPredictor()
    engine = StreamSimulationEngine(
        predictor=predictor, active_policy=RandomQueryPolicy(rate=0.3, seed=1), drift_burst=0
    )
    records = engine.run(_stream(200)).records
    queried = [r for r in records if r["queried"]]
    assert 0 < len(queried) < 200
    assert len(predictor.updates) == len(queried)
    assert predictor.updates == [(float(r["step"]), r["step"] % 2) for r in queried]


def test_model_is_never_updated_when_nothing_is_queried():
    predictor = RecordingPredictor()
    engine = StreamSimulationEngine(predictor=predictor, active_policy=None)
    engine.run(_stream(20))
    assert predictor.updates == []


def test_error_detector_only_sees_queried_items_by_default():
    detector = RecordingDetector(signal="error")
    engine = StreamSimulationEngine(
        predictor=RecordingPredictor(),
        active_policy=RandomQueryPolicy(rate=0.25, seed=3),
        drift_detector=detector,
    )
    records = engine.run(_stream(200)).records
    assert len(detector.seen) == sum(r["queried"] for r in records)
    assert len(detector.seen) < 200


def test_error_detector_sees_everything_when_flag_is_off():
    detector = RecordingDetector(signal="error")
    engine = StreamSimulationEngine(
        predictor=RecordingPredictor(),
        active_policy=RandomQueryPolicy(rate=0.25, seed=3),
        drift_detector=detector,
        supervised_signals_require_query=False,
    )
    engine.run(_stream(50))
    assert len(detector.seen) == 50


def test_label_free_detectors_see_every_item():
    for signal in ("confidence", "embedding"):
        detector = RecordingDetector(signal=signal)
        engine = StreamSimulationEngine(
            predictor=RecordingPredictor(),
            active_policy=RandomQueryPolicy(rate=0.1, seed=3),
            drift_detector=detector,
        )
        engine.run(_stream(60))
        assert len(detector.seen) == 60, signal


def test_always_query_updates_on_every_step():
    predictor = RecordingPredictor()
    StreamSimulationEngine(predictor=predictor, active_policy=AlwaysQueryPolicy()).run(_stream(15))
    assert len(predictor.updates) == 15
