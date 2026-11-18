from __future__ import annotations

import numpy as np
import pytest

from drift import (
    ADWINDetector,
    AdaptiveMMDDriftDetector,
    CUSUMDriftDetector,
    DDMDetector,
    KSDriftDetector,
    MMDDriftDetector,
    OracleDriftDetector,
    PageHinkleyDetector,
    RunningStatsBaselineDetector,
)
from engine.stream_engine import StreamSimulationEngine
from models.base import BasePredictor


def test_adwin_detects_shift():
    detector = ADWINDetector(delta=0.01)
    # Stream of 0s
    for _ in range(100):
        ev = detector.update(0.0)
        assert not ev.drift
    # Abrupt shift to 1s
    drift_detected = False
    for _ in range(100):
        ev = detector.update(1.0)
        if ev.drift:
            drift_detected = True
            break
    assert drift_detected


def test_ddm_detects_error_burst():
    detector = DDMDetector(warm_start=30, warning_threshold=2.0, drift_threshold=3.0)
    # Low error rate
    for _ in range(100):
        detector.update(0.0)
    # Sudden high error rate
    drift_detected = False
    for _ in range(60):
        ev = detector.update(1.0)
        if ev.drift:
            drift_detected = True
            break
    assert drift_detected


def test_page_hinkley_detects_shift():
    detector = PageHinkleyDetector(delta=0.01, threshold=10.0, min_instances=30)
    for _ in range(100):
        detector.update(0.1)
    drift_detected = False
    for _ in range(100):
        ev = detector.update(2.0)
        if ev.drift:
            drift_detected = True
            break
    assert drift_detected


def test_oracle_detector_triggers_at_positions():
    oracle = OracleDriftDetector(drift_positions=[50, 100], delay=2)
    assert not oracle.requires_labels

    events = [oracle.update(step) for step in range(110)]
    drifts = [i for i, ev in enumerate(events) if ev.drift]
    assert drifts == [52, 102]


def test_ks_detector_with_reference():
    detector = KSDriftDetector(window_size=40, test_every=10, alpha=0.05)
    rng = np.random.default_rng(42)
    ref = rng.uniform(0.7, 1.0, size=200)
    detector.fit_reference(ref)

    # In-distribution stream
    for _ in range(40):
        ev = detector.update(float(rng.uniform(0.7, 1.0)))
        assert not ev.drift

    # Shifted stream
    drift_detected = False
    for _ in range(60):
        ev = detector.update(float(rng.uniform(0.1, 0.4)))
        if ev.drift:
            drift_detected = True
            break
    assert drift_detected


def test_cusum_detector_with_reference():
    detector = CUSUMDriftDetector(k=0.5, h=3.0, block_size=10)
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.9, scale=0.05, size=200)
    detector.fit_reference(ref)

    # Shifted stream
    drift_detected = False
    for _ in range(100):
        ev = detector.update(float(rng.normal(loc=0.4, scale=0.05)))
        if ev.drift:
            drift_detected = True
            break
    assert drift_detected


def test_mmd_detector_with_reference():
    detector = MMDDriftDetector(window_size=30, test_every=10, alpha=0.05, n_permutations=50, seed=42)
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    detector.fit_reference(ref)

    # Shifted embeddings
    drift_detected = False
    for _ in range(60):
        ev = detector.update(rng.normal(loc=3.0, scale=1.0, size=5))
        if ev.drift:
            drift_detected = True
            break
    assert drift_detected


class DummyPredictor(BasePredictor):
    def predict_step_online(self, x):
        return {
            "probs": np.array([[0.8, 0.2]]),
            "point_prediction": np.array([0]),
            "uncertainty": np.array([0.2]),
            "embedding": np.array([[1.0, 2.0, 3.0]]),
        }

    def update_online(self, x, y):
        pass

    def reset_adaptation_state(self):
        pass


def test_engine_routes_signals_to_detector():
    oracle = OracleDriftDetector(drift_positions=[5], delay=0)
    engine = StreamSimulationEngine(predictor=DummyPredictor(), drift_detector=oracle)

    records = [
        engine.step({"x": np.zeros(3), "y": 0})
        for _ in range(10)
    ]
    drift_steps = [r["step"] for r in records if r["drift_event"].drift]
    assert drift_steps == [5]
