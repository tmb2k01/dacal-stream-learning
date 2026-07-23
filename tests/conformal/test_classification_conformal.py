"""Tests for ClassificationConformalPredictor online recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from conformal.classification_conformal import ClassificationConformalPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictor(**kwargs) -> ClassificationConformalPredictor:
    """Return a predictor pre-calibrated on 100 synthetic scores."""
    cp = ClassificationConformalPredictor(alpha=0.1, num_classes=3, **kwargs)
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=100)
    cp.calibrate_scores(scores)
    return cp


def _prediction(probs: list[float]) -> dict:
    """Wrap a prob list in the dict format the engine sends to update()."""
    return {"probs": np.array(probs, dtype=float)}


# ---------------------------------------------------------------------------
# Initial calibration
# ---------------------------------------------------------------------------

class TestInitialCalibration:
    def test_calibrate_scores_sets_q_hat(self):
        cp = ClassificationConformalPredictor(alpha=0.1, num_classes=3)
        cp.calibrate_scores(np.array([0.1, 0.5, 0.9]))
        assert cp.q_hat is not None

    def test_calibrate_scores_empty_raises(self):
        cp = ClassificationConformalPredictor(alpha=0.1, num_classes=3)
        with pytest.raises(ValueError, match="empty"):
            cp.calibrate_scores(np.array([]))

    def test_predict_set_before_calibration_raises(self):
        cp = ClassificationConformalPredictor(alpha=0.1, num_classes=3)
        with pytest.raises(ValueError, match="not calibrated"):
            cp.predict_set({"probs": np.array([0.1, 0.2, 0.7])})

    def test_predict_set_returns_expected_keys(self):
        cp = _make_predictor()
        out = cp.predict_set({"probs": np.array([0.1, 0.2, 0.7])})
        assert {"prediction_set", "prediction_set_labels", "set_size", "q_hat", "alpha"} <= out.keys()

    def test_compute_q_hat_correctness(self):
        # With n=3, alpha=0.1: quantile_level = min(1, ceil(4*0.9)/3) = min(1, ceil(3.6)/3) = 1.0
        scores = np.array([0.2, 0.5, 0.8])
        q = ClassificationConformalPredictor._compute_q_hat(scores, alpha=0.1)
        # quantile at 1.0 → max value
        assert q == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Online update: q_hat adapts after min_online_samples
# ---------------------------------------------------------------------------

class TestOnlineUpdate:
    def test_update_does_not_change_q_hat_below_min_online_samples(self):
        cp = _make_predictor(min_online_samples=5)
        initial_q = cp.q_hat

        # Send 4 updates (below threshold)
        for _ in range(4):
            cp.update(_prediction([0.1, 0.1, 0.8]), y_true=2)

        assert cp.q_hat == pytest.approx(initial_q)

    def test_update_changes_q_hat_once_threshold_reached(self):
        cp = _make_predictor(min_online_samples=5)
        initial_q = cp.q_hat

        # 5 updates with very low nonconformity scores → q_hat should drop
        for _ in range(5):
            cp.update(_prediction([0.01, 0.01, 0.98]), y_true=2)  # score ≈ 0.02

        assert cp.q_hat is not None
        assert cp.q_hat != pytest.approx(initial_q)

    def test_update_with_missing_probs_is_noop(self):
        cp = _make_predictor(min_online_samples=1)
        initial_q = cp.q_hat
        cp.update({}, y_true=0)
        assert cp.q_hat == pytest.approx(initial_q)

    def test_update_with_out_of_range_y_is_noop(self):
        cp = _make_predictor(min_online_samples=1)
        initial_q = cp.q_hat
        cp.update(_prediction([0.1, 0.2, 0.7]), y_true=99)
        assert cp.q_hat == pytest.approx(initial_q)

    def test_q_hat_converges_toward_stream_distribution(self):
        """After many high-score updates q_hat should increase."""
        cp = _make_predictor(min_online_samples=1, window_size=200)
        # Calibrate on low scores so initial q_hat is low
        cp.calibrate_scores(np.zeros(100))
        low_q = cp.q_hat

        # Stream of high nonconformity scores (class 0, low prob)
        for _ in range(200):
            cp.update(_prediction([0.01, 0.495, 0.495]), y_true=0)  # score ≈ 0.99

        assert cp.q_hat > low_q

    def test_online_scores_accumulate_in_buffer(self):
        cp = _make_predictor(min_online_samples=1)
        assert len(cp._online_scores) == 0
        for i in range(5):
            cp.update(_prediction([0.1, 0.1, 0.8]), y_true=2)
        assert len(cp._online_scores) == 5


# ---------------------------------------------------------------------------
# Sliding window eviction
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_window_caps_at_window_size(self):
        cp = _make_predictor(window_size=10, min_online_samples=1)
        for _ in range(25):
            cp.update(_prediction([0.1, 0.1, 0.8]), y_true=2)
        assert len(cp._online_scores) == 10

    def test_old_scores_evicted(self):
        """After filling the window with ~0 scores, one high score should not
        dominate once it's been pushed out."""
        cp = _make_predictor(window_size=5, min_online_samples=1)

        # Fill with near-zero nonconformity scores
        for _ in range(5):
            cp.update(_prediction([0.01, 0.01, 0.98]), y_true=2)  # score ≈ 0.02
        low_q = cp.q_hat

        # Push in 5 high-score samples to completely replace the window
        for _ in range(5):
            cp.update(_prediction([0.99, 0.005, 0.005]), y_true=2)  # score ≈ 0.995
        high_q = cp.q_hat

        assert high_q > low_q

    def test_unbounded_window_grows_indefinitely(self):
        cp = _make_predictor(window_size=None, min_online_samples=1)
        for _ in range(1000):
            cp.update(_prediction([0.1, 0.1, 0.8]), y_true=2)
        assert len(cp._online_scores) == 1000


# ---------------------------------------------------------------------------
# reset() restores initial q_hat
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_online_scores(self):
        cp = _make_predictor(min_online_samples=1)
        for _ in range(10):
            cp.update(_prediction([0.1, 0.1, 0.8]), y_true=2)
        cp.reset()
        assert len(cp._online_scores) == 0

    def test_reset_restores_initial_q_hat(self):
        cp = _make_predictor(min_online_samples=1, window_size=200)
        initial_q = cp.q_hat

        # Shift q_hat far from initial by streaming low-score samples
        for _ in range(200):
            cp.update(_prediction([0.01, 0.01, 0.98]), y_true=2)
        assert cp.q_hat != pytest.approx(initial_q)

        cp.reset()
        assert cp.q_hat == pytest.approx(initial_q)

    def test_reset_without_initial_scores_leaves_q_hat_none(self):
        cp = ClassificationConformalPredictor(alpha=0.1, num_classes=3)
        cp.reset()
        assert cp.q_hat is None

    def test_predictor_still_usable_after_reset(self):
        cp = _make_predictor(min_online_samples=1)
        cp.reset()
        out = cp.predict_set({"probs": np.array([0.1, 0.2, 0.7])})
        assert out["q_hat"] is not None


# ---------------------------------------------------------------------------
# recalibrate()
# ---------------------------------------------------------------------------

class TestRecalibrate:
    def test_recalibrate_uses_online_window_when_available(self):
        cp = _make_predictor(min_online_samples=1, window_size=50)

        # Fill window with near-zero scores to guarantee a very low q_hat
        for _ in range(50):
            cp.update(_prediction([0.01, 0.01, 0.98]), y_true=2)
        online_q = cp.q_hat

        # Manually recalibrate — should produce the same result as the last update
        cp.recalibrate()
        assert cp.q_hat == pytest.approx(online_q)

    def test_recalibrate_falls_back_to_initial_when_below_min_online(self):
        cp = _make_predictor(min_online_samples=10)
        initial_q = cp.q_hat

        # Only 3 updates — below min_online_samples
        for _ in range(3):
            cp.update(_prediction([0.1, 0.1, 0.8]), y_true=2)

        cp.recalibrate()
        assert cp.q_hat == pytest.approx(initial_q)

    def test_recalibrate_raises_with_no_scores_at_all(self):
        cp = ClassificationConformalPredictor(alpha=0.1, num_classes=3, min_online_samples=10)
        with pytest.raises(ValueError, match="no calibration scores"):
            cp.recalibrate()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ClassificationConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError):
            ClassificationConformalPredictor(alpha=1.0)

    def test_invalid_min_online_samples_raises(self):
        with pytest.raises(ValueError):
            ClassificationConformalPredictor(min_online_samples=0)

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            ClassificationConformalPredictor(window_size=0)

    def test_none_window_size_allowed(self):
        cp = ClassificationConformalPredictor(window_size=None)
        assert cp.window_size is None

