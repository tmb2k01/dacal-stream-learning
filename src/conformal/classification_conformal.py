from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from conformal.base import BaseConformalCalibrator


class ClassificationConformalPredictor(BaseConformalCalibrator):
    """Split-conformal predictor for classification probability outputs.

    Online recalibration
    --------------------
    When labeled samples arrive during streaming (via ``update()``), their
    nonconformity scores are appended to a sliding window.  Once the window
    contains at least ``min_online_samples`` entries, ``q_hat`` is recomputed
    from that window alone, allowing the threshold to track distribution shift.

    Parameters
    ----------
    alpha:
        Miscoverage level (0 < alpha < 1).
    num_classes:
        Number of output classes.
    window_size:
        Maximum number of online nonconformity scores to retain.  Oldest
        scores are evicted once the cap is reached.  ``None`` means the
        window grows without bound.
    min_online_samples:
        Minimum number of online scores required before ``q_hat`` is updated
        from the sliding window.  Until this threshold is reached the
        predictor uses the ``q_hat`` set during initial calibration.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        num_classes: int = 10,
        window_size: int | None = 500,
        min_online_samples: int = 1,
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if min_online_samples < 1:
            raise ValueError("min_online_samples must be >= 1")
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be >= 1 or None")

        self.alpha = alpha
        self.num_classes = num_classes
        self.window_size = window_size
        self.min_online_samples = min_online_samples

        self.q_hat: float | None = None
        # Scores from the static calibration set (used as fallback after reset).
        self.calibration_scores: np.ndarray | None = None
        # Rolling buffer of nonconformity scores from online labeled samples.
        self._online_scores: deque[float] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Initial (batch) calibration
    # ------------------------------------------------------------------

    def calibrate_loader(self, predictor, loader) -> None:
        scores = []
        for batch in loader:
            x, y = self._unpack_batch(batch)
            prediction = predictor.predict_step_online(x)
            probs = self._to_numpy(prediction["probs"])
            y_np = self._to_numpy(y).astype(int).reshape(-1)
            scores.extend(1.0 - probs[np.arange(len(y_np)), y_np])
        self.calibrate_scores(np.asarray(scores, dtype=float))

    def calibrate_arrays(self, predictor, X: Any, y: Any) -> None:
        prediction = predictor.predict_step_online(X)
        probs = self._to_numpy(prediction["probs"])
        y_np = self._to_numpy(y).astype(int).reshape(-1)
        self.calibrate_scores(1.0 - probs[np.arange(len(y_np)), y_np])

    def calibrate_scores(self, scores: np.ndarray) -> None:
        if scores.size == 0:
            raise ValueError("calibration scores must not be empty")
        self.calibration_scores = np.asarray(scores, dtype=float)
        self.q_hat = self._compute_q_hat(self.calibration_scores, self.alpha)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_set(self, prediction: dict, x=None) -> dict:
        if self.q_hat is None:
            raise ValueError("Conformal predictor is not calibrated yet")
        probs = self._to_numpy(prediction["probs"])
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        prediction_sets = probs >= (1.0 - self.q_hat)
        return {
            "prediction_set": prediction_sets,
            "prediction_set_labels": [
                np.flatnonzero(prediction_set).astype(int).tolist()
                for prediction_set in prediction_sets
            ],
            "set_size": prediction_sets.sum(axis=1),
            "q_hat": self.q_hat,
            "alpha": self.alpha,
        }

    # ------------------------------------------------------------------
    # Online recalibration
    # ------------------------------------------------------------------

    def update(self, prediction: dict, y_true) -> None:
        """Incorporate one newly labeled stream sample and adapt ``q_hat``.

        The nonconformity score ``1 - p_y`` is appended to the sliding window.
        ``q_hat`` is updated once the window holds at least
        ``min_online_samples`` entries.

        Parameters
        ----------
        prediction:
            Dict produced by the engine step; must contain key ``"probs"``
            (the raw softmax probabilities for each class).
        y_true:
            Ground-truth class index for the current sample.
        """
        probs_raw = prediction.get("probs")
        if probs_raw is None:
            return

        probs = self._to_numpy(probs_raw)
        if probs.ndim == 2:
            if probs.shape[0] != 1:
                return
            probs = probs[0]
        elif probs.ndim != 1:
            return
        y_idx = int(y_true)
        if y_idx < 0 or y_idx >= len(probs):
            return

        score = float(1.0 - probs[y_idx])
        self._online_scores.append(score)

        if len(self._online_scores) >= self.min_online_samples:
            self.q_hat = self._compute_q_hat(
                np.asarray(self._online_scores, dtype=float), self.alpha
            )

    def recalibrate(self) -> None:
        """Recompute ``q_hat`` from whichever scores are available.

        Priority: online window (if at least ``min_online_samples``) →
        initial calibration scores.
        """
        if len(self._online_scores) >= self.min_online_samples:
            self.q_hat = self._compute_q_hat(
                np.asarray(self._online_scores, dtype=float), self.alpha
            )
        elif self.calibration_scores is not None:
            self.q_hat = self._compute_q_hat(self.calibration_scores, self.alpha)
        else:
            raise ValueError("Cannot recalibrate: no calibration scores available")

    def reset(self) -> None:
        """Clear online window and restore ``q_hat`` from initial calibration.

        Called by the engine after a drift event.  The initial calibration
        scores are preserved so the predictor stays usable immediately after
        the reset.
        """
        self._online_scores.clear()
        if self.calibration_scores is not None:
            self.q_hat = self._compute_q_hat(self.calibration_scores, self.alpha)
        else:
            self.q_hat = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_q_hat(scores: np.ndarray, alpha: float) -> float:
        """Return the finite-sample corrected (1-alpha) quantile of *scores*."""
        n = len(scores)
        quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, quantile_level, method="higher"))

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        return batch

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)
