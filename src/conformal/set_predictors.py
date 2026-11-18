"""Alternative conformal set predictors sharing the sliding-window machinery of LAC.

Formalized from ``playground/CIFAR10C_multi_conformal_mmd.ipynb``:

* :class:`MondrianConformalPredictor` – class-conditional LAC quantiles.
* :class:`APSConformalPredictor`      – Adaptive Prediction Sets (non-randomized).
* :class:`TopKConformalPredictor`     – calibrated Top-K sets.

All of them support static calibration, online sliding-window recalibration
(FIFO or class-balanced eviction) and ``reset()`` back to the static
calibration, exactly like :class:`ClassificationConformalPredictor`.
"""

from __future__ import annotations

import math

import numpy as np

from conformal.classification_conformal import ClassificationConformalPredictor


class MondrianConformalPredictor(ClassificationConformalPredictor):
    """Class-conditional (Mondrian) LAC conformal predictor.

    A separate ``(1 - alpha)`` quantile of the LAC score ``1 - p_y`` is kept per
    class, which yields class-conditional coverage guarantees.  When a class
    has fewer than ``min_class_samples`` scores in the current window, its
    threshold falls back to the static calibration threshold for that class
    (or to ``1.0``, i.e. always include, when no calibration is available).
    """

    method_name = "mondrian"

    def __init__(self, *args, min_class_samples: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if min_class_samples is None:
            min_class_samples = max(1, math.ceil(1.0 / self.alpha) - 1)
        if min_class_samples < 1:
            raise ValueError("min_class_samples must be >= 1")
        self.min_class_samples = int(min_class_samples)
        self.q_hat_by_class: np.ndarray | None = None
        self._calibration_q_by_class: np.ndarray | None = None

    def calibrate_scores(self, scores: np.ndarray, labels: np.ndarray | None = None) -> None:
        if labels is None:
            raise ValueError("MondrianConformalPredictor requires calibration labels")
        self._calibration_q_by_class = None
        super().calibrate_scores(scores, labels=labels)
        self._calibration_q_by_class = self.q_hat_by_class.copy()

    def _refresh_q_hat(self, scores: np.ndarray, labels: np.ndarray | None) -> None:
        if labels is None:
            raise ValueError("MondrianConformalPredictor requires labels for every score")
        if self._calibration_q_by_class is not None:
            q_by_class = self._calibration_q_by_class.copy()
        else:
            q_by_class = np.ones(self.num_classes, dtype=float)
        for cls in range(self.num_classes):
            cls_scores = scores[labels == cls]
            if len(cls_scores) >= self.min_class_samples:
                q_by_class[cls] = self._compute_q_hat(cls_scores, self.alpha)
        self.q_hat_by_class = q_by_class
        # Scalar summary kept for logging / API compatibility.
        self.q_hat = float(q_by_class.mean())

    def _build_sets(self, probs: np.ndarray) -> np.ndarray:
        return probs >= (1.0 - self.q_hat_by_class.reshape(1, -1))

    def predict_set(self, prediction: dict, x=None) -> dict:
        output = super().predict_set(prediction, x=x)
        output["q_hat_by_class"] = self.q_hat_by_class.copy()
        return output

    def reset(self) -> None:
        super().reset()
        if self.calibration_scores is None:
            self.q_hat_by_class = None


class APSConformalPredictor(ClassificationConformalPredictor):
    """Adaptive Prediction Sets (Romano et al., 2020), non-randomized variant.

    Score: cumulative probability mass of all classes ranked at or above the
    true label.  Set: smallest prefix of the ranked classes whose cumulative
    mass does not exceed ``q_hat`` (the top-1 class is always included).
    """

    method_name = "aps"

    def nonconformity_scores(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        rank = np.argmax(order == y.reshape(-1, 1), axis=1)
        return cumsum[np.arange(len(y)), rank]

    def _build_sets(self, probs: np.ndarray) -> np.ndarray:
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        # Small tolerance avoids dropping the true label because of float round-off.
        keep = cumsum <= self.q_hat + 1e-12
        keep[:, 0] = True
        sets = np.zeros_like(probs, dtype=bool)
        np.put_along_axis(sets, order, keep, axis=1)
        return sets


class TopKConformalPredictor(ClassificationConformalPredictor):
    """Calibrated Top-K conformal predictor.

    Score: 1-based rank of the true label under the model's probabilities.
    Set: the ``k = q_hat`` most probable classes.
    """

    method_name = "topk"

    def nonconformity_scores(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        order = np.argsort(-probs, axis=1, kind="stable")
        return (np.argmax(order == y.reshape(-1, 1), axis=1) + 1).astype(float)

    def _build_sets(self, probs: np.ndarray) -> np.ndarray:
        k = int(np.clip(round(self.q_hat), 1, probs.shape[1]))
        order = np.argsort(-probs, axis=1, kind="stable")[:, :k]
        sets = np.zeros_like(probs, dtype=bool)
        np.put_along_axis(sets, order, True, axis=1)
        return sets
