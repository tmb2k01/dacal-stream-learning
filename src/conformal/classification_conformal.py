from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from conformal.base import BaseConformalCalibrator


class ClassificationConformalPredictor(BaseConformalCalibrator):
    """Split-conformal predictor for classification probability outputs (LAC score).

    The default nonconformity score is the Least Ambiguous Classifier (LAC)
    score ``s(x, y) = 1 - p_y``.  Subclasses override
    :meth:`nonconformity_scores` and :meth:`_build_sets` to implement other
    set predictors (APS, Top-K) or :meth:`_refresh_q_hat` for
    class-conditional (Mondrian) quantiles while reusing the shared sliding
    window machinery.

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
    min_samples_per_class:
        Minimum number of samples retained per class in the sliding
        window.  When the window is full, the oldest sample whose class holds
        more than ``min_samples_per_class`` entries is evicted instead of the
        oldest sample overall, so rare classes are not flushed out.  ``0``
        (default) gives plain FIFO eviction.
    """

    method_name = "lac"

    def __init__(
        self,
        alpha: float = 0.1,
        num_classes: int = 10,
        window_size: int | None = 500,
        min_online_samples: int = 1,
        min_samples_per_class: int = 0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if min_online_samples < 1:
            raise ValueError("min_online_samples must be >= 1")
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be >= 1 or None")
        if min_samples_per_class < 0:
            raise ValueError("min_samples_per_class must be >= 0")
        if window_size is not None and min_samples_per_class * num_classes > window_size:
            raise ValueError("min_samples_per_class * num_classes must not exceed window_size")

        self.alpha = alpha
        self.num_classes = num_classes
        self.window_size = window_size
        self.min_online_samples = min_online_samples
        self.min_samples_per_class = min_samples_per_class
        self.q_hat: float | None = None

        # Scores (and labels) from the static calibration set, used as fallback after reset.
        self.calibration_scores: np.ndarray | None = None
        self.calibration_labels: np.ndarray | None = None

        # Rolling buffer of nonconformity scores from online labeled samples,
        # with the parallel buffer of their class labels (oldest first).
        self._online_scores: deque[float] = deque()
        self._online_labels: deque[int] = deque()
        self._class_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Score / set hooks (override in subclasses)
    # ------------------------------------------------------------------

    def nonconformity_scores(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the nonconformity score of the true label for each row."""
        return 1.0 - probs[np.arange(len(y)), y]

    def _build_sets(self, probs: np.ndarray) -> np.ndarray:
        """Return a boolean ``(n, K)`` membership matrix for the current ``q_hat``."""
        return probs >= (1.0 - self.q_hat)

    def _refresh_q_hat(self, scores: np.ndarray, labels: np.ndarray | None) -> None:
        """Recompute the threshold(s) from a set of scores (and their labels)."""
        self.q_hat = self._compute_q_hat(scores, self.alpha)

    # ------------------------------------------------------------------
    # Initial (batch) calibration
    # ------------------------------------------------------------------

    def calibrate_loader(self, predictor, loader) -> None:
        probs_parts, label_parts = [], []
        for batch in loader:
            x, y = self._unpack_batch(batch)
            prediction = predictor.predict_step_online(x)
            probs_parts.append(self._to_numpy(prediction["probs"]))
            label_parts.append(self._to_numpy(y).astype(int).reshape(-1))
        self.calibrate_probs(np.concatenate(probs_parts), np.concatenate(label_parts))

    def calibrate_arrays(self, predictor, X: Any, y: Any) -> None:
        prediction = predictor.predict_step_online(X)
        self.calibrate_probs(self._to_numpy(prediction["probs"]), y)

    def calibrate_probs(self, probs: Any, y: Any) -> None:
        """Calibrate from precomputed class probabilities and true labels."""
        probs_np = self._as_2d(self._to_numpy(probs))
        y_np = self._to_numpy(y).astype(int).reshape(-1)
        self.calibrate_scores(self.nonconformity_scores(probs_np, y_np), labels=y_np)

    def calibrate_scores(self, scores: np.ndarray, labels: np.ndarray | None = None) -> None:
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            raise ValueError("calibration scores must not be empty")
        self.calibration_scores = scores
        self.calibration_labels = None if labels is None else np.asarray(labels, dtype=int)
        self._refresh_q_hat(self.calibration_scores, self.calibration_labels)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_set(self, prediction: dict, x=None) -> dict:
        if self.q_hat is None:
            raise ValueError("Conformal predictor is not calibrated yet")
        probs = self._as_2d(self._to_numpy(prediction["probs"]))
        prediction_sets = self._build_sets(probs)
        return {
            "prediction_set": prediction_sets,
            "prediction_set_labels": [
                np.flatnonzero(prediction_set).astype(int).tolist()
                for prediction_set in prediction_sets
            ],
            "set_size": prediction_sets.sum(axis=1),
            "q_hat": self.q_hat,
            "alpha": self.alpha,
            "method": self.method_name,
        }

    # ------------------------------------------------------------------
    # Online recalibration
    # ------------------------------------------------------------------

    def update(self, prediction: dict, y_true) -> None:
        """Incorporate one newly labeled stream sample and adapt ``q_hat``.

        The nonconformity score is appended to the sliding window
        (class-aware eviction keeps at least ``min_samples_per_class`` per class).
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

        score = float(self.nonconformity_scores(probs.reshape(1, -1), np.array([y_idx]))[0])
        self._append_online(score, y_idx)

        if len(self._online_scores) >= self.min_online_samples:
            self._refresh_from_window()

    def recalibrate(self) -> None:
        """Recompute ``q_hat`` from whichever scores are available.

        Priority: online window (if at least ``min_online_samples``) →
        initial calibration scores.
        """
        if len(self._online_scores) >= self.min_online_samples:
            self._refresh_from_window()
        elif self.calibration_scores is not None:
            self._refresh_q_hat(self.calibration_scores, self.calibration_labels)
        else:
            raise ValueError("Cannot recalibrate: no calibration scores available")

    def reset(self) -> None:
        """Clear online window and restore ``q_hat`` from initial calibration.

        Called by the engine after a drift event.  The initial calibration
        scores are preserved so the predictor stays usable immediately after
        the reset.
        """
        self._online_scores.clear()
        self._online_labels.clear()
        self._class_counts.clear()
        if self.calibration_scores is not None:
            self._refresh_q_hat(self.calibration_scores, self.calibration_labels)
        else:
            self.q_hat = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_from_window(self) -> None:
        self._refresh_q_hat(
            np.asarray(self._online_scores, dtype=float),
            np.asarray(self._online_labels, dtype=int),
        )

    def _append_online(self, score: float, label: int) -> None:
        """Add a labeled score, evicting class-aware when the window is full."""
        if self.window_size is not None and len(self._online_scores) >= self.window_size:
            self._evict_one()
        self._online_scores.append(score)
        self._online_labels.append(label)
        self._class_counts[label] = self._class_counts.get(label, 0) + 1

    def _evict_one(self) -> None:
        """Evict the oldest sample whose class exceeds ``min_samples_per_class``.

        Falls back to the oldest sample overall if every class is at or below
        the per-class minimum.
        """
        evict_idx = 0
        if self.min_samples_per_class > 0:
            for idx, label in enumerate(self._online_labels):
                if self._class_counts[label] > self.min_samples_per_class:
                    evict_idx = idx
                    break

        label = self._online_labels[evict_idx]
        del self._online_scores[evict_idx]
        del self._online_labels[evict_idx]
        self._class_counts[label] -= 1
        if self._class_counts[label] == 0:
            del self._class_counts[label]

    @staticmethod
    def _compute_q_hat(scores: np.ndarray, alpha: float) -> float:
        """Return the finite-sample corrected (1-alpha) quantile of *scores*."""
        n = len(scores)
        quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, quantile_level, method="higher"))

    @staticmethod
    def _as_2d(probs: np.ndarray) -> np.ndarray:
        return probs.reshape(1, -1) if probs.ndim == 1 else probs

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


# Explicit alias: the default predictor is the LAC / standard split-conformal method.
LACConformalPredictor = ClassificationConformalPredictor
