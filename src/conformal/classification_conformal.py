from __future__ import annotations

from typing import Any

import numpy as np
import torch

from conformal.base import BaseConformalCalibrator


class ClassificationConformalPredictor(BaseConformalCalibrator):
    """Split-conformal predictor for classification probability outputs."""

    def __init__(self, alpha: float = 0.1, num_classes: int = 10):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.num_classes = num_classes
        self.q_hat: float | None = None
        self.calibration_scores: np.ndarray | None = None

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
        n = len(self.calibration_scores)
        quantile_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(self.calibration_scores, quantile_level, method="higher"))

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

    def update(self, prediction: dict, y_true) -> None:
        pass

    def recalibrate(self) -> None:
        if self.calibration_scores is None:
            raise ValueError("Cannot recalibrate without calibration scores")
        self.calibrate_scores(self.calibration_scores)

    def reset(self) -> None:
        self.q_hat = None
        self.calibration_scores = None

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
