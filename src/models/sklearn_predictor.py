from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier

from models.base import BasePredictor


class SklearnClassifierPredictor(BasePredictor):
    """Incremental sklearn classifier predictor for vector streams."""

    def __init__(
        self,
        classes: list[int] | None = None,
        loss: str = "log_loss",
        alpha: float = 0.0001,
        random_state: int = 42,
    ):
        self.classes = np.array(classes if classes is not None else [0, 1])
        self.model = SGDClassifier(loss=loss, alpha=alpha, random_state=random_state)
        self._is_fitted = False

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> SklearnClassifierPredictor:
        return cls(
            classes=model_cfg.get("classes"),
            loss=model_cfg.get("loss", "log_loss"),
            alpha=model_cfg.get("alpha", 0.0001),
            random_state=model_cfg.get("random_state", 42),
        )

    def fit_initial(self, X: Any, y: Any) -> None:
        self.model.partial_fit(np.asarray(X), np.asarray(y), classes=self.classes)
        self._is_fitted = True

    def predict_step_online(self, x: Any) -> dict[str, Any]:
        x_arr = np.atleast_2d(np.asarray(x))
        if not self._is_fitted:
            probs = np.full((x_arr.shape[0], len(self.classes)), 1.0 / len(self.classes))
            point_prediction = self.classes[np.argmax(probs, axis=1)]
        else:
            probs = self.model.predict_proba(x_arr)
            point_prediction = self.model.predict(x_arr)

        return {
            "probs": probs,
            "point_prediction": point_prediction,
            "uncertainty": 1.0 - probs.max(axis=1),
        }

    def update_online(self, x: Any, y: Any) -> None:
        self.model.partial_fit(
            np.atleast_2d(np.asarray(x)),
            np.asarray(y).reshape(-1),
            classes=self.classes,
        )
        self._is_fitted = True

    def reset_adaptation_state(self) -> None:
        self._is_fitted = False
