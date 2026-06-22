from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier

from models.base import BasePredictor


class SklearnClassifierPredictor(L.LightningModule, BasePredictor):
    """Incremental sklearn classifier predictor with a LightningModule interface."""

    def __init__(
        self,
        classes: list[int] | None = None,
        loss: str = "log_loss",
        alpha: float = 0.0001,
        random_state: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
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

    def forward(self, x: Any) -> torch.Tensor:
        prediction = self.predict_step_online(x)
        return torch.as_tensor(prediction["probs"], dtype=torch.float32, device=self.device)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        self.update_online(x, y)
        prediction = self.predict_step_online(x)
        loss = self._negative_log_likelihood(prediction["probs"], y)
        self.log("train_loss", loss)
        return loss

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        x, _ = self._unpack_batch(batch) if self._has_targets(batch) else (batch, None)
        return self.predict_step_online(x)

    def configure_optimizers(self):
        return None

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

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        return batch

    @staticmethod
    def _has_targets(batch: Any) -> bool:
        return isinstance(batch, dict) and "y" in batch or isinstance(batch, (tuple, list))

    def _negative_log_likelihood(self, probs: np.ndarray, y: Any) -> torch.Tensor:
        y_arr = np.asarray(y).reshape(-1)
        class_to_column = {label: idx for idx, label in enumerate(self.classes)}
        columns = np.array([class_to_column[label] for label in y_arr])
        clipped = np.clip(probs[np.arange(len(y_arr)), columns], 1e-12, 1.0)
        return torch.as_tensor(-np.log(clipped).mean(), dtype=torch.float32, device=self.device)
