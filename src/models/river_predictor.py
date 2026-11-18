"""River-backed incremental classifiers (method M6: pure test-then-train stream learning).

Formalized from ``playground/lsudden_3_eda.ipynb`` and ``playground/moving_rbf.ipynb``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from river import linear_model, naive_bayes, optim, preprocessing, tree

from models.base import BasePredictor


def _build_river_estimator(estimator: str, learning_rate: float, scale: bool, seed: int):
    if estimator in {"softmax", "softmax_regression"}:
        model = linear_model.SoftmaxRegression(optimizer=optim.SGD(learning_rate))
    elif estimator in {"hoeffding_tree", "hoeffding"}:
        model = tree.HoeffdingTreeClassifier()
    elif estimator in {"adaptive_tree", "hoeffding_adaptive_tree"}:
        model = tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    elif estimator in {"gaussian_nb", "naive_bayes"}:
        model = naive_bayes.GaussianNB()
    else:
        raise ValueError(f"Unsupported river estimator: {estimator!r}")
    return preprocessing.StandardScaler() | model if scale else model


class RiverClassifierPredictor(BasePredictor):
    """Adapter exposing a river classifier through the ``BasePredictor`` API.

    Inputs may be numpy arrays (converted to ``{feature_index: value}`` dicts)
    or river-style feature dicts.  Probabilities are always returned over the
    fixed ``classes`` list so downstream conformal predictors see a stable
    ``(n, K)`` matrix even before every class has been observed.
    """

    def __init__(
        self,
        classes: list[int] | None = None,
        estimator: str = "softmax",
        learning_rate: float = 0.01,
        scale: bool = True,
        seed: int = 42,
    ) -> None:
        self.classes = np.array(classes if classes is not None else [0, 1])
        self.estimator = estimator
        self.learning_rate = float(learning_rate)
        self.scale = bool(scale)
        self.seed = int(seed)
        self.model = _build_river_estimator(estimator, self.learning_rate, self.scale, self.seed)

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> RiverClassifierPredictor:
        num_classes = model_cfg.get("num_classes")
        classes = model_cfg.get("classes")
        if classes is None and num_classes is not None:
            classes = list(range(int(num_classes)))
        return cls(
            classes=classes,
            estimator=model_cfg.get("estimator", "softmax"),
            learning_rate=model_cfg.get("learning_rate", 0.01),
            scale=model_cfg.get("scale", True),
            seed=model_cfg.get("random_state", model_cfg.get("seed", 42)),
        )

    # ------------------------------------------------------------------
    # BasePredictor
    # ------------------------------------------------------------------

    def fit_initial(self, X: Any, y: Any) -> None:
        for x_i, y_i in zip(self._rows(X), np.asarray(y).reshape(-1), strict=True):
            self.model.learn_one(x_i, int(y_i))

    def predict_step_online(self, x: Any) -> dict[str, Any]:
        rows = self._rows(x)
        probs = np.zeros((len(rows), len(self.classes)), dtype=float)
        for i, row in enumerate(rows):
            proba = self.model.predict_proba_one(row) or {}
            for j, cls in enumerate(self.classes):
                probs[i, j] = float(proba.get(int(cls), 0.0))
        totals = probs.sum(axis=1, keepdims=True)
        uniform = np.full_like(probs, 1.0 / len(self.classes))
        probs = np.where(totals > 0, probs / np.where(totals > 0, totals, 1.0), uniform)
        return {
            "probs": probs,
            "point_prediction": self.classes[np.argmax(probs, axis=1)],
            "uncertainty": 1.0 - probs.max(axis=1),
        }

    def update_online(self, x: Any, y: Any) -> None:
        for row, y_i in zip(self._rows(x), np.asarray(y).reshape(-1), strict=True):
            self.model.learn_one(row, int(y_i))

    def reset_adaptation_state(self) -> None:
        """Discard the learned concept (classic reset-on-drift strategy)."""
        self.model = _build_river_estimator(
            self.estimator, self.learning_rate, self.scale, self.seed
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rows(x: Any) -> list[dict]:
        if isinstance(x, dict):
            return [x]
        if isinstance(x, list) and x and isinstance(x[0], dict):
            return x
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        return [{i: float(v) for i, v in enumerate(row)} for row in arr]
