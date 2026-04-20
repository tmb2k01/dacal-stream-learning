from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor

from conformal.base import BaseConformalCalibrator
from conformal.conformal_localizer import ConformalLocalizer


class ConformalLocalizerCalibrator(BaseConformalCalibrator):
    """Wraps ConformalLocalizer as a BaseConformalCalibrator for the streaming engine.

    Behaviour
    ---------
    * ``predict_set``:  Returns the per-sample conformal p-value and an anomaly flag.
    * ``update``:       Buffers labelled (x, y) pairs; triggers ``recalibrate`` once
                        the buffer reaches ``min_recalibrate_samples``.
    * ``recalibrate``:  Re-fits the underlying ConformalLocalizer on the current buffer.
    * ``reset``:        Clears the buffer and discards the fitted localizer.
    """

    def __init__(
        self,
        model,
        alpha: float = 0.2,
        cv_params: dict = {},
        cv_runs: int = 5,
        localizer=KNeighborsRegressor(n_neighbors=1),
        n_min_members: int = 100,
        bootstrap_fraction: float = 1.0,
        cover_search: int = 50,
        n_jobs: int = -1,
        min_recalibrate_samples: int = 200,
    ):
        self.alpha = alpha
        self.min_recalibrate_samples = min_recalibrate_samples
        self._localizer = ConformalLocalizer(
            model=model,
            cv_params=cv_params,
            cv_runs=cv_runs,
            localizer=localizer,
            n_min_members=n_min_members,
            bootstrap_fraction=bootstrap_fraction,
            cover_search=cover_search,
            alpha=alpha,
            n_jobs=n_jobs,
        )
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[Any] = []

    # ------------------------------------------------------------------
    # BaseConformalCalibrator interface
    # ------------------------------------------------------------------

    def predict_set(self, prediction: dict, x=None) -> dict:
        """Return conformal p-value and anomaly flag for input *x*.

        Parameters
        ----------
        prediction:
            Raw model prediction dict (passed through unchanged).
        x:
            Feature vector / matrix of shape ``(1, n_features)`` or ``(n_features,)``.

        Returns
        -------
        dict with keys:
            ``p_value``  – conformal p-value(s) from the localizer.
            ``anomaly``  – True when p-value < alpha (anomalous sample).
            ``prediction`` – original prediction dict.
        """
        if self._localizer.localizer_trained is None:
            # Not yet fitted; return None scores
            return {"p_value": None, "anomaly": None, "prediction": prediction}

        x_arr = np.atleast_2d(x)
        p_value = self._localizer.score_samples(x_arr)
        anomaly = bool((p_value < self.alpha).any())
        return {"p_value": p_value, "anomaly": anomaly, "prediction": prediction}

    def update(self, prediction: dict, y_true) -> None:
        """Buffer a labelled sample and recalibrate when enough data is available."""
        x = prediction.get("x") if isinstance(prediction, dict) else None
        if x is not None:
            self._buffer_X.append(np.atleast_1d(x))
            self._buffer_y.append(y_true)
            if len(self._buffer_y) >= self.min_recalibrate_samples:
                self.recalibrate()

    def recalibrate(self) -> None:
        """Re-fit the ConformalLocalizer on the current labelled buffer."""
        if len(self._buffer_y) < 2:
            return
        X = np.vstack(self._buffer_X)
        y = np.array(self._buffer_y)
        self._localizer = clone(self._localizer)
        self._localizer.fit(X, y)

    def reset(self) -> None:
        """Discard the buffer and fitted state (called on drift events)."""
        self._buffer_X = []
        self._buffer_y = []
        self._localizer.localizer_trained = None

