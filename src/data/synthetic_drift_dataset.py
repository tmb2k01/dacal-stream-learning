from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from data.stream_dataset import BaseStreamDataset


class SyntheticDriftStreamDataset(BaseStreamDataset):
    """Synthetic binary stream with abrupt phase-wise concept drift."""

    def __init__(
        self,
        seed: int = 42,
        n_per_phase: int = 1_000,
        n_phases: int = 4,
        n_features: int = 10,
        noise_std: float = 0.6,
        drift_angle: float = float(np.pi / 2),
    ):
        if n_per_phase < 2:
            raise ValueError("n_per_phase must be >= 2")
        if n_per_phase % 2 != 0:
            raise ValueError("n_per_phase must be even")
        if n_phases < 2:
            raise ValueError("n_phases must be >= 2")
        if n_features < 2:
            raise ValueError("n_features must be >= 2")

        self.seed = seed
        self.n_per_phase = n_per_phase
        self.n_phases = n_phases
        self.n_features = n_features
        self.noise_std = noise_std
        self.drift_angle = drift_angle

        self.X_train: np.ndarray
        self.y_train: np.ndarray
        self.X_stream: np.ndarray
        self.y_stream: np.ndarray
        self.phase_idx: np.ndarray
        self.drift_positions: list[int]
        self._build_stream()

    def _build_stream(self) -> None:
        rng = np.random.default_rng(self.seed)
        phases_X: list[np.ndarray] = []
        phases_y: list[np.ndarray] = []

        n_per_class = self.n_per_phase // 2
        for phase in range(self.n_phases):
            X_phase, y_phase = self._make_phase(
                rng=rng,
                n_per_class=n_per_class,
                n_features=self.n_features,
                noise_std=self.noise_std,
                rotation=phase * self.drift_angle,
            )
            phases_X.append(X_phase)
            phases_y.append(y_phase)

        self.X_train = phases_X[0]
        self.y_train = phases_y[0]
        self.X_stream = np.vstack(phases_X)
        self.y_stream = np.hstack(phases_y)
        self.phase_idx = np.repeat(np.arange(self.n_phases), self.n_per_phase)
        self.drift_positions = [self.n_per_phase * p for p in range(1, self.n_phases)]

    @staticmethod
    def _rotation_matrix(n_features: int, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.eye(n_features)
        rot[0, 0], rot[0, 1] = c, -s
        rot[1, 0], rot[1, 1] = s, c
        return rot

    def _make_phase(
        self,
        rng: np.random.Generator,
        n_per_class: int,
        n_features: int,
        noise_std: float,
        rotation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_mean = np.zeros(n_features)
        base_mean[0] = 1.5
        rot = self._rotation_matrix(n_features=n_features, angle=rotation)
        mean0 = rot @ base_mean
        mean1 = -rot @ base_mean

        x0 = rng.normal(loc=mean0, scale=noise_std, size=(n_per_class, n_features))
        x1 = rng.normal(loc=mean1, scale=noise_std, size=(n_per_class, n_features))
        X = np.vstack([x0, x1])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def __len__(self) -> int:
        return int(self.X_stream.shape[0])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        drift_points = set(self.drift_positions)
        for idx, (x_i, y_i, phase_i) in enumerate(
            zip(self.X_stream, self.y_stream, self.phase_idx, strict=False)
        ):
            yield {
                "x": x_i,
                "y": int(y_i),
                "metadata": {
                    "index": idx,
                    "phase": int(phase_i),
                    "is_drift_point": idx in drift_points,
                },
            }

    def query(self, batch: Any) -> int:
        """Return the true label for a batch produced from this stream."""
        if isinstance(batch, dict):
            metadata = batch.get("metadata", {})
            idx = metadata.get("index")
            if idx is not None:
                return int(self.y_stream[idx])
            if batch.get("y") is not None:
                return int(batch["y"])

        metadata = getattr(batch, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("index") is not None:
            return int(self.y_stream[metadata["index"]])

        y_value = getattr(batch, "y", None)
        if isinstance(y_value, (int, float, np.integer, np.floating, bool, np.bool_)):
            return int(y_value)

        raise ValueError("Batch does not contain index or label information for oracle query.")

