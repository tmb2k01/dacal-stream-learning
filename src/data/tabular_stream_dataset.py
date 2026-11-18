"""Tabular drift streams: river synthetic generators and CSV benchmark files.

Used by method M6 (pure incremental river baselines) and the tabular columns
of the benchmark matrix.  All streams expose the ``BaseStreamDataset``
contract (``X_stream``, ``y_stream``, ``drift_positions``) plus held-out
``X_train`` / ``y_train`` and optional ``X_cal`` / ``y_cal`` drawn from the
pre-drift concept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from river.datasets import synth

from data.stream_dataset import BaseStreamDataset


def _take(generator, n: int, n_features: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for x_i, y_i in generator.take(n):
        keys = sorted(x_i)
        X.append([float(x_i[k]) for k in keys])
        y.append(int(y_i))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


class RiverSynthStreamDataset(BaseStreamDataset):
    """Abrupt-drift stream built by concatenating river generator concepts.

    Supported generators:

    * ``sea``     – SEA concepts; ``concepts`` are SEA variants (0–3).
    * ``rbf``     – RandomRBF; each concept is a different ``seed_model``.
    * ``agrawal`` – Agrawal; ``concepts`` are classification functions (0–9).
    """

    dataset_name = "river_synth"

    def __init__(
        self,
        generator: str = "sea",
        n_per_phase: int = 1000,
        concepts: list[int] | None = None,
        seed: int = 42,
        noise: float = 0.0,
        n_train: int = 1000,
        n_calibration: int = 1000,
        n_classes: int = 3,
        n_features: int = 10,
        n_centroids: int = 20,
    ) -> None:
        if n_per_phase < 1:
            raise ValueError("n_per_phase must be >= 1")
        self.generator = generator
        self.n_per_phase = int(n_per_phase)
        self.concepts = list(concepts) if concepts is not None else [0, 1, 2, 3]
        if len(self.concepts) < 2:
            raise ValueError("at least two concepts are required to create drift")
        self.seed = int(seed)
        self.noise = float(noise)
        self.n_train = int(n_train)
        self.n_calibration = int(n_calibration)
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.n_centroids = int(n_centroids)
        self._build_stream()

    @classmethod
    def from_config(cls, dataset_cfg: dict[str, Any]) -> RiverSynthStreamDataset:
        return cls(
            generator=dataset_cfg.get("generator", "sea"),
            n_per_phase=dataset_cfg.get("n_per_phase", 1000),
            concepts=dataset_cfg.get("concepts"),
            seed=dataset_cfg.get("seed", 42),
            noise=dataset_cfg.get("noise", 0.0),
            n_train=dataset_cfg.get("n_train", 1000),
            n_calibration=dataset_cfg.get("n_calibration", 1000),
            n_classes=dataset_cfg.get("n_classes", 3),
            n_features=dataset_cfg.get("n_features", 10),
            n_centroids=dataset_cfg.get("n_centroids", 20),
        )

    def _make_generator(self, concept: int, sample_seed: int):
        if self.generator == "sea":
            return synth.SEA(variant=int(concept), noise=self.noise, seed=sample_seed)
        if self.generator == "rbf":
            return synth.RandomRBF(
                seed_model=self.seed * 1000 + int(concept),
                seed_sample=sample_seed,
                n_classes=self.n_classes,
                n_features=self.n_features,
                n_centroids=self.n_centroids,
            )
        if self.generator == "agrawal":
            return synth.Agrawal(
                classification_function=int(concept),
                seed=sample_seed,
                perturbation=self.noise,
            )
        raise ValueError(f"Unsupported river generator: {self.generator!r}")

    def _build_stream(self) -> None:
        parts_X, parts_y = [], []
        for phase, concept in enumerate(self.concepts):
            X_p, y_p = _take(self._make_generator(concept, self.seed + phase), self.n_per_phase)
            parts_X.append(X_p)
            parts_y.append(y_p)
        self.X_stream = np.vstack(parts_X)
        self.y_stream = np.hstack(parts_y)
        self.phase_idx = np.repeat(np.arange(len(self.concepts)), self.n_per_phase)
        self.drift_positions = [self.n_per_phase * p for p in range(1, len(self.concepts))]

        # Held-out pre-drift data from the phase-0 concept with independent sample seeds.
        self.X_train, self.y_train = _take(
            self._make_generator(self.concepts[0], self.seed + 10_007), self.n_train
        )
        if self.n_calibration > 0:
            self.X_cal, self.y_cal = _take(
                self._make_generator(self.concepts[0], self.seed + 20_011), self.n_calibration
            )
        self.num_classes = int(max(self.y_stream.max(), self.y_train.max()) + 1)

    def _metadata_for_index(self, idx: int) -> dict[str, Any]:
        return {"phase": int(self.phase_idx[idx])}


class CSVStreamDataset(BaseStreamDataset):
    """Stream read from a CSV drift benchmark (e.g. ``data/LSudden_3.csv``).

    The first ``n_train`` rows train the initial model, the next
    ``n_calibration`` rows calibrate the conformal predictor, and the
    remaining rows (up to ``max_rows``) form the stream.  ``drift_positions``
    are given in *original file row* coordinates and shifted to stream
    coordinates automatically.
    """

    dataset_name = "csv_stream"

    def __init__(
        self,
        path: str | Path,
        label_column: str | int = -1,
        drift_positions: list[int] | None = None,
        n_train: int = 1000,
        n_calibration: int = 1000,
        max_rows: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.label_column = label_column
        self.file_drift_positions = list(drift_positions or [])
        self.n_train = int(n_train)
        self.n_calibration = int(n_calibration)
        self.max_rows = max_rows
        self._build_stream()

    @classmethod
    def from_config(cls, dataset_cfg: dict[str, Any]) -> CSVStreamDataset:
        return cls(
            path=dataset_cfg["path"],
            label_column=dataset_cfg.get("label_column", -1),
            drift_positions=dataset_cfg.get("drift_positions"),
            n_train=dataset_cfg.get("n_train", 1000),
            n_calibration=dataset_cfg.get("n_calibration", 1000),
            max_rows=dataset_cfg.get("max_rows"),
        )

    def _build_stream(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"CSV stream not found: {self.path} (see scripts/download_*.sh)"
            )
        df = pd.read_csv(self.path, nrows=self.max_rows)
        label = (
            df.columns[self.label_column]
            if isinstance(self.label_column, int)
            else self.label_column
        )
        y_raw = df[label].to_numpy()
        X = df.drop(columns=[label]).to_numpy(dtype=float)
        classes, y = np.unique(y_raw, return_inverse=True)
        self.classes = classes
        self.num_classes = len(classes)

        offset = self.n_train + self.n_calibration
        if offset >= len(y):
            raise ValueError("n_train + n_calibration leaves no rows for the stream")
        self.X_train, self.y_train = X[: self.n_train], y[: self.n_train]
        if self.n_calibration > 0:
            self.X_cal = X[self.n_train : offset]
            self.y_cal = y[self.n_train : offset]
        self.X_stream, self.y_stream = X[offset:], y[offset:]
        n_stream = len(self.y_stream)
        self.drift_positions = [
            p - offset for p in self.file_drift_positions if 0 < p - offset < n_stream
        ]
        self.phase_idx = np.searchsorted(
            np.asarray(self.drift_positions), np.arange(n_stream), side="right"
        )

    def _metadata_for_index(self, idx: int) -> dict[str, Any]:
        return {"phase": int(self.phase_idx[idx])}
