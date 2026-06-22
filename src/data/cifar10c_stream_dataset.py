from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import CIFAR10

from data.stream_dataset import BaseStreamDataset


@dataclass(frozen=True)
class CIFAR10CSegment:
    name: str
    corruption: str
    count: int
    severity: int
    start: int


class CIFAR10CStreamDataset(BaseStreamDataset):
    """Ordered CIFAR-10/CIFAR-10-C stream made from configurable corruption segments."""

    dataset_name = "cifar10c_stream"

    def __init__(
        self,
        data_dir: str | Path,
        segments: list[dict[str, Any]],
        labels_file: str = "labels.npy",
        default_severity: int = 1,
        clean_root: str | Path | None = None,
        clean_data_file: str | Path | None = None,
        clean_labels_file: str | Path | None = None,
        channel_first: bool = False,
        normalize: bool = False,
        dtype: str = "float32",
    ):
        if not segments:
            raise ValueError("segments must contain at least one stream segment")

        self.data_dir = Path(data_dir)
        self.labels_file = labels_file
        self.default_severity = default_severity
        self.clean_root = Path(clean_root) if clean_root is not None else None
        self.clean_data_file = Path(clean_data_file) if clean_data_file is not None else None
        self.clean_labels_file = Path(clean_labels_file) if clean_labels_file is not None else None
        self.channel_first = channel_first
        self.normalize = normalize
        self.dtype = dtype

        self.segments = [self._parse_segment(segment, idx) for idx, segment in enumerate(segments)]

        self.X_stream: np.ndarray
        self.y_stream: np.ndarray
        self.segment_idx: np.ndarray
        self.source_indices: np.ndarray
        self.drift_positions: list[int]
        self._segment_metadata: list[dict[str, Any]]
        self._build_stream()

    @classmethod
    def from_config(cls, dataset_cfg: dict[str, Any]) -> CIFAR10CStreamDataset:
        return cls(
            data_dir=dataset_cfg.get("data_dir", "data/CIFAR-10-C"),
            labels_file=dataset_cfg.get("labels_file", "labels.npy"),
            default_severity=dataset_cfg.get("severity", 1),
            clean_root=dataset_cfg.get("clean_root"),
            clean_data_file=dataset_cfg.get("clean_data_file"),
            clean_labels_file=dataset_cfg.get("clean_labels_file"),
            channel_first=dataset_cfg.get("channel_first", False),
            normalize=dataset_cfg.get("normalize", False),
            dtype=dataset_cfg.get("dtype", "float32"),
            segments=dataset_cfg["segments"],
        )

    def _parse_segment(self, segment: dict[str, Any], idx: int) -> CIFAR10CSegment:
        corruption = segment.get("corruption", segment.get("name"))
        if corruption is None:
            raise ValueError(f"segments[{idx}] must define 'corruption' or 'name'")

        count = int(segment["count"])
        severity = int(segment.get("severity", self.default_severity))
        start = int(segment.get("start", 0))
        name = segment.get("name", corruption)

        if count <= 0:
            raise ValueError(f"segments[{idx}].count must be positive")
        if not 1 <= severity <= 5:
            raise ValueError(f"segments[{idx}].severity must be between 1 and 5")
        if start < 0:
            raise ValueError(f"segments[{idx}].start must be non-negative")

        return CIFAR10CSegment(
            name=str(name),
            corruption=str(corruption),
            count=count,
            severity=severity,
            start=start,
        )

    def _build_stream(self) -> None:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        segment_parts: list[np.ndarray] = []
        source_index_parts: list[np.ndarray] = []
        self._segment_metadata = []

        offset = 0
        self.drift_positions = []
        for segment_idx, segment in enumerate(self.segments):
            x_segment, y_segment, source_indices = self._load_segment(segment)
            x_parts.append(self._format_images(x_segment))
            y_parts.append(y_segment.astype(int, copy=False))
            segment_parts.append(np.full(segment.count, segment_idx, dtype=int))
            source_index_parts.append(source_indices)
            self._segment_metadata.append(
                {
                    "name": segment.name,
                    "corruption": segment.corruption,
                    "severity": segment.severity,
                    "start": segment.start,
                    "count": segment.count,
                }
            )

            if segment_idx > 0:
                self.drift_positions.append(offset)
            offset += segment.count

        self.X_stream = np.concatenate(x_parts, axis=0)
        self.y_stream = np.concatenate(y_parts, axis=0)
        self.segment_idx = np.concatenate(segment_parts, axis=0)
        self.source_indices = np.concatenate(source_index_parts, axis=0)

    def _load_segment(self, segment: CIFAR10CSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if segment.corruption in {"clean", "clear"}:
            X, y = self._load_clean_data()
            source_indices = self._segment_indices(segment.start, segment.count, len(y))
            return X[source_indices], y[source_indices], source_indices

        x_path = self.data_dir / f"{segment.corruption}.npy"
        y_path = self.data_dir / self.labels_file
        if not x_path.exists():
            raise FileNotFoundError(f"CIFAR-10-C corruption file not found: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"CIFAR-10-C labels file not found: {y_path}")

        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        severity_start = (segment.severity - 1) * 10_000
        severity_stop = segment.severity * 10_000
        if X.shape[0] < severity_stop:
            raise ValueError(
                f"{x_path} does not contain severity {segment.severity}; "
                f"expected at least {severity_stop} images, found {X.shape[0]}"
            )

        labels = y[severity_start:severity_stop] if y.shape[0] >= severity_stop else y[:10_000]
        source_indices = self._segment_indices(segment.start, segment.count, len(labels))
        image_indices = source_indices + severity_start
        return X[image_indices], labels[source_indices], image_indices

    def _load_clean_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.clean_data_file is not None:
            X = np.load(self.clean_data_file, mmap_mode="r")
            if self.clean_labels_file is None:
                raise ValueError("clean_labels_file must be set when clean_data_file is used")
            y = np.load(self.clean_labels_file, mmap_mode="r")
            return X, y

        root = self.clean_root if self.clean_root is not None else self.data_dir.parent
        try:
            clean_dataset = CIFAR10(root=str(root), train=False, download=False)
        except RuntimeError:
            clean_dataset = CIFAR10(root=str(root), train=False, download=True)
        return np.asarray(clean_dataset.data), np.asarray(clean_dataset.targets)

    @staticmethod
    def _segment_indices(start: int, count: int, available: int) -> np.ndarray:
        stop = start + count
        if stop > available:
            raise ValueError(
                f"Requested segment slice [{start}:{stop}], but only {available} samples exist"
            )
        return np.arange(start, stop, dtype=int)

    def _format_images(self, images: np.ndarray) -> np.ndarray:
        X = np.asarray(images)
        if self.normalize:
            X = X.astype(self.dtype, copy=False) / 255.0
        else:
            X = X.astype(self.dtype, copy=False)

        if self.channel_first:
            X = np.transpose(X, (0, 3, 1, 2))
        return X

    def _metadata_for_index(self, idx: int) -> dict[str, Any]:
        segment_i = int(self.segment_idx[idx])
        segment_metadata = self._segment_metadata[segment_i]
        return {
            "segment": segment_i,
            "segment_name": segment_metadata["name"],
            "corruption": segment_metadata["corruption"],
            "severity": segment_metadata["severity"],
            "source_index": int(self.source_indices[idx]),
        }
