from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class BaseStreamDataset(ABC):
    """Base class for stream datasets with labels and drift-position metadata."""

    dataset_name: str | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path):
        """Build a dataset from a YAML file containing a top-level dataset section."""
        with Path(config_path).open() as config_file:
            config = yaml.safe_load(config_file)

        dataset_cfg = config["dataset"]
        if cls.dataset_name is not None and dataset_cfg.get("name") != cls.dataset_name:
            raise ValueError(f"dataset.name must be {cls.dataset_name!r}")
        return cls.from_config(dataset_cfg)

    @classmethod
    def from_config(cls, dataset_cfg: dict[str, Any]):
        raise NotImplementedError(f"{cls.__name__}.from_config() is not implemented")

    @abstractmethod
    def _build_stream(self) -> None:
        """Populate X_stream, y_stream, and drift_positions."""
        pass

    def __len__(self) -> int:
        return int(self.X_stream.shape[0])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        drift_points = set(self.drift_positions)
        for idx, (x_i, y_i) in enumerate(zip(self.X_stream, self.y_stream, strict=False)):
            yield {
                "x": x_i,
                "y": self._format_label(y_i),
                "metadata": {
                    "index": idx,
                    **self._metadata_for_index(idx),
                    "is_drift_point": idx in drift_points,
                },
            }

    def _metadata_for_index(self, idx: int) -> dict[str, Any]:
        return {}

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

    @staticmethod
    def _format_label(y_value: Any) -> Any:
        if isinstance(y_value, (int, float, np.integer, np.floating, bool, np.bool_)):
            return int(y_value)
        return y_value
