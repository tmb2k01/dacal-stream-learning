from __future__ import annotations

from data import CIFAR10CStreamDataset, SyntheticDriftStreamDataset


class DatasetFactory:
    @staticmethod
    def create(config: dict):
        dataset_cfg: dict = config.get("dataset")
        dataset_name = dataset_cfg.get("name")

        if dataset_name == "synthetic_drift":
            return SyntheticDriftStreamDataset.from_config(dataset_cfg)

        if dataset_name == "cifar10c_stream":
            return CIFAR10CStreamDataset.from_config(dataset_cfg)

        raise ValueError(f"Unsupported dataset configuration. dataset.name={dataset_name!r}")
