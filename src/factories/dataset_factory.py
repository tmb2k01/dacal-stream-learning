from __future__ import annotations

from typing import Any

from data.synthetic_drift_dataset import SyntheticDriftStreamDataset


class DatasetFactory:
    @staticmethod
    def create(config: dict[str, Any]):
        dataset_cfg = config["dataset"]
        dataset_name = dataset_cfg.get("name")

        if dataset_name == "synthetic_drift":
            return SyntheticDriftStreamDataset(
                seed=dataset_cfg.get("seed", 42),
                n_per_phase=dataset_cfg.get("n_per_phase", 1_000),
                n_phases=dataset_cfg.get("n_phases", 4),
                n_features=dataset_cfg.get("n_features", 10),
                noise_std=dataset_cfg.get("noise_std", 0.6),
                drift_angle=dataset_cfg.get("drift_angle", 1.5707963267948966),
            )

        task_type = config["task"]["type"]

        # if task_type == "classification":
        #     return ClassificationStreamDataset(
        #         data_path=dataset_cfg["data_path"],
        #         feature_columns=dataset_cfg.get("feature_columns"),
        #         target_column=dataset_cfg["target_column"],
        #         stream_order=dataset_cfg.get("stream_order", "sequential"),
        #     )

        # if task_type == "regression":
        #     return RegressionStreamDataset(
        #         data_path=dataset_cfg["data_path"],
        #         feature_columns=dataset_cfg.get("feature_columns"),
        #         target_column=dataset_cfg["target_column"],
        #         stream_order=dataset_cfg.get("stream_order", "sequential"),
        #     )

        # if task_type == "timeseries":
        #     return TimeSeriesStreamDataset(
        #         data_path=dataset_cfg["data_path"],
        #         target_column=dataset_cfg["target_column"],
        #         lookback=dataset_cfg["lookback"],
        #         horizon=dataset_cfg["horizon"],
        #         time_column=dataset_cfg.get("time_column"),
        #         feature_columns=dataset_cfg.get("feature_columns"),
        #         stream_order=dataset_cfg.get("stream_order", "sequential"),
        #     )

        raise ValueError(
            "Unsupported dataset configuration. "
            f"dataset.name={dataset_name!r}, task.type={task_type!r}"
        )
