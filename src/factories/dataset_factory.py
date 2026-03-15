from __future__ import annotations

from typing import Any



class DatasetFactory:
    @staticmethod
    def create(config: dict[str, Any]):
        dataset_cfg = config["dataset"]
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

        raise ValueError(f"Unsupported task type for dataset creation: {task_type}")