from __future__ import annotations

from typing import Any


class TaskFactory:
    @staticmethod
    def create(config: dict[str, Any]):
        task_cfg = config["task"]
        task_type = task_cfg["type"]

        # if task_type == "classification":
        #     return ClassificationTask(
        #         num_classes=task_cfg["num_classes"],
        #         class_names=task_cfg.get("class_names"),
        #     )

        # if task_type == "regression":
        #     return RegressionTask(
        #         target_dim=task_cfg.get("target_dim", 1),
        #     )

        # if task_type == "timeseries":
        #     return TimeSeriesTask(
        #         horizon=task_cfg["horizon"],
        #         target_dim=task_cfg.get("target_dim", 1),
        #         aggregation=task_cfg.get("aggregation", "mean"),
        #     )

        raise ValueError(f"Unsupported task type: {task_type}")
