from __future__ import annotations

from typing import Any


class CalibratorFactory:
    @staticmethod
    def create(config: dict[str, Any], task):
        calibrator_cfg = config["calibrator"]
        task_type = config["task"]["type"]
        name = calibrator_cfg["name"]

        calibrator_cfg.get("alpha", 0.1)
        calibrator_cfg.get("window_size", 200)
        calibrator_cfg.get("min_window_size", 50)
        calibrator_cfg.get("max_window_size", 500)
        calibrator_cfg.get("decay", 0.01)

        # if task_type == "classification":
        #     if name == "sliding_window":
        #         return SlidingWindowClassificationCalibrator(
        #             task=task,
        #             alpha=alpha,
        #             window_size=window_size,
        #         )

        #     if name == "adaptive_window":
        #         return AdaptiveWindowClassificationCalibrator(
        #             task=task,
        #             alpha=alpha,
        #             window_size=window_size,
        #             min_window_size=min_window_size,
        #             max_window_size=max_window_size,
        #         )

        #     if name == "weighted":
        #         return WeightedClassificationCalibrator(
        #             task=task,
        #             alpha=alpha,
        #             window_size=window_size,
        #             decay=decay,
        #         )

        # if task_type in {"regression", "timeseries"}:
        #     if name == "sliding_window":
        #         return SlidingWindowRegressionCalibrator(
        #             task=task,
        #             alpha=alpha,
        #             window_size=window_size,
        #         )

        #     if name == "adaptive_window":
        #         return AdaptiveWindowRegressionCalibrator(
        #             task=task,
        #             alpha=alpha,
        #             window_size=window_size,
        #             min_window_size=min_window_size,
        #             max_window_size=max_window_size,
        #         )

        #     if name == "weighted":
        #         return WeightedRegressionCalibrator(
        #             task=task,
        #             alpha=alpha,
        #             window_size=window_size,
        #             decay=decay,
        #         )

        raise ValueError(f"Unsupported calibrator '{name}' for task type '{task_type}'")
