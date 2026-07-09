from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

from conformal.localizer_calibrator import ConformalLocalizerCalibrator


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

        if name == "conformal_localizer":
            return ConformalLocalizerCalibrator(
                model=calibrator_cfg.get("model", RandomForestClassifier()),
                alpha=calibrator_cfg.get("alpha", 0.2),
                cv_params=calibrator_cfg.get("cv_params", {}),
                cv_runs=calibrator_cfg.get("cv_runs", 5),
                localizer=calibrator_cfg.get("localizer", KNeighborsRegressor(n_neighbors=1)),
                n_min_members=calibrator_cfg.get("n_min_members", 100),
                bootstrap_fraction=calibrator_cfg.get("bootstrap_fraction", 1.0),
                cover_search=calibrator_cfg.get("cover_search", 50),
                n_jobs=calibrator_cfg.get("n_jobs", -1),
                min_recalibrate_samples=calibrator_cfg.get("min_recalibrate_samples", 200),
            )

        raise ValueError(f"Unsupported calibrator '{name}' for task type '{task_type}'")
