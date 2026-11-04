from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

from conformal import (
    APSConformalPredictor,
    ClassificationConformalPredictor,
    MondrianConformalPredictor,
    TopKConformalPredictor,
)
from conformal.localizer_calibrator import ConformalLocalizerCalibrator

# Registry of probability-based set predictors (all share the sliding-window API).
SET_PREDICTORS: dict[str, type[ClassificationConformalPredictor]] = {
    "classification": ClassificationConformalPredictor,
    "lac": ClassificationConformalPredictor,
    "mondrian": MondrianConformalPredictor,
    "aps": APSConformalPredictor,
    "topk": TopKConformalPredictor,
}


class CalibratorFactory:
    @staticmethod
    def create_set_predictor(
        conformal_cfg: dict[str, Any],
        num_classes: int,
    ) -> ClassificationConformalPredictor:
        """Build an (uncalibrated) LAC / Mondrian / APS / Top-K set predictor."""
        name = conformal_cfg.get("name", "classification")
        if name not in SET_PREDICTORS:
            raise ValueError(
                f"Unsupported conformal predictor: {name!r}. "
                f"Expected one of {sorted(SET_PREDICTORS)}"
            )
        kwargs: dict[str, Any] = {
            "alpha": conformal_cfg.get("alpha", 0.1),
            "num_classes": num_classes,
            "window_size": conformal_cfg.get("window_size", None),
            "min_online_samples": conformal_cfg.get("min_online_samples", 1),
            "min_samples_per_class": conformal_cfg.get("min_samples_per_class", 0),
        }
        if name == "mondrian":
            kwargs["min_class_samples"] = conformal_cfg.get("min_class_samples")
        return SET_PREDICTORS[name](**kwargs)

    @staticmethod
    def create(config: dict[str, Any], task=None):
        calibrator_cfg = config.get("calibrator") or config.get("conformal") or {}
        task_type = config.get("task", {}).get("type", "classification")
        name = calibrator_cfg.get("name", "classification")

        if name in SET_PREDICTORS:
            num_classes = int(
                calibrator_cfg.get("num_classes", config.get("model", {}).get("num_classes", 10))
            )
            return CalibratorFactory.create_set_predictor(calibrator_cfg, num_classes)

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
