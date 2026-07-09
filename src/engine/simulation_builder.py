from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from conformal import ClassificationConformalPredictor
from engine.stream_engine import StreamSimulationEngine
from factories.dataset_factory import DatasetFactory
from factories.detector_factory import DriftDetectorFactory
from factories.model_factory import ModelFactory
from factories.policy_factory import PolicyFactory


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open() as config_file:
        return yaml.safe_load(config_file)


def build_simulation(config: dict[str, Any]) -> tuple[StreamSimulationEngine, Any]:
    dataset = DatasetFactory.create(config)
    predictor = ModelFactory.create(config)
    fit_initial_model(predictor, dataset)
    calibrator = build_conformal_predictor(config, predictor, dataset)
    active_policy = PolicyFactory.create(config)
    drift_detector = DriftDetectorFactory.create(config)

    engine = StreamSimulationEngine(
        predictor=predictor,
        calibrator=calibrator,
        active_policy=active_policy,
        drift_detector=drift_detector,
        label_oracle=dataset,
    )
    return engine, dataset


def build_simulation_from_yaml(config_path: str | Path) -> tuple[StreamSimulationEngine, Any]:
    return build_simulation(load_config(config_path))


def fit_initial_model(predictor: Any, dataset: Any) -> None:
    if hasattr(predictor, "fit_initial") and hasattr(dataset, "X_train"):
        predictor.fit_initial(dataset.X_train, dataset.y_train)


def build_conformal_predictor(config: dict[str, Any], predictor: Any, dataset: Any):
    conformal_cfg = config.get("conformal", {})
    if not conformal_cfg.get("enabled", True):
        return None
    if conformal_cfg.get("name", "classification") != "classification":
        raise ValueError(f"Unsupported conformal predictor: {conformal_cfg.get('name')!r}")

    model_cfg = config["model"]
    training_cfg = model_cfg.get("training", {})
    calibrator = ClassificationConformalPredictor(
        alpha=conformal_cfg.get("alpha", 0.1),
        num_classes=_num_classes(model_cfg, dataset),
    )

    if hasattr(predictor, "clean_cifar10_calibration_loader"):
        loader = predictor.clean_cifar10_calibration_loader(
            root=training_cfg.get("clean_root", "data"),
            calibration_size=training_cfg.get(
                "calibration_size",
                conformal_cfg.get("calibration_size", 5000),
            ),
            batch_size=training_cfg.get("batch_size", 128),
            num_workers=training_cfg.get("num_workers", 0),
            seed=training_cfg.get("seed", 42),
        )
        calibrator.calibrate_loader(predictor, loader)
        return calibrator

    if hasattr(dataset, "X_train") and hasattr(dataset, "y_train"):
        calibrator.calibrate_arrays(predictor, dataset.X_train, dataset.y_train)
        return calibrator

    raise ValueError("No calibration source is available for the configured predictor")


def _num_classes(model_cfg: dict[str, Any], dataset: Any) -> int:
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    if "classes" in model_cfg:
        return len(model_cfg["classes"])
    if hasattr(dataset, "y_train"):
        return len({int(y) for y in dataset.y_train})
    return 10
