from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from conformal import ClassificationConformalPredictor
from engine.stream_engine import StreamSimulationEngine
from factories.dataset_factory import DatasetFactory
from factories.model_factory import ModelFactory


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open() as config_file:
        return yaml.safe_load(config_file)


def build_simulation(config: dict[str, Any]) -> tuple[StreamSimulationEngine, Any]:
    dataset = DatasetFactory.create(config)
    predictor = ModelFactory.create(config)
    calibrator = build_conformal_predictor(config, predictor)

    engine = StreamSimulationEngine.from_components(
        dataset=dataset,
        predictor=predictor,
        calibrator=calibrator,
    )
    return engine, dataset


def build_simulation_from_yaml(config_path: str | Path) -> tuple[StreamSimulationEngine, Any]:
    return build_simulation(load_config(config_path))


def build_conformal_predictor(config: dict[str, Any], predictor):
    conformal_cfg = config.get("conformal", {})
    if not conformal_cfg.get("enabled", True):
        return None
    if conformal_cfg.get("name", "classification") != "classification":
        raise ValueError(f"Unsupported conformal predictor: {conformal_cfg.get('name')!r}")

    model_cfg = config["model"]
    training_cfg = model_cfg.get("training", {})
    calibrator = ClassificationConformalPredictor(
        alpha=conformal_cfg.get("alpha", 0.1),
        num_classes=model_cfg.get("num_classes", 10),
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

    if hasattr(predictor, "fit_initial"):
        return calibrator

    raise ValueError("No calibration source is available for the configured predictor")
