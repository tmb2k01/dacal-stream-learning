from __future__ import annotations

from pathlib import Path
from typing import Any

from models import LightningSimpleCNNPredictor, SklearnClassifierPredictor


class ModelFactory:
    @staticmethod
    def create(config: dict[str, Any]):
        model_cfg = config["model"]
        model_name = model_cfg["name"]

        if model_name == "simple_cnn":
            predictor = LightningSimpleCNNPredictor.from_config(model_cfg)
            ModelFactory._maybe_prepare_simple_cnn(predictor, model_cfg)
            return predictor

        if model_name == "sgd_classifier":
            return SklearnClassifierPredictor.from_config(model_cfg)

        raise ValueError(f"Unsupported model: model.name={model_name!r}")

    @staticmethod
    def _maybe_prepare_simple_cnn(
        predictor: LightningSimpleCNNPredictor,
        model_cfg: dict[str, Any],
    ) -> None:
        checkpoint_path = model_cfg.get("checkpoint_path")
        force_retrain = model_cfg.get("force_retrain", False)

        if checkpoint_path and Path(checkpoint_path).exists() and not force_retrain:
            predictor.load(checkpoint_path)
            return

        training_cfg = model_cfg.get("training", {})
        if not training_cfg.get("auto_train", False):
            return

        predictor.fit_clean_cifar10(
            root=training_cfg.get("clean_root", "data"),
            epochs=training_cfg.get("epochs", 10),
            batch_size=training_cfg.get("batch_size", 128),
            num_workers=training_cfg.get("num_workers", 0),
            seed=training_cfg.get("seed", 42),
            calibration_size=training_cfg.get("calibration_size", 5000),
            accelerator=training_cfg.get("accelerator", "auto"),
            devices=training_cfg.get("devices", "auto"),
        )
        if checkpoint_path:
            predictor.save(checkpoint_path)
