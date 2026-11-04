from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models import RiverClassifierPredictor, SimpleCNN, SklearnClassifierPredictor


class ModelFactory:
    @staticmethod
    def create(config: dict[str, Any]):
        model_cfg = config["model"]
        model_name = model_cfg["name"]

        if model_name == "simple_cnn":
            predictor = SimpleCNN.from_config(model_cfg)
            ModelFactory._maybe_prepare_simple_cnn(predictor, model_cfg)
            return predictor

        if model_name == "sgd_classifier":
            return SklearnClassifierPredictor.from_config(model_cfg)

        if model_name in {"river_classifier", "river_softmax"}:
            return RiverClassifierPredictor.from_config(model_cfg)

        raise ValueError(f"Unsupported model: model.name={model_name!r}")

    @staticmethod
    def _maybe_prepare_simple_cnn(
        predictor: SimpleCNN,
        model_cfg: dict[str, Any],
    ) -> None:
        checkpoint_path = model_cfg.get("checkpoint_path")
        force_retrain = model_cfg.get("force_retrain", False)

        if checkpoint_path and Path(checkpoint_path).exists() and not force_retrain:
            predictor.load(checkpoint_path)
            predictor.snapshot_initial_state()
            ModelFactory._maybe_set_anchor(predictor, model_cfg)
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
        predictor.snapshot_initial_state()
        ModelFactory._maybe_set_anchor(predictor, model_cfg)

    @staticmethod
    def _maybe_set_anchor(predictor: SimpleCNN, model_cfg: dict[str, Any]) -> None:
        """Give the model clean *training* images to mix into online updates (never stream data)."""
        size = int(model_cfg.get("online_anchor_size", 0))
        if size <= 0 or model_cfg.get("online_anchor_batch", 0) <= 0:
            return
        training_cfg = model_cfg.get("training", {})
        subset = SimpleCNN._clean_cifar10_subset(
            root=training_cfg.get("clean_root", "data"),
            seed=training_cfg.get("seed", 42),
            calibration_size=training_cfg.get("calibration_size", 5000),
            split="train",
            transform=None,
        )
        picks = subset.indices[:size]
        data = subset.dataset
        x = np.stack([np.asarray(data.data[i]) for i in picks]).astype("float32")
        y = np.asarray([data.targets[i] for i in picks])
        predictor.set_anchor_data(x, y)
