from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from factories.calibrator_factory import CalibratorFactory, SET_PREDICTORS
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
    if hasattr(predictor, "snapshot_initial_state"):
        predictor.snapshot_initial_state()
    calibrator = build_conformal_predictor(config, predictor, dataset)
    active_policy = PolicyFactory.create(config)
    drift_detector = build_drift_detector(config, predictor, dataset, calibrator)

    engine = StreamSimulationEngine(
        predictor=predictor,
        calibrator=calibrator,
        active_policy=active_policy,
        drift_detector=drift_detector,
        label_oracle=dataset,
        **config.get("engine", {}),
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

    name = conformal_cfg.get("name", "classification")
    if name not in SET_PREDICTORS and name != "classification":
        raise ValueError(f"Unsupported conformal predictor: {name!r}")

    model_cfg = config["model"]
    training_cfg = model_cfg.get("training", {})
    calibrator = CalibratorFactory.create_set_predictor(
        conformal_cfg,
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

    if hasattr(dataset, "X_cal") and hasattr(dataset, "y_cal"):
        calibrator.calibrate_arrays(predictor, dataset.X_cal, dataset.y_cal)
        return calibrator

    if hasattr(dataset, "X_train") and hasattr(dataset, "y_train"):
        calibrator.calibrate_arrays(predictor, dataset.X_train, dataset.y_train)
        return calibrator

    raise ValueError("No calibration source is available for the configured predictor")


def build_drift_detector(
    config: dict[str, Any], predictor: Any, dataset: Any, calibrator: Any = None
):
    drift_positions = getattr(dataset, "drift_positions", None)
    drift_detector = DriftDetectorFactory.create(config, drift_positions=drift_positions)
    if drift_detector is None:
        return None

    if hasattr(drift_detector, "fit_reference"):
        reference_data = _collect_reference_data(config, predictor, dataset, calibrator)
        if reference_data is not None:
            drift_detector.fit_reference(reference_data)

    return drift_detector


def _collect_reference_data(
    config: dict[str, Any], predictor: Any, dataset: Any, calibrator: Any = None
) -> dict[str, np.ndarray] | None:
    # 1. Image / loader-based (CIFAR-10)
    if hasattr(predictor, "clean_cifar10_calibration_loader"):
        training_cfg = config.get("model", {}).get("training", {})
        conformal_cfg = config.get("conformal", {})
        cal_size = training_cfg.get("calibration_size", conformal_cfg.get("calibration_size", 2000))
        loader = predictor.clean_cifar10_calibration_loader(
            root=training_cfg.get("clean_root", "data"),
            calibration_size=cal_size,
            batch_size=training_cfg.get("batch_size", 128),
            num_workers=training_cfg.get("num_workers", 0),
            seed=training_cfg.get("seed", 42),
        )
        probs_list, emb_list = [], []
        for x_b, _ in loader:
            out = predictor.predict_step_online(x_b)
            p = out.get("probs")
            if p is not None:
                if hasattr(p, "detach"):
                    p = p.detach().cpu().numpy()
                probs_list.append(np.asarray(p))
            if "embedding" in out:
                e = out["embedding"]
                if hasattr(e, "detach"):
                    e = e.detach().cpu().numpy()
                emb_list.append(np.asarray(e))

        all_probs = np.concatenate(probs_list, axis=0) if probs_list else None
        all_embs = np.concatenate(emb_list, axis=0) if emb_list else None
        ref: dict[str, np.ndarray] = {}
        if all_probs is not None:
            ref["probs"] = all_probs
            ref["confidence"] = np.max(all_probs, axis=-1)
        if all_embs is not None:
            ref["embedding"] = all_embs
        return ref

    # 2. Array-based (Synthetic / Tabular)
    X_ref = None
    if hasattr(dataset, "X_cal"):
        X_ref = dataset.X_cal
    elif hasattr(dataset, "X_train"):
        X_ref = dataset.X_train

    if X_ref is not None:
        out = predictor.predict_step_online(X_ref)
        p = out.get("probs")
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        probs_arr = np.asarray(p) if p is not None else None
        emb = out.get("embedding")
        if emb is not None:
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            emb_arr = np.asarray(emb)
        else:
            emb_arr = np.asarray(X_ref)

        ref = {}
        if probs_arr is not None:
            ref["probs"] = probs_arr
            ref["confidence"] = np.max(probs_arr, axis=-1)
        ref["embedding"] = emb_arr
        return ref

    return None


def _num_classes(model_cfg: dict[str, Any], dataset: Any) -> int:
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    if "classes" in model_cfg:
        return len(model_cfg["classes"])
    if hasattr(dataset, "y_train"):
        return len({int(y) for y in dataset.y_train})
    return 10
