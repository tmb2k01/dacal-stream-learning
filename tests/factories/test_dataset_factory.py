from __future__ import annotations

import pytest

from data import SyntheticDriftStreamDataset
from factories.dataset_factory import DatasetFactory


def _base_config() -> dict:
    return {
        "dataset": {"name": "synthetic_drift"},
        "task": {"type": "classification"},
    }


def test_create_returns_synthetic_dataset_with_defaults() -> None:
    dataset = DatasetFactory.create(_base_config())

    assert isinstance(dataset, SyntheticDriftStreamDataset)
    assert dataset.seed == 42
    assert dataset.n_per_phase == 1_000
    assert dataset.n_phases == 4
    assert dataset.n_features == 10
    assert dataset.noise_std == 0.6
    assert dataset.drift_angle == pytest.approx(1.5707963267948966)


def test_create_respects_synthetic_overrides() -> None:
    config = _base_config()
    config["dataset"] = {
        "name": "synthetic_drift",
        "seed": 9,
        "n_per_phase": 30,
        "n_phases": 5,
        "n_features": 6,
        "noise_std": 0.2,
        "drift_angle": 0.8,
    }

    dataset = DatasetFactory.create(config)

    assert isinstance(dataset, SyntheticDriftStreamDataset)
    assert dataset.seed == 9
    assert dataset.n_per_phase == 30
    assert dataset.n_phases == 5
    assert dataset.n_features == 6
    assert dataset.noise_std == 0.2
    assert dataset.drift_angle == 0.8


def test_create_raises_for_unsupported_dataset_name() -> None:
    config = {
        "dataset": {"name": "unknown_dataset"},
        "task": {"type": "classification"},
    }

    with pytest.raises(ValueError, match="Unsupported dataset configuration"):
        DatasetFactory.create(config)


def test_create_raises_keyerror_for_missing_dataset_section() -> None:
    config = {"task": {"type": "classification"}}

    with pytest.raises(KeyError, match="dataset"):
        DatasetFactory.create(config)


def test_create_raises_keyerror_for_missing_task_type_on_non_synthetic() -> None:
    config = {
        "dataset": {"name": "unknown_dataset"},
        "task": {},
    }

    with pytest.raises(KeyError, match="type"):
        DatasetFactory.create(config)

