import sys
from pathlib import Path

import numpy as np

# Ensure src is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.synthetic_drift_dataset import SyntheticDriftStreamDataset
from factories.dataset_factory import DatasetFactory


def test_synthetic_drift_dataset_is_reproducible_with_same_seed():
    ds1 = SyntheticDriftStreamDataset(seed=123, n_per_phase=20, n_phases=3, n_features=4)
    ds2 = SyntheticDriftStreamDataset(seed=123, n_per_phase=20, n_phases=3, n_features=4)

    assert np.allclose(ds1.X_stream, ds2.X_stream)
    assert np.array_equal(ds1.y_stream, ds2.y_stream)
    assert np.array_equal(ds1.phase_idx, ds2.phase_idx)
    assert ds1.drift_positions == ds2.drift_positions


def test_synthetic_drift_dataset_has_expected_drift_positions():
    dataset = SyntheticDriftStreamDataset(seed=7, n_per_phase=12, n_phases=4, n_features=5)
    assert dataset.drift_positions == [12, 24, 36]


def test_synthetic_drift_dataset_iterator_schema_and_length():
    dataset = SyntheticDriftStreamDataset(seed=1, n_per_phase=10, n_phases=3, n_features=3)
    items = list(dataset)

    assert len(items) == 30
    assert len(dataset) == 30
    assert set(items[0]) == {"x", "y", "metadata"}
    assert items[10]["metadata"]["is_drift_point"] is True
    assert items[11]["metadata"]["is_drift_point"] is False


def test_synthetic_drift_dataset_query_uses_item_index_metadata():
    dataset = SyntheticDriftStreamDataset(seed=5, n_per_phase=14, n_phases=3, n_features=4)
    sample = next(iter(dataset))

    queried = dataset.query(sample)

    assert queried == int(dataset.y_stream[0])


def test_dataset_factory_creates_synthetic_drift_dataset():
    config = {
        "task": {"type": "classification"},
        "dataset": {
            "name": "synthetic_drift",
            "seed": 99,
            "n_per_phase": 16,
            "n_phases": 5,
            "n_features": 6,
            "noise_std": 0.2,
        },
    }

    dataset = DatasetFactory.create(config)

    assert isinstance(dataset, SyntheticDriftStreamDataset)
    assert dataset.seed == 99
    assert dataset.n_per_phase == 16
    assert dataset.n_phases == 5
    assert dataset.n_features == 6
    assert dataset.noise_std == 0.2
