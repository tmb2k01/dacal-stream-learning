import numpy as np

from data.synthetic_dataset import SyntheticDriftStreamDataset


def test_training_samples_are_disjoint_from_stream() -> None:
    dataset = SyntheticDriftStreamDataset(seed=42, n_per_phase=20, n_phases=3)

    assert dataset.X_train.shape == (20, 10)
    assert dataset.drift_positions == [20, 40]
    assert len(dataset) == 60
    assert not any(
        np.array_equal(train_sample, stream_sample)
        for train_sample in dataset.X_train
        for stream_sample in dataset.X_stream
    )
    same_seed = SyntheticDriftStreamDataset(seed=42, n_per_phase=20, n_phases=3)
    assert np.array_equal(dataset.X_train, same_seed.X_train)
