from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from models import SimpleCNN


def _image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(32, 32, 3)).astype("float32")


def _weights(model: SimpleCNN) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.parameters()])


def test_update_online_changes_weights_and_lowers_loss_on_the_sample():
    torch.manual_seed(0)
    model = SimpleCNN(online_lr=1e-2, online_dropout=False)
    x, y = _image(1), 3
    before = _weights(model)
    p_before = model.predict_step_online(x)["probs"][0, y].item()
    for _ in range(5):
        model.update_online(x, y)
    p_after = model.predict_step_online(x)["probs"][0, y].item()
    assert not torch.equal(before, _weights(model))
    assert p_after > p_before


def test_update_online_keeps_batchnorm_statistics_when_frozen():
    model = SimpleCNN(online_lr=1e-2, freeze_bn_online=True)
    bn = model.features[1]
    before = bn.running_mean.clone()
    model.update_online(_image(2), 1)
    assert torch.equal(before, bn.running_mean)


def test_disabling_online_dropout_makes_updates_deterministic():
    x, y = _image(3), 2

    def run() -> torch.Tensor:
        torch.manual_seed(7)
        model = SimpleCNN(online_lr=1e-2, online_dropout=False)
        model.update_online(x, y)
        return _weights(model)

    assert torch.equal(run(), run())


def test_replay_buffer_is_bounded_and_used():
    model = SimpleCNN(online_lr=1e-3, online_replay_size=4, online_replay_batch=3)
    for i in range(10):
        model.update_online(_image(i), i % 10)
    assert len(model._replay) == 4
    x, y = model._with_replay(torch.zeros(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))
    assert x.shape[0] == 4 and y.shape[0] == 4  # new sample + 3 replayed


def test_restore_initial_reset_clears_replay_and_weights():
    model = SimpleCNN(online_lr=1e-2, reset_mode="restore_initial", online_replay_size=4)
    model.snapshot_initial_state()
    start = _weights(model)
    model.update_online(_image(0), 1)
    model.reset_adaptation_state()
    assert torch.equal(start, _weights(model))
    assert len(model._replay) == 0


def test_train_augmentation_changes_the_same_image_between_draws():
    model = SimpleCNN()
    transform = model._train_transform()
    image = Image.fromarray(_image(5).astype("uint8"))
    torch.manual_seed(0)
    draws = [transform(image) for _ in range(8)]
    assert all(d.shape == (3, 32, 32) for d in draws)
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_augmentation_can_be_overridden_and_disabled():
    model = SimpleCNN(
        augmentation={"random_crop_padding": 0, "horizontal_flip": False, "color_jitter": 0.0}
    )
    transform = model._train_transform()
    image = Image.fromarray(_image(6).astype("uint8"))
    assert torch.equal(transform(image), transform(image))


def test_head_only_updates_leave_the_feature_extractor_untouched():
    model = SimpleCNN(online_lr=1e-2, online_trainable="head", online_dropout=False)
    features_before = torch.cat([p.detach().flatten().clone() for p in model.features.parameters()])
    fc1_before = model.fc1.weight.detach().clone()
    head_before = model.fc2.weight.detach().clone()
    model.update_online(_image(4), 5)
    features_after = torch.cat([p.detach().flatten() for p in model.features.parameters()])
    assert torch.equal(features_before, features_after)
    assert torch.equal(fc1_before, model.fc1.weight)
    assert not torch.equal(head_before, model.fc2.weight)


def test_anchor_samples_are_mixed_into_the_update_batch():
    model = SimpleCNN(online_anchor_batch=4)
    anchor_x = np.stack([_image(i) for i in range(10)])
    model.set_anchor_data(anchor_x, np.arange(10))
    x, y = model._with_anchor(torch.zeros(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))
    assert x.shape[0] == 5 and y.shape[0] == 5
