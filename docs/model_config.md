# Model Config

Model configuration lives in the top-level `model` section of each YAML file.
`ModelFactory` currently supports image classification with the CIFAR-10
`SimpleCNN` and vector classification with an incremental sklearn classifier.

## CIFAR-10 SimpleCNN

Use this model with `CIFAR10CStreamDataset`:

```yaml
model:
  name: simple_cnn
  num_classes: 10
  device: auto
  checkpoint_path: artifacts/models/simplecnn_cifar10_seed42_ep10.pt
  force_retrain: false
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2470, 0.2435, 0.2616]
  dropout: 0.4
  training:
    auto_train: true
    clean_root: data
    seed: 42
    epochs: 10
    batch_size: 128
    calibration_size: 5000
    num_workers: 0
    accelerator: auto
    devices: auto
    lr: 0.001
    weight_decay: 0.0001
    scheduler: cosine_annealing
```

This creates a `LightningSimpleCNNPredictor`, so training is done through a
Lightning `Trainer` while online prediction/update still uses the repository's
`BasePredictor` interface. These defaults match the CIFAR playground notebooks:

`epochs`: `10`.

`batch_size`: `128`.

`calibration_size`: number of clean CIFAR-10 train samples held out before
training. These samples are not used for model training; they are used after
training to calibrate the conformal predictor.

`optimizer`: Adam with `lr: 0.001` and `weight_decay: 0.0001`.

`scheduler`: cosine annealing with `T_max = epochs`.

`accelerator` and `devices`: passed to `Trainer`. The default `auto` lets
Lightning select CPU/GPU automatically.

`transforms`: random crop with padding `4`, random horizontal flip, tensor
conversion, and CIFAR-10 normalization for training. Prediction uses tensor
conversion and the same normalization.

`checkpoint_path`: if this file exists and `force_retrain` is `false`, it is
loaded. If it does not exist and `training.auto_train` is `true`, the model is
trained on clean CIFAR-10 train data from `training.clean_root`, then saved.

`device`: `auto` selects CUDA when available, otherwise CPU.

## Synthetic Drift SGD Classifier

Use this model with `SyntheticDriftStreamDataset`:

```yaml
model:
  name: sgd_classifier
  classes: [0, 1]
  loss: log_loss
  alpha: 0.0001
  random_state: 42
```

The stream engine calls `fit_initial(X_train, y_train)` when a dataset exposes
initial training data. For `SyntheticDriftStreamDataset`, that is the first
phase of the stream.

## Conformal Prediction

The stream simulation uses split conformal classification sets when the
top-level `conformal` section is enabled:

```yaml
conformal:
  name: classification
  enabled: true
  alpha: 0.1
```

`alpha`: target miscoverage rate. For example, `alpha: 0.1` targets 90 percent
coverage.

For CIFAR-10, calibration uses the clean training samples reserved by
`model.training.calibration_size`.

## Running A Simulation

```bash
PYTHONPATH=src .venv/bin/python src/main.py \
  --config configs/cifar10c_stream.yaml \
  --max-steps 20
```

For CIFAR-10-C, make sure `data/CIFAR-10-C` contains the extracted corruption
`.npy` files before running the stream.
