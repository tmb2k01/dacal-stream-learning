"""Train the augmented clean-CIFAR-10 SimpleCNN checkpoint used by the benchmark.

Training uses only the CIFAR-10 *train* split minus a held-out calibration subset.
The CIFAR-10 test split (and therefore CIFAR-10-C) is never touched during training.

Usage:
    PYTHONPATH=src python -m experiments.train_checkpoint --epochs 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from experiments.cifar10c_suite import model_config
from factories.model_factory import ModelFactory


def clean_test_accuracy(model, root: str = "data") -> float:
    test = CIFAR10(root=root, train=False, download=False)
    out = model.predict_batched(np.asarray(test.data).astype("float32"))
    return float((out["probs"].argmax(axis=1) == np.asarray(test.targets)).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")
    args = parser.parse_args()

    cfg = {"model": model_config(epochs=args.epochs, seed=args.seed, force_retrain=args.force)}
    start = time.perf_counter()
    model = ModelFactory.create(cfg)
    print(f"checkpoint ready in {time.perf_counter() - start:.0f}s: {cfg['model']['checkpoint_path']}")
    print(f"augmentation: {model.augmentation}")
    print(f"clean CIFAR-10 test accuracy: {clean_test_accuracy(model):.4f}")
    print(f"device: {next(model.parameters()).device}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
