"""Verify that training images and streaming images are different, and that no stream image repeats.

Checks (exit code 1 if any fails):

1. Model training indices (CIFAR-10 train split) and conformal-calibration indices are disjoint.
2. Every benchmark stream window uses each underlying test image at most once, across all
   segments and across the 5 default seed windows.
3. Stream images come from the CIFAR-10 *test* split, whose indices cannot appear in the train split.
4. Pixel-level check: no stream clean image is byte-identical to any image the model trained on
   (CIFAR-10 contains a handful of exact train/test duplicates; they are reported and counted).
5. The validation pool used for tuning is disjoint from every benchmark window.

Usage:
    PYTHONPATH=src python -m experiments.verify_data_splits [--n-seeds 5] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

from data import CIFAR10CStreamDataset
from experiments.cifar10c_suite import (
    BENCH_POOL_END,
    CLEAN_ROOT,
    SEGMENT_PLAN,
    TEST_POOL_SIZE,
    seed_window_start,
    stream_config,
)
from models import SimpleCNN


def _hashes(images: np.ndarray) -> list[str]:
    return [hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest() for img in images]


def verify(n_seeds: int = 5, count: int = 250, seed: int = 42, calibration_size: int = 5000) -> dict:
    train_full = CIFAR10(root=CLEAN_ROOT, train=True, download=False)
    test = CIFAR10(root=CLEAN_ROOT, train=False, download=False)

    train_idx = np.array(SimpleCNN._clean_cifar10_subset(CLEAN_ROOT, seed, calibration_size, "train", None).indices)
    cal_idx = np.array(
        SimpleCNN._clean_cifar10_subset(CLEAN_ROOT, seed, calibration_size, "calibration", None).indices
    )

    report: dict = {
        "train_images": int(len(train_idx)),
        "calibration_images": int(len(cal_idx)),
        "train_calibration_overlap": int(np.intersect1d(train_idx, cal_idx).size),
    }

    train_hashes = set(_hashes(train_full.data[train_idx]))
    test_hashes = _hashes(test.data)
    report["train_test_exact_duplicates_in_whole_test_set"] = int(sum(h in train_hashes for h in test_hashes))

    seen_pool: set[int] = set()
    windows = []
    for k in range(n_seeds):
        start = seed_window_start(k, count)
        ds = CIFAR10CStreamDataset.from_config(stream_config(start, count))  # raises on overlap
        underlying = ds.underlying_indices
        unique = int(np.unique(underlying).size)

        clean_mask = ds.segment_idx == 0
        clean_idx = underlying[clean_mask]
        clean_pixels_match = bool(np.array_equal(ds.X_stream[clean_mask], test.data[clean_idx].astype("float32")))
        labels_match = bool(np.array_equal(ds.y_stream, np.asarray(test.targets)[underlying]))
        dup_with_train = int(sum(test_hashes[i] in train_hashes for i in underlying))

        shared_with_earlier = int(np.intersect1d(underlying, np.fromiter(seen_pool, dtype=int)).size)
        seen_pool.update(int(i) for i in underlying)
        windows.append(
            {
                "seed_index": k,
                "stream_length": int(len(ds)),
                "unique_underlying_images": unique,
                "all_unique": unique == len(ds),
                "clean_pixels_equal_test_split": clean_pixels_match,
                "labels_equal_test_targets": labels_match,
                "exact_duplicates_of_training_images": dup_with_train,
                "shared_with_earlier_windows": shared_with_earlier,
                "max_index": int(underlying.max()),
            }
        )
    report["windows"] = windows

    val_pool = set(range(BENCH_POOL_END, TEST_POOL_SIZE))
    report["validation_pool_overlap_with_benchmark_windows"] = int(len(val_pool & seen_pool))
    report["segments"] = [name for name, _ in SEGMENT_PLAN]

    report["passed"] = bool(
        report["train_calibration_overlap"] == 0
        and report["validation_pool_overlap_with_benchmark_windows"] == 0
        and all(
            w["all_unique"]
            and w["clean_pixels_equal_test_split"]
            and w["labels_equal_test_targets"]
            and w["shared_with_earlier_windows"] == 0
            and w["max_index"] < BENCH_POOL_END
            for w in windows
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", default="artifacts/benchmarks/data_split_verification.json")
    args = parser.parse_args()

    report = verify(n_seeds=args.n_seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nRESULT:", "PASS" if report["passed"] else "FAIL")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
