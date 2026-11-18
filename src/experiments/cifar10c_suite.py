"""Shared CIFAR-10-C benchmark definitions: model, stream, and the 11 compared configurations.

Image pools (all indices refer to the 10,000 CIFAR-10 *test* images, which CIFAR-10-C corrupts):

- ``[0, BENCH_POOL_END)``      : benchmark streams (one disjoint 3-segment window per seed)
- ``[BENCH_POOL_END, 10_000)`` : validation streams used only to tune online-update options

The model is trained on the CIFAR-10 *train* split only, so train and stream images never overlap.
Inside a stream, every segment takes its own ``start`` offset, so no underlying image is shown twice.
"""

from __future__ import annotations

from typing import Any

CHECKPOINT_PATH = "artifacts/models/simplecnn_cifar10_aug_seed42_ep20.pt"
BENCH_POOL_END = 8000
TEST_POOL_SIZE = 10_000
DATA_DIR = "data/CIFAR-10-C"
CLEAN_ROOT = "data"

# Online-update options of SimpleCNN, chosen on the validation stream
# (artifacts/benchmarks/online_update_tuning_stage2.csv): Adam 3e-4 on the two dense layers,
# dropout off, no replay, no anchors. Full-network single-sample updates lost to the frozen model.
ONLINE_DEFAULTS: dict[str, Any] = {
    "reset_mode": "none",
    "online_optimizer": "adam",
    "online_lr": 3e-4,
    "online_dropout": False,
    "online_steps": 1,
    "online_replay_size": 0,
    "online_replay_batch": 8,
    "online_trainable": "fc",
    "online_anchor_batch": 0,
    "online_anchor_size": 0,
}

# Segment order of the stream: (display name, CIFAR-10-C corruption).
SEGMENT_PLAN = [
    ("clean", "clean"),
    ("gaussian-noise", "gaussian_noise"),
    ("shot-noise", "shot_noise"),
]


def model_config(
    epochs: int = 20,
    seed: int = 42,
    force_retrain: bool = False,
    online: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": "simple_cnn",
        "num_classes": 10,
        "device": "auto",
        "checkpoint_path": CHECKPOINT_PATH,
        "force_retrain": force_retrain,
        "dropout": 0.4,
        "training": {
            "auto_train": True,
            "clean_root": CLEAN_ROOT,
            "seed": seed,
            "epochs": epochs,
            "batch_size": 128,
            "calibration_size": 5000,
            "num_workers": 0,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine_annealing",
            "augmentation": {
                "random_crop_padding": 4,
                "horizontal_flip": True,
                "color_jitter": 0.2,
                "random_erasing": 0.0,
            },
        },
        **{**ONLINE_DEFAULTS, **(online or {})},
    }


def stream_config(
    first_start: int,
    count_per_segment: int = 250,
    severity: int = 1,
    pool_end: int = BENCH_POOL_END,
) -> dict[str, Any]:
    """Three-segment stream where segment ``i`` reads images ``first_start + i*count`` onward."""
    last_stop = first_start + len(SEGMENT_PLAN) * count_per_segment
    if last_stop > pool_end:
        raise ValueError(f"stream window [{first_start}, {last_stop}) exceeds the pool end {pool_end}")
    segments = []
    for i, (name, corruption) in enumerate(SEGMENT_PLAN):
        seg: dict[str, Any] = {
            "name": name,
            "corruption": corruption,
            "count": count_per_segment,
            "start": first_start + i * count_per_segment,
        }
        if corruption != "clean":
            seg["severity"] = severity
        segments.append(seg)
    return {
        "name": "cifar10c_stream",
        "data_dir": DATA_DIR,
        "clean_root": CLEAN_ROOT,
        "segments": segments,
        "enforce_disjoint": True,
    }


def seed_window_start(seed_index: int, count_per_segment: int = 250) -> int:
    return seed_index * len(SEGMENT_PLAN) * count_per_segment


def conformal_config() -> dict[str, Any]:
    return {
        "name": "classification",
        "enabled": True,
        "alpha": 0.1,
        "window_size": 200,
        "min_online_samples": 20,
    }


def configurations(seed: int, rate: float = 0.05, burst_rate: float = 0.8) -> list[dict[str, Any]]:
    """The 11 compared configurations. ``group`` drives figure grouping (<= 5 curves per figure)."""
    al = {"enabled": True, "name": "random", "rate": rate, "burst_rate": burst_rate, "seed": seed}

    def det(name: str, group: str, kind: str, signal: str, detector: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "group": group,
            "supervised": kind == "Supervised",
            "signal": signal,
            "active_policy": al,
            "drift_detector": {"enabled": True, **detector},
        }

    return [
        {
            "name": "Lower Baseline (Frozen, No AL, No Drift)",
            "group": "reference",
            "supervised": False,
            "signal": "none",
            "active_policy": {"enabled": False},
            "drift_detector": {"enabled": False},
        },
        {
            "name": "Baseline + AL (No Drift Detector)",
            "group": "reference",
            "supervised": False,
            "signal": "none",
            "active_policy": al,
            "drift_detector": {"enabled": False},
        },
        det("Upper Bound: Oracle", "reference", "Privileged", "step", {"name": "oracle", "delay": 0}),
        det(
            "Supervised: Baseline Z-Score", "supervised", "Supervised", "error",
            {"name": "baseline", "warmup_size": 10, "threshold_std": 2.5, "consecutive": 2},
        ),
        det("Supervised: ADWIN", "supervised", "Supervised", "error", {"name": "adwin", "delta": 0.01}),
        det(
            "Supervised: DDM", "supervised", "Supervised", "error",
            {"name": "ddm", "warm_start": 10, "warning_threshold": 1.5, "drift_threshold": 2.5},
        ),
        det(
            "Supervised: Page-Hinkley", "supervised", "Supervised", "error",
            {"name": "page_hinkley", "delta": 0.005, "threshold": 5.0, "min_instances": 10},
        ),
        det(
            "Label-Free: KS Test (Confidence)", "label_free", "Label-Free", "confidence",
            {"name": "ks", "window_size": 50, "test_every": 20, "alpha": 0.05, "signal": "confidence", "seed": seed},
        ),
        det(
            "Label-Free: CUSUM (Confidence)", "label_free", "Label-Free", "confidence",
            {"name": "cusum", "k": 0.5, "h": 4.0, "block_size": 20, "signal": "confidence"},
        ),
        det(
            "Label-Free: MMD (Embedding)", "label_free", "Label-Free", "embedding",
            {
                "name": "mmd", "window_size": 50, "test_every": 20, "alpha": 0.05,
                "n_permutations": 50, "pca_components": 16, "seed": seed,
            },
        ),
        det(
            "Label-Free: Adaptive MMD", "label_free", "Label-Free", "embedding",
            {
                "name": "adaptive_mmd", "window_size": 50, "test_every": 20, "alpha": 0.05,
                "n_permutations": 50, "pca_components": 16, "rebuild_size": 100, "seed": seed,
            },
        ),
    ]


# Figures: <= 5 curves each. "Baseline + AL" is repeated in every group as the no-detector control.
FIGURE_GROUPS: list[tuple[str, str, list[str]]] = [
    (
        "reference",
        "Reference points: frozen floor, active learning only, oracle ceiling",
        [
            "Lower Baseline (Frozen, No AL, No Drift)",
            "Baseline + AL (No Drift Detector)",
            "Upper Bound: Oracle",
        ],
    ),
    (
        "supervised",
        "Supervised (error-based) detectors vs. the no-detector control",
        [
            "Baseline + AL (No Drift Detector)",
            "Supervised: Baseline Z-Score",
            "Supervised: ADWIN",
            "Supervised: DDM",
            "Supervised: Page-Hinkley",
        ],
    ),
    (
        "label_free",
        "Label-free (confidence / embedding) detectors vs. the no-detector control",
        [
            "Baseline + AL (No Drift Detector)",
            "Label-Free: KS Test (Confidence)",
            "Label-Free: CUSUM (Confidence)",
            "Label-Free: MMD (Embedding)",
            "Label-Free: Adaptive MMD",
        ],
    ),
]
