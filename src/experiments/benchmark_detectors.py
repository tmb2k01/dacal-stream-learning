"""CIFAR-10-C drift detector benchmark (11 configurations, several seeds, disjoint streams).

Stream: ``clean -> gaussian noise -> shot noise`` (250 images each). Every segment reads its own
slice of the CIFAR-10 test images, so no image is shown twice, and the model is trained on the
CIFAR-10 train split only. Each seed uses a different, non-overlapping stream window.

Active learning: labels are revealed only for queried items (``rate`` background, ``burst_rate``
after an alarm). Every revealed label triggers one online model update and one conformal
recalibration step. Detectors that read the prediction error only see queried items.

Usage:
    PYTHONPATH=src python -m experiments.benchmark_detectors --seeds 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.simulation_builder import build_simulation
from experiments.cifar10c_suite import (
    BENCH_POOL_END,
    SEGMENT_PLAN,
    configurations,
    conformal_config,
    model_config,
    seed_window_start,
    stream_config,
)

BASE_SEED = 42
DEFAULT_OUTPUT_DIR = "artifacts/benchmarks"
FILE_PREFIX = "cifar10c_v2"


def build_run_config(
    item: dict[str, Any],
    seed: int,
    first_start: int,
    count: int = 250,
    severity: int = 1,
    pool_end: int = BENCH_POOL_END,
    online: dict[str, Any] | None = None,
    detector_sees_all_labels: bool = False,
) -> dict[str, Any]:
    model_cfg = model_config(online={"online_seed": seed, **(online or {})})
    return {
        "dataset": stream_config(first_start, count, severity, pool_end),
        "model": model_cfg,
        "conformal": conformal_config(),
        "active_policy": item["active_policy"],
        "drift_detector": item["drift_detector"],
        "engine": {
            "drift_burst": 100,
            "burst_rate": 0.8,
            "seed": seed,
            "supervised_signals_require_query": not detector_sees_all_labels,
        },
    }


def _covered(record: dict[str, Any]) -> float:
    pred_set = record.get("conformal_output", {}).get("prediction_set")
    y_true = record.get("y_true")
    if pred_set is None or y_true is None:
        return float("nan")
    if hasattr(pred_set, "detach"):
        pred_set = pred_set.detach().cpu().numpy()
    return float(bool(pred_set[0, int(y_true)]))


def _set_size(record: dict[str, Any]) -> float:
    size = record.get("conformal_output", {}).get("set_size")
    if size is None:
        return float("nan")
    if hasattr(size, "detach"):
        size = size.detach().cpu().numpy()
    return float(np.asarray(size).reshape(-1)[0])


def summarize_records(
    records: list[dict[str, Any]], drift_positions: list[int], segment_len: int
) -> dict[str, Any]:
    """Turn raw per-step records into the benchmark metrics."""
    n = len(records)
    correct = np.array([float(r["correct"]) if r["correct"] is not None else np.nan for r in records])
    covered = np.array([_covered(r) for r in records])
    sizes = np.array([_set_size(r) for r in records])
    pre_end = drift_positions[0]

    alarms = [
        r["step"]
        for r in records
        if r.get("drift_event") is not None and getattr(r["drift_event"], "drift", False)
    ]

    # An alarm counts as detecting drift i only if it falls inside that drift's own segment.
    bounds = [*drift_positions, n]
    delays: list[float | None] = []
    for i, dp in enumerate(drift_positions):
        inside = [a for a in alarms if dp <= a < bounds[i + 1]]
        delays.append(float(inside[0] - dp) if inside else None)
    found = [d for d in delays if d is not None]

    seg_acc = {
        name: float(np.nanmean(correct[i * segment_len : (i + 1) * segment_len]))
        for i, (name, _) in enumerate(SEGMENT_PLAN)
    }
    return {
        "acc_overall": float(np.nanmean(correct)),
        "acc_pre": float(np.nanmean(correct[:pre_end])),
        "acc_post": float(np.nanmean(correct[pre_end:])),
        "acc_clean": seg_acc["clean"],
        "acc_gaussian": seg_acc["gaussian-noise"],
        "acc_shot": seg_acc["shot-noise"],
        "cov_overall": float(np.nanmean(covered)),
        "cov_pre": float(np.nanmean(covered[:pre_end])),
        "cov_post": float(np.nanmean(covered[pre_end:])),
        "set_size": float(np.nanmean(sizes)),
        "set_size_post": float(np.nanmean(sizes[pre_end:])),
        "total_alarms": len(alarms),
        "false_alarms": sum(1 for a in alarms if a < pre_end),
        "drifts_detected": len(found),
        "detection_rate": len(found) / len(drift_positions),
        "mean_delay": float(np.mean(found)) if found else float("nan"),
        "labels_used": int(sum(1 for r in records if r["queried"])),
        "alarm_steps": alarms,
    }


def step_series(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Per-step arrays used by the trajectory figures."""
    return {
        "correct": np.array([float(r["correct"]) if r["correct"] is not None else np.nan for r in records]),
        "covered": np.array([_covered(r) for r in records]),
        "set_size": np.array([_set_size(r) for r in records]),
        "queried": np.array([float(r["queried"]) for r in records]),
    }


def run_single_simulation(cfg: dict[str, Any], segment_len: int = 250) -> tuple[dict[str, Any], dict]:
    start = time.perf_counter()
    engine, dataset = build_simulation(cfg)
    result = engine.run(dataset)
    metrics = summarize_records(result.records, dataset.drift_positions, segment_len)
    metrics["online_updates"] = int(getattr(engine.predictor, "online_updates", 0))
    metrics["runtime_sec"] = time.perf_counter() - start
    series = step_series(result.records)
    series["drift_positions"] = np.array(dataset.drift_positions)
    return metrics, series


def run_benchmark(
    n_seeds: int = 5,
    count: int = 250,
    severity: int = 1,
    online: dict[str, Any] | None = None,
    detector_sees_all_labels: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    all_series: dict[str, np.ndarray] = {}
    print(f"\nCIFAR-10-C benchmark: {n_seeds} seeds, {count} images/segment, severity {severity}")
    for k in range(n_seeds):
        seed = BASE_SEED + k
        first_start = seed_window_start(k, count)
        print(f"-- seed {seed}: stream uses test images [{first_start}, {first_start + 3 * count})")
        for item in configurations(seed):
            cfg = build_run_config(
                item, seed, first_start, count, severity,
                online=online, detector_sees_all_labels=detector_sees_all_labels,
            )
            metrics, series = run_single_simulation(cfg, segment_len=count)
            for key, values in series.items():
                all_series[f"{item['name']}|{seed}|{key}"] = values
            all_series[f"{item['name']}|{seed}|alarms"] = np.array(metrics["alarm_steps"], dtype=int)
            metrics.update(
                {
                    "solution": item["name"],
                    "group": item["group"],
                    "type": "Supervised" if item["supervised"] else (
                        "Privileged" if item["signal"] == "step" else "Label-Free"
                    ),
                    "signal": item["signal"],
                    "seed": seed,
                }
            )
            rows.append(metrics)
            print(
                f"   {item['name']:<42} acc={metrics['acc_overall']:.3f} post={metrics['acc_post']:.3f} "
                f"cov_post={metrics['cov_post']:.3f} alarms={metrics['total_alarms']:<3} "
                f"labels={metrics['labels_used']:<4} updates={metrics['online_updates']}"
            )

    raw = pd.DataFrame(rows)
    agg = aggregate(raw)
    raw.drop(columns=["alarm_steps"]).to_csv(out / f"{FILE_PREFIX}_runs_raw{suffix}.csv", index=False)
    raw[["solution", "seed", "alarm_steps"]].to_json(out / f"{FILE_PREFIX}_alarm_steps{suffix}.json", orient="records")
    agg.to_csv(out / f"{FILE_PREFIX}_benchmark{suffix}.csv", index=False)
    np.savez_compressed(out / f"{FILE_PREFIX}_series{suffix}.npz", **all_series)
    print(f"\nSaved {out / f'{FILE_PREFIX}_benchmark{suffix}.csv'}")
    return raw, agg


METRICS = [
    "acc_overall", "acc_pre", "acc_post", "acc_clean", "acc_gaussian", "acc_shot",
    "cov_overall", "cov_pre", "cov_post", "set_size", "set_size_post",
    "total_alarms", "false_alarms", "drifts_detected", "detection_rate", "mean_delay",
    "labels_used", "online_updates",
]


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds, preserving the configuration order of the suite."""
    order = list(dict.fromkeys(raw["solution"]))
    grouped = raw.groupby("solution", sort=False)
    mean = grouped[METRICS].mean()
    std = grouped[METRICS].std(ddof=1).add_suffix("_std")
    meta = grouped[["group", "type", "signal"]].first()
    n_found = grouped["mean_delay"].count().rename("delay_n_seeds")
    out = pd.concat([meta, mean, std, n_found], axis=1).loc[order].reset_index()
    out["n_seeds"] = int(raw["seed"].nunique())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--count", type=int, default=250, help="images per stream segment")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--detector-sees-all-labels",
        action="store_true",
        help="ablation: error-based detectors see the label of every item, not only queried ones",
    )
    args = parser.parse_args()

    run_benchmark(
        n_seeds=args.seeds,
        count=args.count,
        severity=args.severity,
        detector_sees_all_labels=args.detector_sees_all_labels,
        output_dir=args.output_dir,
        suffix="_free_labels" if args.detector_sees_all_labels else "",
    )
