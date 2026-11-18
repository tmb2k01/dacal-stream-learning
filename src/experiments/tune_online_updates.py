"""Tune SimpleCNN online-update options on VALIDATION streams (test images 8000-9999).

The benchmark windows only use test images [0, 8000), so nothing tuned here is evaluated on
data that influenced the choice. Only the no-detector active-learning setup is used, at the
benchmark label rate, because the question is "does the model learn from the labels it gets".

Usage:
    PYTHONPATH=src python -m experiments.tune_online_updates
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.simulation_builder import build_simulation
from experiments.benchmark_detectors import build_run_config
from experiments.cifar10c_suite import configurations

VAL_WINDOW_STARTS = (8000, 9000)  # each window is 750 images: [8000, 8750) and [9000, 9750)
POLICY_SEEDS = (100, 101)
POOL_END = 10_000

# Stage 1: plain single-sample fine-tuning of the whole network (every variant lost to frozen).
GRID_STAGE1 = {
    "opt": [("adam", 1e-4), ("adam", 3e-4), ("adam", 1e-3), ("sgd", 1e-3), ("sgd", 1e-2)],
    "online_dropout": [True, False],
    "online_replay_size": [0, 64],
    "online_steps": [1, 3],
    "online_trainable": ["all"],
    "online_anchor_batch": [0],
}
# Stage 2: restrict what is trained and mix in clean training images ("anchors").
GRID_STAGE2 = {
    "opt": [("adam", 1e-4), ("adam", 3e-4), ("adam", 1e-3), ("sgd", 1e-3)],
    "online_dropout": [False],
    "online_replay_size": [0, 64],
    "online_steps": [1],
    "online_trainable": ["head", "fc", "all"],
    "online_anchor_batch": [0, 8, 32],
}
GRIDS = {"stage1": GRID_STAGE1, "stage2": GRID_STAGE2}


def run_variant(online: dict, rate: float, window_starts=VAL_WINDOW_STARTS, seeds=POLICY_SEEDS) -> dict:
    """Mean accuracy over validation windows x policy seeds for one online-update variant."""
    suite = configurations(0, rate=max(rate, 0.01))
    name = "Baseline + AL" if rate > 0 else "Lower Baseline"
    base_item = next(c for c in suite if c["name"].startswith(name))
    accs, post, clean = [], [], []
    for start in window_starts:
        for seed in seeds:
            item = base_item
            if rate > 0:
                item = {**base_item, "active_policy": {**base_item["active_policy"], "seed": seed}}
            cfg = build_run_config(item, seed, start, pool_end=POOL_END, online=online)
            engine, ds = build_simulation(cfg)
            res = engine.run(ds)
            correct = np.array([float(r["correct"]) for r in res.records])
            accs.append(correct.mean())
            clean.append(correct[:250].mean())
            post.append(correct[250:].mean())
    return {"acc": float(np.mean(accs)), "acc_clean": float(np.mean(clean)), "acc_post": float(np.mean(post))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--stage", choices=sorted(GRIDS), default="stage2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = args.out or f"artifacts/benchmarks/online_update_tuning_{args.stage}.csv"

    frozen = run_variant({}, rate=0.0)
    print(f"frozen model (no labels): {frozen}")

    rows = []
    t0 = time.perf_counter()
    for (opt, lr), dropout, replay, steps, trainable, anchor in itertools.product(*GRIDS[args.stage].values()):
        online = {
            "online_trainable": trainable,
            "online_anchor_batch": anchor,
            "online_optimizer": opt,
            "online_lr": lr,
            "online_dropout": dropout,
            "online_replay_size": replay,
            "online_replay_batch": 8,
            "online_steps": steps,
        }
        res = run_variant(online, rate=args.rate)
        rows.append({**online, **res, "gain_vs_frozen": res["acc"] - frozen["acc"]})
        print(
            f"{opt:<4} lr={lr:<7} train={trainable:<4} anchor={anchor:<2} replay={replay:<3} steps={steps} "
            f"-> acc={res['acc']:.4f} (gain {res['acc'] - frozen['acc']:+.4f}) [{time.perf_counter() - t0:.0f}s]"
        )

    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\nTop 8:\n", df.head(8).to_string(index=False))
    print(f"\nfrozen acc = {frozen['acc']:.4f}; saved {out_path}")


if __name__ == "__main__":
    main()
