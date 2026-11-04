"""Figures for the CIFAR-10-C benchmark, split into small groups (at most 5 curves per picture).

Reads what ``benchmark_detectors`` wrote (``*_series.npz`` and ``*_benchmark.csv``), so figures and
tables always come from the same runs. For each group in ``FIGURE_GROUPS`` it writes:

- ``cifar10c_v2_trajectories_<group>.png``: rolling accuracy, rolling coverage, rolling
  prediction-set size, and an alarm timeline (all seeds).
- ``cifar10c_v2_summary_<group>.png``: post-drift accuracy, post-drift coverage, detection delay,
  and label usage, as mean +/- std over seeds.

Usage:
    PYTHONPATH=src python -m experiments.evaluate_and_plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmark_detectors import DEFAULT_OUTPUT_DIR, FILE_PREFIX
from experiments.cifar10c_suite import FIGURE_GROUPS, SEGMENT_PLAN

MAX_CURVES_PER_FIGURE = 5
TARGET_COVERAGE = 0.90

COLORS = {
    "Lower Baseline (Frozen, No AL, No Drift)": "#d95f02",
    "Baseline + AL (No Drift Detector)": "#555555",
    "Upper Bound: Oracle": "#2ca02c",
    "Supervised: Baseline Z-Score": "#e6ab02",
    "Supervised: ADWIN": "#1b9e77",
    "Supervised: DDM": "#e7298a",
    "Supervised: Page-Hinkley": "#7570b3",
    "Label-Free: KS Test (Confidence)": "#a6761d",
    "Label-Free: CUSUM (Confidence)": "#ff7f00",
    "Label-Free: MMD (Embedding)": "#1f78b4",
    "Label-Free: Adaptive MMD": "#6a3d9a",
}
SHORT = {
    "Lower Baseline (Frozen, No AL, No Drift)": "Lower Baseline",
    "Baseline + AL (No Drift Detector)": "AL only (no detector)",
}


def short_name(name: str) -> str:
    for prefix in ("Supervised: ", "Label-Free: ", "Upper Bound: "):
        name = name.removeprefix(prefix)
    return SHORT.get(name, name)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean that ignores NaNs; early points use the shorter available history."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            out[i] = chunk.mean()
    return out


def load_series(npz_path: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    raw = np.load(npz_path)
    series: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for key in raw.files:
        name, seed, field = key.split("|")
        series.setdefault(name, {}).setdefault(int(seed), {})[field] = raw[key]
    return series


def _mean_rolling(runs: dict[int, dict[str, np.ndarray]], field: str, window: int) -> np.ndarray:
    return np.mean([rolling_mean(r[field], window) for r in runs.values()], axis=0)


def _style_axis(ax, drift_positions: list[int], n_steps: int) -> None:
    bounds = [0, *drift_positions, n_steps]
    for i in range(len(bounds) - 1):
        if i % 2 == 1:
            ax.axvspan(bounds[i], bounds[i + 1], color="#e8eff5", alpha=0.6, zorder=0)
    for dp in drift_positions:
        ax.axvline(dp, color="#222222", linestyle="--", linewidth=1.1, zorder=1)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, n_steps)


def plot_trajectories(
    series: dict, names: list[str], title: str, output_path: Path, acc_window: int = 50, cov_window: int = 100
) -> None:
    if len(names) > MAX_CURVES_PER_FIGURE:
        raise ValueError(f"at most {MAX_CURVES_PER_FIGURE} curves per figure, got {len(names)}")

    first = next(iter(series[names[0]].values()))
    drift_positions = [int(v) for v in first["drift_positions"]]
    n_steps = len(first["correct"])
    n_seeds = len(series[names[0]])

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 13), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2.2, 2.4]}
    )
    fig.suptitle(f"{title}\n(mean over {n_seeds} seeds)", fontsize=14, fontweight="bold")
    for ax in axes:
        _style_axis(ax, drift_positions, n_steps)
    x = np.arange(n_steps)

    panels = [
        (axes[0], "correct", acc_window, f"Rolling accuracy (window {acc_window})", (0.3, 1.0)),
        (axes[1], "covered", cov_window, f"Rolling coverage (window {cov_window})", (0.3, 1.02)),
        (axes[2], "set_size", cov_window, f"Rolling prediction-set size (window {cov_window})", None),
    ]
    for ax, field, window, label, ylim in panels:
        for name in names:
            ax.plot(
                x, _mean_rolling(series[name], field, window), label=short_name(name),
                color=COLORS[name], linewidth=2.0, zorder=3,
            )
        ax.set_ylabel(label, fontsize=10)
        if ylim:
            ax.set_ylim(*ylim)
    axes[1].axhline(TARGET_COVERAGE, color="red", linestyle="--", linewidth=1.3, label="target 90%")
    axes[0].legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.95)
    axes[1].legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.95)

    for i, (seg_name, _) in enumerate(SEGMENT_PLAN):
        left = ([0, *drift_positions])[i]
        axes[0].text(left + 5, 0.995, seg_name, fontsize=9, color="#222222", va="top", fontweight="bold")

    # Alarm timeline: one row per detector, one dot per alarm, sub-rows for seeds.
    ax_t = axes[3]
    detectors = [n for n in names if any(len(r["alarms"]) for r in series[n].values()) or "Oracle" in n]
    detectors = [n for n in names if n in detectors]
    for row, name in enumerate(detectors):
        for k, (_, run) in enumerate(sorted(series[name].items())):
            offset = (k - (n_seeds - 1) / 2) * 0.12
            ax_t.scatter(
                run["alarms"], np.full(len(run["alarms"]), row + offset), s=18,
                color=COLORS[name], edgecolors="black", linewidths=0.3, zorder=4,
            )
    ax_t.set_yticks(range(len(detectors)))
    ax_t.set_yticklabels([short_name(n) for n in detectors], fontsize=9)
    ax_t.set_ylim(-0.7, max(len(detectors) - 0.3, 0.7))
    ax_t.set_ylabel("Alarms (one dot per\nalarm, rows = seeds)", fontsize=10)
    ax_t.set_xlabel("Stream step (dashed lines = true drift points)", fontsize=11)
    if not detectors:
        ax_t.text(0.5, 0.5, "no detector in this group", transform=ax_t.transAxes, ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def plot_summary(agg: pd.DataFrame, names: list[str], title: str, output_path: Path) -> None:
    if len(names) > MAX_CURVES_PER_FIGURE:
        raise ValueError(f"at most {MAX_CURVES_PER_FIGURE} bars per panel, got {len(names)}")
    df = agg.set_index("solution").loc[names]
    y = np.arange(len(names))[::-1]
    colors = [COLORS[n] for n in names]
    labels = [short_name(n) for n in names]
    n_seeds = int(df["n_seeds"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    fig.suptitle(f"{title}\n(mean +/- std over {n_seeds} seeds)", fontsize=14, fontweight="bold")

    def bars(ax, col, xlabel, xlim=None, fmt="{:.3f}", ref=None):
        vals = df[col].to_numpy(dtype=float)
        errs = df[f"{col}_std"].to_numpy(dtype=float)
        ax.barh(y, np.nan_to_num(vals), xerr=np.nan_to_num(errs), color=colors, edgecolor="black",
                linewidth=0.6, capsize=3, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6, zorder=0)
        if xlim:
            ax.set_xlim(*xlim)
        if ref is not None:
            ax.axvline(ref, color="red", linestyle="--", linewidth=1.3, zorder=4)
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for yi, v, e in zip(y, vals, errs, strict=True):
            text = "n/a" if np.isnan(v) else fmt.format(v)
            right = 0 if np.isnan(v) else v + (0 if np.isnan(e) else e)
            ax.text(right + 0.015 * span, yi, text, va="center", fontsize=9, fontweight="bold")

    lo = max(0.0, float(np.nanmin(df["acc_post"])) - 0.12)
    bars(axes[0, 0], "acc_post", "Accuracy on corrupted segments (higher is better)", (lo, 1.0))
    lo_c = max(0.0, float(np.nanmin(df["cov_post"])) - 0.12)
    bars(axes[0, 1], "cov_post", "Coverage on corrupted segments (target 0.90)", (lo_c, 1.02), ref=TARGET_COVERAGE)

    # Bottom panels: extra facts go into two-line tick labels so they never collide with bars.
    ax = axes[1, 0]
    bars(ax, "mean_delay", "Mean detection delay in steps (lower is better)", fmt="{:.0f}")
    ax.set_yticklabels(
        [
            f"{lab}\n" + ("no detector" if df.loc[n, "signal"] == "none"
                          else f"found {df.loc[n, 'detection_rate'] * 100:.0f}% of drifts")
            for lab, n in zip(labels, names, strict=True)
        ],
        fontsize=8.5,
    )
    ax = axes[1, 1]
    bars(ax, "labels_used", "Labels queried out of 750 steps", fmt="{:.0f}")
    ax.set_yticklabels(
        [
            f"{lab}\nalarms {df.loc[n, 'total_alarms']:.1f} (false {df.loc[n, 'false_alarms']:.1f})"
            for lab, n in zip(labels, names, strict=True)
        ],
        fontsize=8.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots-dir", default="artifacts/plots")
    parser.add_argument("--suffix", default="", help="e.g. _free_labels for the ablation run")
    args = parser.parse_args()

    bench = Path(args.benchmark_dir)
    plots = Path(args.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    series = load_series(bench / f"{FILE_PREFIX}_series{args.suffix}.npz")
    agg = pd.read_csv(bench / f"{FILE_PREFIX}_benchmark{args.suffix}.csv")

    for group, title, names in FIGURE_GROUPS:
        plot_trajectories(series, names, f"CIFAR-10-C stream: {title}",
                          plots / f"{FILE_PREFIX}_trajectories_{group}{args.suffix}.png")
        plot_summary(agg, names, f"CIFAR-10-C summary: {title}",
                     plots / f"{FILE_PREFIX}_summary_{group}{args.suffix}.png")


if __name__ == "__main__":
    main()
