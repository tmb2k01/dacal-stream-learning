#!/usr/bin/env python3
"""Clean CIFAR-10 Baseline Runner.

Executes the 'Lowest Baseline' (static/frozen model on clear CIFAR-10 pictures only)
using the repository's simulation engine and split-conformal calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

from engine import build_simulation, load_config 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lowest baseline on clean CIFAR-10")
    parser.add_argument(
        "--config",
        default="configs/cifar10_clean_baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Number of streaming test images to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--output-plot",
        default="playground/clean_cifar10_baseline.png",
        help="Path to save baseline diagnostic plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = REPO_ROOT / args.config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    engine, dataset = build_simulation(config)
    print(f"Stream loaded: {len(dataset)} clean images available.")
    print(f"Conformal calibrator: alpha={engine.calibrator.alpha}, q_hat={engine.calibrator.q_hat:.4f}")
    print(f"Active policy: {config.get('active_policy', {}).get('enabled', False)} (model weights frozen: TRUE)")

    print(f"\nRunning simulation for {args.max_steps} steps...")
    result = engine.run(dataset, max_steps=args.max_steps)
    metrics = result.state.metrics

    seen = metrics.get("seen", 0)
    correct = metrics.get("correct", 0)
    accuracy = metrics.get("accuracy", 0.0)
    coverage = metrics.get("coverage", 0.0)
    efficiency = metrics.get("efficiency", 0.0)
    informativeness = metrics.get("informativeness", 0.0)

    drift_alerts = [r["step"] for r in result.records if r["drift_event"] and r["drift_event"].drift]
    drift_warnings = [r["step"] for r in result.records if r["drift_event"] and r["drift_event"].warning]

    print("\n" + "=" * 60)
    print("LOWEST BASELINE SUMMARY (CLEAR CIFAR-10 ONLY)")
    print("=" * 60)
    print("Model Architecture         : SimpleCNN (Frozen weights)")
    print("Dataset                    : Clear CIFAR-10 test split (native torchvision)")
    print(f"Total Steps Evaluated      : {seen}")
    print(f"Point Accuracy             : {accuracy * 100:.2f}% ({correct}/{seen})")
    print(f"Conformal Alpha (Target)   : {engine.calibrator.alpha:.2f} (Target Cov: {100 * (1 - engine.calibrator.alpha):.1f}%)")
    print(f"Empirical Coverage         : {coverage * 100:.2f}% ({metrics.get('total_covered', 0)}/{seen})")
    print(f"Efficiency (Avg Set Size)  : {efficiency:.3f} classes")
    print(f"Informativeness            : {informativeness:.3f}")
    print(f"Drift Alarms Triggered     : {len(drift_alerts)} (Steps: {drift_alerts if drift_alerts else 'None'})")
    print(f"Drift Warnings Triggered   : {len(drift_warnings)}")
    print("=" * 60)

    # Class-wise table
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    rows = []
    for cls_idx_str, count in metrics.get("class_counts", {}).items():
        cls_idx = int(cls_idx_str)
        cov_cls = metrics.get("coverage_by_class", {}).get(cls_idx_str, 0.0)
        gap = metrics.get("coverage_gap_by_class", {}).get(cls_idx_str, 0.0)
        rows.append({
            "Class ID": cls_idx,
            "Class Name": class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx),
            "Count": count,
            "Coverage": f"{cov_cls * 100:.1f}%",
            "Target Gap": f"{gap * 100:+.1f}%",
        })
    df_classes = pd.DataFrame(rows).sort_values("Class ID")
    print("\nPer-Class Conformal Coverage Breakdown:")
    print(df_classes.to_string(index=False))

    # Generate timeline plots
    steps = [r["step"] for r in result.records]
    correct_flags = [1 if r["correct"] else 0 for r in result.records]
    cum_acc = np.cumsum(correct_flags) / (np.arange(len(correct_flags)) + 1)

    set_sizes = [
        int(r["conformal_output"]["set_size"][0])
        if "set_size" in r["conformal_output"]
        else 0
        for r in result.records
    ]
    is_covered = [
        1 if bool(r["conformal_output"].get("prediction_set", [[]])[0, int(r["y_true"])])
        else 0
        for r in result.records
    ]
    cum_cov = np.cumsum(is_covered) / (np.arange(len(is_covered)) + 1)
    rolling_sizes = pd.Series(set_sizes).rolling(window=50, min_periods=1).mean()
    detector_scores = [
        r["drift_event"].score if r["drift_event"] is not None else 0.0
        for r in result.records
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1. Cumulative Accuracy
    axes[0].plot(steps, cum_acc, color="royalblue", lw=1.5, label="Cumulative Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Lowest Baseline: Clean CIFAR-10 Stream Performance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # 2. Cumulative Coverage
    axes[1].plot(steps, cum_cov, color="forestgreen", lw=1.5, label="Empirical Coverage")
    axes[1].axhline(1.0 - engine.calibrator.alpha, color="crimson", ls="--", label=f"Target {1.0 - engine.calibrator.alpha:.2f}")
    axes[1].set_ylabel("Coverage")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    # 3. Prediction Set Size
    axes[2].plot(steps, rolling_sizes, color="darkorange", lw=1.5, label="Avg Set Size (rolling 50)")
    axes[2].set_ylabel("Set Size (classes)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    # 4. Drift Detector Score
    axes[3].plot(steps, detector_scores, color="purple", lw=1.2, label="Drift Detector Score (z-score)")
    if hasattr(engine.drift_detector, "threshold_std"):
        axes[3].axhline(engine.drift_detector.threshold_std, color="red", ls=":", label="Drift Threshold")
    axes[3].set_ylabel("Detector Score")
    axes[3].set_xlabel("Stream Step")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    plot_path = REPO_ROOT / args.output_plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nDiagnostic timeline plot saved to: {plot_path}")

    # Generate companion diagnostics figure: Set size distribution & Class-wise coverage
    fig_diag, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # A) Prediction Set Size Distribution Histogram
    unique_sizes, size_counts = np.unique(set_sizes, return_counts=True)
    bars = ax_hist.bar(unique_sizes, size_counts, color="teal", edgecolor="black", alpha=0.85, width=0.6)
    ax_hist.set_xlabel("Prediction Set Size (Number of Classes)")
    ax_hist.set_ylabel("Sample Count")
    ax_hist.set_title(f"Prediction Set Size Distribution (N={len(set_sizes)})")
    ax_hist.set_xticks(unique_sizes)
    ax_hist.grid(axis="y", alpha=0.3)
    for bar, count in zip(bars, size_counts, strict=False):
        pct = count / len(set_sizes) * 100
        ax_hist.annotate(
            f"{count}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # B) Per-Class Empirical Coverage Bar Chart
    cls_names = [r["Class Name"] for r in rows]
    cls_covs = [float(r["Coverage"].rstrip("%")) for r in rows]
    target_cov = (1.0 - engine.calibrator.alpha) * 100

    bar_colors = ["forestgreen" if c >= target_cov else "salmon" for c in cls_covs]
    ax_bar.bar(cls_names, cls_covs, color=bar_colors, edgecolor="black", alpha=0.85)
    ax_bar.axhline(target_cov, color="crimson", ls="--", lw=1.8, label=f"Target Coverage ({target_cov:.0f}%)")
    ax_bar.set_ylabel("Empirical Coverage (%)")
    ax_bar.set_title("Per-Class Conformal Coverage vs Target")
    ax_bar.set_ylim(0, 105)
    ax_bar.tick_params(axis="x", rotation=35)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(loc="lower right")

    plt.tight_layout()
    diag_path = plot_path.parent / "clean_cifar10_baseline_diagnostics.png"
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print(f"Conformal diagnostics plot saved to: {diag_path}")


if __name__ == "__main__":
    main()
