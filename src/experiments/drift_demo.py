"""End-to-end demo: lower/upper baselines and all drift detectors on sudden vs. gradual noise drift.

Scenarios = noise type x drift kind (CIFAR-10-C, one stream per scenario and seed):

- ``sudden``  : ``clean`` -> one jump to the noise at the highest severity.
- ``gradual`` : ``clean`` -> the same noise with severity 1, 2, 3, 4, 5 (progressively worse).

Both kinds have the same length and end at the same severity, so they differ only in how the
drift arrives. Noise types: Gaussian, shot, impulse and speckle noise.

Compared algorithms (13):

- **Lower baseline**: frozen pretrained model, no labels, no detector, static clean calibration.
- **Upper baseline**: an oracle alarms at every shift and every label of the following segment is
  revealed, so the model learns (online update + recalibration) at every shift.
- AL only (control), 4 supervised detectors, 4 label-free detectors, and the oracle with the
  normal 5% / 80% label protocol of the CIFAR-10-C benchmark.

Usage (from the repository root):
    PYTHONPATH=src python -m experiments.drift_demo                 # full demo
    PYTHONPATH=src python -m experiments.drift_demo --quick         # smoke run: 1 noise, 1 seed
    PYTHONPATH=src python -m experiments.drift_demo --plots-only    # redraw from cached runs
"""

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmark_detectors import _covered, _set_size
from experiments.cifar10c_suite import (
    CHECKPOINT_PATH,
    LOWER_BASELINE_NAME,
    UPPER_BASELINE_NAME,
    configurations,
    conformal_config,
    model_config,
    upper_baseline,
)

BASE_SEED = 42
DEFAULT_OUTPUT_DIR = "artifacts/demo"
TARGET_COVERAGE = 0.90
AL_ONLY_NAME = "Baseline + AL (No Drift Detector)"
ORACLE_NAME = "Upper Bound: Oracle"

NOISES = {
    "gaussian_noise": "Gaussian",
    "shot_noise": "Shot",
    "impulse_noise": "Impulse",
    "speckle_noise": "Speckle",
}
RAMP_LEVELS = (1, 2, 3, 4, 5)

COLORS = {
    LOWER_BASELINE_NAME: "#d95f02",
    UPPER_BASELINE_NAME: "#1a9641",
    AL_ONLY_NAME: "#555555",
    ORACLE_NAME: "#66c2a5",
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
    LOWER_BASELINE_NAME: "Lower baseline (frozen)",
    UPPER_BASELINE_NAME: "Upper baseline (learns every shift)",
    AL_ONLY_NAME: "AL only (no detector)",
    ORACLE_NAME: "Oracle (5%/80% labels)",
}
BASELINES = (LOWER_BASELINE_NAME, UPPER_BASELINE_NAME)


def short_name(name: str) -> str:
    for prefix in ("Supervised: ", "Label-Free: "):
        name = name.removeprefix(prefix)
    return SHORT.get(name, name)


@dataclass(frozen=True)
class Scenario:
    kind: str  # "sudden" | "gradual"
    noise: str  # CIFAR-10-C corruption file name
    segments: tuple[tuple[str, str, int | None, int], ...]  # (name, corruption, severity, count)

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.noise}"

    @property
    def label(self) -> str:
        return f"{NOISES[self.noise]}\n{self.kind}"

    @property
    def length(self) -> int:
        return sum(seg[3] for seg in self.segments)


def build_scenarios(
    noises: list[str], clean_len: int = 300, stage_len: int = 100
) -> list[Scenario]:
    scenarios = []
    for noise in noises:
        clean = ("clean", "clean", None, clean_len)
        top = RAMP_LEVELS[-1]
        sudden = (clean, (f"{noise} s{top}", noise, top, stage_len * len(RAMP_LEVELS)))
        gradual = (
            clean,
            *((f"{noise} s{s}", noise, s, stage_len) for s in RAMP_LEVELS),
        )
        scenarios.append(Scenario("sudden", noise, sudden))
        scenarios.append(Scenario("gradual", noise, gradual))
    return scenarios


def build_run_config(
    scenario: Scenario,
    item: dict[str, Any],
    seed: int,
    first_start: int,
    data_dir: str,
    clean_root: str,
    checkpoint: str,
    stage_len: int,
) -> dict[str, Any]:
    segments, start = [], first_start
    for name, corruption, severity, count in scenario.segments:
        seg: dict[str, Any] = {
            "name": name,
            "corruption": corruption,
            "count": count,
            "start": start,
        }
        if severity is not None:
            seg["severity"] = severity
        segments.append(seg)
        start += count

    model_cfg = model_config(online={"online_seed": seed})
    model_cfg["checkpoint_path"] = checkpoint
    model_cfg["training"]["clean_root"] = clean_root
    # A burst has to outlast the longest segment so the upper baseline labels all of it.
    burst = max(seg[3] for seg in scenario.segments) + 1
    is_upper = item["name"] == UPPER_BASELINE_NAME
    return {
        "dataset": {
            "name": "cifar10c_stream",
            "data_dir": data_dir,
            "clean_root": clean_root,
            "segments": segments,
            "enforce_disjoint": True,
        },
        "model": model_cfg,
        "conformal": conformal_config(),
        "active_policy": item["active_policy"],
        "drift_detector": item["drift_detector"],
        "engine": {
            "drift_burst": burst if is_upper else 100,
            "burst_rate": 1.0 if is_upper else 0.8,
            "seed": seed,
            "supervised_signals_require_query": True,
        },
    }


def all_configurations(seed: int) -> list[dict[str, Any]]:
    """Lower baseline first, upper baseline second, then the other 10 configurations."""
    items = configurations(seed)
    return [items[0], upper_baseline(seed), *items[1:]]


def run_one(cfg: dict[str, Any]) -> dict[str, Any]:
    from engine.simulation_builder import build_simulation

    t0 = time.perf_counter()
    engine, dataset = build_simulation(cfg)
    result = engine.run(dataset)
    records = result.records
    alarms = [
        r["step"]
        for r in records
        if r.get("drift_event") is not None and getattr(r["drift_event"], "drift", False)
    ]
    return {
        "correct": np.array(
            [np.nan if r["correct"] is None else float(r["correct"]) for r in records]
        ),
        "covered": np.array([_covered(r) for r in records]),
        "set_size": np.array([_set_size(r) for r in records]),
        "queried": np.array([float(r["queried"]) for r in records]),
        "alarms": np.array(alarms, dtype=int),
        "drift_positions": np.array(dataset.drift_positions, dtype=int),
        "online_updates": int(getattr(engine.predictor, "online_updates", 0)),
        "runtime_sec": time.perf_counter() - t0,
    }


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    onset = int(run["drift_positions"][0])
    alarms = run["alarms"]
    post_alarms = alarms[alarms >= onset]
    sl = slice(onset, None)
    return {
        "acc_overall": float(np.nanmean(run["correct"])),
        "acc_pre": float(np.nanmean(run["correct"][:onset])),
        "acc_post": float(np.nanmean(run["correct"][sl])),
        "cov_post": float(np.nanmean(run["covered"][sl])),
        "set_size_post": float(np.nanmean(run["set_size"][sl])),
        "alarms": int(len(alarms)),
        "false_alarms": int((alarms < onset).sum()),
        "detected": bool(len(post_alarms)),
        "delay": float(post_alarms[0] - onset) if len(post_alarms) else float("nan"),
        "labels": int(run["queried"].sum()),
        "online_updates": run["online_updates"],
        "runtime_sec": run["runtime_sec"],
    }


def run_job(
    scenario: Scenario,
    seed_index: int,
    data_dir: str,
    clean_root: str,
    checkpoint: str,
    stage_len: int,
) -> tuple[str, int, dict[str, dict[str, Any]]]:
    import torch

    torch.set_num_threads(1)
    seed = BASE_SEED + seed_index
    first_start = seed_index * scenario.length
    out = {}
    for item in all_configurations(seed):
        cfg = build_run_config(
            scenario, item, seed, first_start, data_dir, clean_root, checkpoint, stage_len
        )
        out[item["name"]] = run_one(cfg)
    return scenario.key, seed_index, out


def ensure_checkpoint(checkpoint: str, clean_root: str) -> None:
    if Path(checkpoint).exists():
        return
    from factories.model_factory import ModelFactory

    print(f"Checkpoint {checkpoint} not found: training SimpleCNN once (CPU, can take a while)")
    cfg = model_config()
    cfg["checkpoint_path"] = checkpoint
    cfg["training"]["clean_root"] = clean_root
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    ModelFactory.create({"model": cfg})


def resolve_data_paths(data_root: str | None) -> tuple[str, str]:
    candidates = [data_root] if data_root else ["data", "playground/data"]
    for root in candidates:
        if root and (Path(root) / "CIFAR-10-C" / "labels.npy").exists():
            return str(Path(root) / "CIFAR-10-C"), str(root)
    raise FileNotFoundError(
        f"CIFAR-10-C not found under {candidates}; see scripts/download_CIFAR-10-C.sh "
        "or pass --data-root"
    )


def run_all(
    scenarios: list[Scenario],
    n_seeds: int,
    data_dir: str,
    clean_root: str,
    checkpoint: str,
    stage_len: int,
    workers: int,
    cache_path: Path,
) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cache = pickle.load(fh)
    todo = [(sc, k) for sc in scenarios for k in range(n_seeds) if (sc.key, k) not in cache]
    print(f"{len(scenarios) * n_seeds} jobs, {len(todo)} to run, {workers} workers")
    if not todo:
        return cache

    ensure_checkpoint(checkpoint, clean_root)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_job, sc, k, data_dir, clean_root, checkpoint, stage_len): (sc, k)
            for sc, k in todo
        }
        for done, fut in enumerate(as_completed(futures), 1):
            key, k, runs = fut.result()
            cache[(key, k)] = runs
            with cache_path.open("wb") as fh:
                pickle.dump(cache, fh)
            print(f"[{done}/{len(todo)}] {key} seed#{k} done ({time.perf_counter() - t0:.0f}s)")
    return cache


def build_tables(
    cache: dict[tuple[str, int], dict[str, dict[str, Any]]], scenarios: list[Scenario], n_seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sc in scenarios:
        for k in range(n_seeds):
            for name, run in cache[(sc.key, k)].items():
                rows.append(
                    {"scenario": sc.key, "kind": sc.kind, "noise": sc.noise, "seed": k,
                     "algorithm": name, **summarize(run)}
                )  # fmt: skip
    raw = pd.DataFrame(rows)
    order = list(dict.fromkeys(raw["algorithm"]))
    metrics = [
        "acc_overall", "acc_pre", "acc_post", "cov_post", "set_size_post",
        "alarms", "false_alarms", "detected", "delay", "labels", "online_updates",
    ]  # fmt: skip
    agg = (
        raw.groupby(["scenario", "kind", "noise", "algorithm"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    agg["algorithm"] = pd.Categorical(agg["algorithm"], categories=order, ordered=True)
    # Share of the lower->upper accuracy gap that an algorithm closes after the first drift.
    lo = agg[agg.algorithm == LOWER_BASELINE_NAME].set_index("scenario")["acc_post"]
    hi = agg[agg.algorithm == UPPER_BASELINE_NAME].set_index("scenario")["acc_post"]
    gap = (hi - lo).reindex(agg["scenario"]).to_numpy()
    base = lo.reindex(agg["scenario"]).to_numpy()
    agg["gap_closed"] = np.where(np.abs(gap) > 0.01, (agg["acc_post"] - base) / gap, np.nan)
    return raw, agg.sort_values(["scenario", "algorithm"]).reset_index(drop=True)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):  # skip the noisy warm-up of the trailing window
        chunk = values[max(0, i - window + 1) : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            out[i] = chunk.mean()
    return out


def _mean_series(runs_by_seed: list[dict[str, Any]], key: str, window: int) -> np.ndarray:
    return np.mean([rolling_mean(r[key], window) for r in runs_by_seed], axis=0)


def plot_trajectories(
    cache, scenario: Scenario, n_seeds: int, out_dir: Path, window: int = 40
) -> Path:
    names = list(cache[(scenario.key, 0)].keys())
    supervised = [n for n in names if n.startswith("Supervised")]
    label_free = [n for n in names if n.startswith("Label-Free")]
    common = [LOWER_BASELINE_NAME, UPPER_BASELINE_NAME, AL_ONLY_NAME, ORACLE_NAME]
    drifts = cache[(scenario.key, 0)][common[0]]["drift_positions"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for col, (title, group) in enumerate(
        [("Supervised detectors", supervised), ("Label-free detectors", label_free)]
    ):
        for row, (key, ylab) in enumerate([("correct", "Accuracy"), ("covered", "Coverage")]):
            ax = axes[row, col]
            for name in [*common, *group]:
                runs = [cache[(scenario.key, k)][name] for k in range(n_seeds)]
                is_base = name in BASELINES
                ax.plot(
                    _mean_series(runs, key, window),
                    color=COLORS[name],
                    lw=2.8 if is_base else 1.4,
                    ls="--" if name == LOWER_BASELINE_NAME else "-",
                    alpha=1.0 if is_base else 0.85,
                    label=short_name(name),
                )
            for d in drifts:
                ax.axvline(d, color="k", lw=0.8, ls=":")
            if key == "covered":
                ax.axhline(TARGET_COVERAGE, color="gray", lw=0.8, ls="-.")
            ax.set_ylabel(f"{ylab} (rolling {window})")
            ax.set_ylim(0.0 if key == "correct" else 0.3, 1.02)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("stream step")
        axes[0, col].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle(
        f"{NOISES[scenario.noise]} noise, {scenario.kind} drift "
        f"(mean of {n_seeds} seeds, dotted = shift points)"
    )
    fig.tight_layout()
    path = out_dir / f"trajectories_{scenario.kind}_{scenario.noise}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_heatmaps(agg: pd.DataFrame, scenarios: list[Scenario], out_dir: Path) -> Path:
    algs = list(agg["algorithm"].cat.categories)
    cols = [sc.key for sc in scenarios]
    labels = [sc.label for sc in scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 1.5 * len(cols) + 8), 7.5))
    for ax, (metric, title, cmap, fmt, vmin, vmax) in zip(
        axes,
        [
            ("acc_post", "Accuracy after the first shift", "RdYlGn", "{:.2f}", None, None),
            ("gap_closed", "Share of the lower->upper gap closed", "RdYlGn", "{:.0%}", 0.0, 1.0),
        ],
        strict=True,
    ):
        mat = (
            agg.pivot(index="algorithm", columns="scenario", values=metric)
            .reindex(index=algs, columns=cols)
            .to_numpy(dtype=float)
        )
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(cols)), labels, fontsize=8)
        ax.set_yticks(range(len(algs)), [short_name(a) for a in algs], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    path = out_dir / "overview_heatmaps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_damage(agg: pd.DataFrame, out_dir: Path) -> Path:
    noises = list(dict.fromkeys(agg["noise"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    x = np.arange(len(noises))
    for ax, kind in zip(axes, ["sudden", "gradual"], strict=True):
        for off, name in zip((-0.2, 0.2), BASELINES, strict=True):
            vals = [
                agg[(agg.kind == kind) & (agg.noise == n) & (agg.algorithm == name)][
                    "acc_post"
                ].iloc[0]
                for n in noises
            ]
            bars = ax.bar(x + off, vals, 0.38, color=COLORS[name], label=short_name(name))
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, [NOISES[n] for n in noises])
        ax.set_title(f"{kind.capitalize()} drift")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy after first shift")
    axes[0].legend(fontsize=8)
    fig.suptitle("The two baselines per noise type and drift kind")
    fig.tight_layout()
    path = out_dir / "overview_baselines.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_cost_benefit(agg: pd.DataFrame, out_dir: Path) -> Path:
    mean = agg.groupby("algorithm", observed=True)[["acc_post", "labels", "gap_closed"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    for name, row in mean.iterrows():
        is_base = name in BASELINES
        ax.scatter(
            row["labels"], row["acc_post"], s=140 if is_base else 70,
            color=COLORS[name], marker="*" if is_base else "o", zorder=3,
        )  # fmt: skip
        ax.annotate(short_name(name), (row["labels"], row["acc_post"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")  # fmt: skip
    ax.set_xlabel("labels queried per run (mean over scenarios)")
    ax.set_ylabel("accuracy after first shift")
    ax.set_title("Accuracy vs. label cost")
    ax.grid(alpha=0.25)

    ax = axes[1]
    gap = mean["gap_closed"].drop(list(BASELINES)).sort_values()
    ax.barh(
        [short_name(n) for n in gap.index], gap.to_numpy(), color=[COLORS[n] for n in gap.index]
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(1, color=COLORS[UPPER_BASELINE_NAME], lw=1.5, ls="--", label="upper baseline = 100%")
    ax.set_xlabel("share of lower->upper accuracy gap closed (mean over scenarios)")
    ax.set_title("How close each algorithm gets to the upper baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "overview_cost_benefit.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_report(
    raw: pd.DataFrame, agg: pd.DataFrame, scenarios: list[Scenario], n_seeds: int,
    figures: list[Path], out_dir: Path, stage_len: int,
) -> Path:  # fmt: skip
    summary = (
        agg.groupby("algorithm", observed=True)[
            [
                "acc_post",
                "cov_post",
                "set_size_post",
                "alarms",
                "false_alarms",
                "delay",
                "labels",
                "gap_closed",
            ]
        ]
        .mean()
        .join(agg.groupby("algorithm", observed=True)["detected"].mean().rename("detected"))
    )
    lines = [
        "# Drift demo: baselines and detectors under sudden and gradual noise drift",
        "",
        f"{len(scenarios)} scenarios x {n_seeds} seeds, {len(NOISES)} noise types "
        f"({', '.join(NOISES.values())}). Every stream is 300 clean steps followed by 500 drifted "
        f"steps. *Sudden*: one jump to severity 5. *Gradual*: severity 1,2,3,4,5 for {stage_len} steps each.",
        "Each seed uses a disjoint window of CIFAR-10 test images; the model is trained on the train split.",
        "All numbers are measured **after the first shift** and averaged over scenarios and seeds.",
        "",
        "- **Lower baseline**: frozen model, no labels, no detector, static clean calibration.",
        "- **Upper baseline**: oracle alarm at every shift, then every label of the segment is revealed "
        "(online update + recalibration), no background labels before the first shift.",
        "- *Gap closed* = (acc - lower) / (upper - lower).",
        "- *Delay* = steps from the first shift to the first alarm (only runs that alarmed); "
        "*Detected* = share of runs with an alarm after the first shift; *False alarms* = alarms before it.",
        "",
        "## Overall",
        "",
        "| Algorithm | Acc | Coverage | Set size | Alarms | False alarms | Detected | Delay | Labels | Gap closed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in summary.iterrows():
        lines.append(
            f"| {short_name(name)} | {r.acc_post:.3f} | {r.cov_post:.3f} | {r.set_size_post:.2f} | "
            f"{r.alarms:.1f} | {r.false_alarms:.1f} | {r.detected:.0%} | "
            f"{'n/a' if np.isnan(r.delay) else f'{r.delay:.0f}'} | {r.labels:.0f} | "
            f"{'n/a' if np.isnan(r.gap_closed) else f'{r.gap_closed:.0%}'} |"
        )
    for kind in ("sudden", "gradual"):
        sub = agg[agg.kind == kind]
        pivot = sub.pivot(index="algorithm", columns="noise", values="acc_post")
        noises = [n for n in NOISES if n in pivot.columns]
        lines += [
            "",
            f"## Accuracy after the first shift: {kind} drift",
            "",
            "| Algorithm | " + " | ".join(NOISES[n] for n in noises) + " | Mean |",
            "|---|" + "---|" * (len(noises) + 1),
        ]
        for name in pivot.index:
            vals = pivot.loc[name, noises]
            lines.append(
                f"| {short_name(name)} | "
                + " | ".join(f"{v:.3f}" for v in vals)
                + f" | {vals.mean():.3f} |"
            )
    lines += ["", "## Figures", ""]
    lines += [f"![{p.stem}]({p.name})" for p in figures]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--noises", nargs="+", default=list(NOISES), choices=list(NOISES))
    parser.add_argument("--stage-len", type=int, default=100, help="steps per severity stage")
    parser.add_argument("--clean-len", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--data-root", default=None, help="folder with CIFAR-10-C/ and cifar-10-batches-py/"
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="1 noise type, 1 seed")
    parser.add_argument("--plots-only", action="store_true", help="only redraw from the cache")
    args = parser.parse_args()

    if args.quick:
        args.noises, args.seeds = args.noises[:1], 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(args.noises, args.clean_len, args.stage_len)
    if args.seeds * scenarios[0].length > 8000:
        raise ValueError("seeds * stream length exceeds the 8000 benchmark test images")
    data_dir, clean_root = resolve_data_paths(args.data_root)
    cache_path = out_dir / f"runs_cache_c{args.clean_len}_s{args.stage_len}.pkl"

    if args.plots_only and not cache_path.exists():
        raise FileNotFoundError(cache_path)
    cache = run_all(
        scenarios, args.seeds, data_dir, clean_root, args.checkpoint, args.stage_len,
        args.workers, cache_path,
    )  # fmt: skip

    raw, agg = build_tables(cache, scenarios, args.seeds)
    raw.to_csv(out_dir / "runs_raw.csv", index=False)
    agg.to_csv(out_dir / "summary_by_scenario.csv", index=False)
    (out_dir / "scenarios.txt").write_text(
        "\n".join(str(asdict(sc)) for sc in scenarios), encoding="utf-8"
    )

    figures = [
        plot_cost_benefit(agg, out_dir),
        plot_heatmaps(agg, scenarios, out_dir),
        plot_damage(agg, out_dir),
    ]
    figures += [plot_trajectories(cache, sc, args.seeds, out_dir) for sc in scenarios]
    report = write_report(raw, agg, scenarios, args.seeds, figures, out_dir, args.stage_len)
    print(f"\nReport: {report}")
    print((report.read_text(encoding="utf-8")).split("## Accuracy")[0])


if __name__ == "__main__":
    main()
