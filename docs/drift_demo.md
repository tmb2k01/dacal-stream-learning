# Drift demo: lower/upper baseline, sudden vs. gradual drift, four noise types

One command runs every algorithm on 8 CIFAR-10-C scenarios and writes tables and figures.

```powershell
$env:PYTHONPATH = "src"
python -m experiments.drift_demo                 # full demo: 4 noises x 2 drift kinds x 3 seeds
python -m experiments.drift_demo --quick         # smoke run: Gaussian only, 1 seed
python -m experiments.drift_demo --plots-only    # redraw tables/figures from the cached runs
```

Options: `--seeds`, `--noises gaussian_noise shot_noise impulse_noise speckle_noise`, `--workers`
(parallel processes), `--data-root` (folder containing `CIFAR-10-C/` and `cifar-10-batches-py/`;
default tries `data/` then `playground/data/`), `--output-dir` (default `artifacts/demo`).
Runs are cached per scenario and seed in `runs_cache_*.pkl`, so an interrupted run resumes.
If the checkpoint `artifacts/models/simplecnn_cifar10_aug_seed42_ep20.pt` is missing it is trained once
(about 25 minutes on CPU). The full demo took about 35 minutes with 7 workers on 14 cores.

## Scenarios

Every stream has 300 clean steps followed by 500 drifted steps, and each seed uses a disjoint window of
CIFAR-10 test images.

| Drift kind | Drifted part |
|---|---|
| `sudden` | one jump to severity 5 of the noise |
| `gradual` | the same noise at severity 1, 2, 3, 4, 5 for 100 steps each (progressively worse; each step is a shift point) |

Noise types: Gaussian, shot, impulse, speckle (`src/experiments/drift_demo.py`, `NOISES`).
Both kinds have the same length and end at the same severity.

## The two baselines

- **Lower baseline**: frozen model, no labels, no detector, static clean conformal calibration.
- **Upper baseline** (`upper_baseline()` in `cifar10c_suite.py`): an oracle alarms at *every* shift and, after each
  alarm, *every* label of the segment is revealed. Each label causes one online update and one conformal
  recalibration step. There are no background labels before the first shift.

The other 10 algorithms are the configurations of the CIFAR-10-C benchmark (`docs/cifar10c_benchmark_results.md`):
AL only (5% labels, no detector), oracle with 5% / 80%-burst labels, four supervised detectors
(Z-Score, ADWIN, DDM, Page-Hinkley) and four label-free detectors (KS, CUSUM, MMD, Adaptive MMD).

## Results (3 seeds, measured after the first shift, mean over the 8 scenarios)

| Algorithm | Acc | Coverage | Set size | Alarms | False alarms | Detected | Delay | Labels | Gap closed |
|---|---|---|---|---|---|---|---|---|---|
| Lower baseline (frozen) | 0.405 | 0.526 | 1.58 | 0.0 | 0.0 | 0% | n/a | 0 | 0% |
| Upper baseline (learns every shift) | 0.546 | 0.907 | 3.94 | 3.0 | 0.0 | 100% | 0 | 499 | 100% |
| AL only (no detector) | 0.464 | 0.858 | 4.98 | 0.0 | 0.0 | 0% | n/a | 45 | 40% |
| Oracle (5%/80% labels) | 0.528 | 0.926 | 4.69 | 3.0 | 0.0 | 100% | 0 | 313 | 90% |
| Baseline Z-Score | 0.488 | 0.859 | 4.69 | 0.3 | 0.0 | 33% | 121 | 75 | 57% |
| ADWIN | 0.464 | 0.858 | 4.98 | 0.0 | 0.0 | 0% | n/a | 45 | 40% |
| DDM | 0.511 | 0.858 | 4.05 | 1.0 | 0.0 | 100% | 154 | 124 | 71% |
| Page-Hinkley | 0.477 | 0.823 | 4.19 | 2.4 | 0.0 | 79% | 296 | 172 | 48% |
| KS Test (Confidence) | 0.516 | 0.800 | 3.35 | 9.2 | 0.0 | 100% | 62 | 438 | 80% |
| CUSUM (Confidence) | 0.509 | 0.743 | 2.68 | 13.3 | 0.3 | 100% | 86 | 438 | 74% |
| MMD (Embedding) | 0.521 | 0.822 | 3.56 | 11.7 | 2.0 | 100% | 32 | 556 | 82% |
| Adaptive MMD | 0.529 | 0.875 | 3.95 | 3.9 | 0.7 | 100% | 45 | 357 | 88% |

*Gap closed* = (acc - lower) / (upper - lower). *Delay* = steps from the first shift to the first alarm.

What the numbers say:

1. **The drift hurts a frozen model a lot, and sudden drift hurts more than gradual.** Frozen accuracy after the first
   shift is 0.33 (sudden) vs. 0.48 (gradual) on average; impulse noise at severity 5 is the worst case (0.31).
   Coverage of the frozen model falls to 0.53 (target 0.90).
2. **Learning at every shift recovers +14 accuracy points** (0.405 -> 0.546) and restores coverage (0.907), at the price of
   labels for the whole drifted stream and larger prediction sets (3.9 classes).
3. **Detectors close 40-90% of the gap.** Adaptive MMD (88%) and the label-free KS / MMD (80-82%) come closest; the
   oracle with burst labels (90%) is the practical ceiling. The oracle with 5%/80% labels matches the upper baseline in
   gradual drift (0.600 vs. 0.601) because it has almost the same label access there (492 vs. 499 labels: its
      bursts cover the whole drifted stream). In sudden drift it uses only 134 labels and closes 81% of the gap.
4. **Supervised detectors are limited by label scarcity.** ADWIN never fires (identical to AL only), Z-Score fires in 33% of
   runs, Page-Hinkley needs 296 steps. DDM is the best supervised detector and cheapest (124 labels).
5. **Label-free detectors alarm often.** MMD alarms 11.7 times and uses 556 labels (more than the upper baseline); CUSUM
   alarms 13.3 times with the lowest coverage (0.74). Adaptive MMD gets similar accuracy with 3.9 alarms and 357 labels.
6. **Noise types behave alike**: the ranking of the algorithms is stable across Gaussian, shot, impulse and speckle noise.
   Impulse noise is the hardest drift for adaptation: every algorithm, including the upper baseline, reaches only
   0.41 in the sudden case.

Caveats: 3 seeds, one checkpoint, hand-set detector parameters, unequal label budgets (see the limitations in
`docs/cifar10c_benchmark_results.md`); differences of about 0.01-0.02 are within noise.

## Figures

All figures are written to `artifacts/demo/`; copies of the overview figures are in `docs/figures/`.
`report.md` there contains the tables for every scenario, and `runs_raw.csv` / `summary_by_scenario.csv` the raw numbers.

![Accuracy vs. label cost](figures/drift_demo_overview_cost_benefit.png)
![Heatmaps over all scenarios](figures/drift_demo_overview_heatmaps.png)
![Baselines per noise type](figures/drift_demo_overview_baselines.png)
![Gaussian, sudden](figures/drift_demo_trajectories_sudden_gaussian_noise.png)
![Gaussian, gradual](figures/drift_demo_trajectories_gradual_gaussian_noise.png)

The trajectory figures (`trajectories_<kind>_<noise>.png`, one per scenario) show rolling accuracy and coverage of both
baselines plus all detectors, with the shift points dotted.
