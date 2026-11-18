# CIFAR-10-C Benchmark Results


## 1. Setup

**Stream.** `clean → gaussian noise → shot noise`, 250 images each (750 steps, severity 1).
Segment `i` of seed `k` reads CIFAR-10 test images `[750k + 250i, 750k + 250(i+1))`. Corrupted
file `j` is the corrupted version of test image `j`, so every one of the 750 images appears exactly
once. Seeds use non-overlapping windows (test images 0–3749). Test images 8000–9999 are reserved for tuning.

**Model.** `SimpleCNN` trained on 45,000 CIFAR-10 *train* images (5,000 more are held out for
conformal calibration; the 10,000 test images are never used for training).

**Data verification** (`artifacts/benchmarks/data_split_verification.json`, result **PASS**):
- training and calibration indices overlap in 0 images;
- all 5 stream windows have 750 distinct underlying images, and no image is shared between windows;
- stream labels equal the CIFAR-10 test labels, and clean-segment pixels equal the test split;
- 0 exact pixel duplicates between the 45,000 training images and the 10,000 test images;
- the tuning pool (8000–9999) is disjoint from all benchmark windows.

**Active learning protocol (identical for all configurations except the frozen baseline).**
1. The model predicts, and the conformal calibrator builds a prediction set (LAC, target coverage 90%).
2. A random policy queries the true label with probability 5%. After a drift alarm, labels are queried
   with probability 80% for the next 100 steps ("burst").
3. Every queried label causes **one online model update** (Adam, lr 3e-4, only the two dense layers, dropout off)
   and one conformal recalibration step. The runs confirm `online updates == labels queried` in every row.
4. Unlabeled steps cause no update. Error-based detectors only receive the error of queried steps.
5. An alarm resets the conformal window and the detector. The weights are kept.

**Metrics.** "Corrupted" = the two corrupted segments (steps 250–749). Accuracy and coverage are
measured on every step, using the true label for evaluation only. *Coverage* = share of steps whose
prediction set contains the true class (target 0.90). *Set size* = mean number of classes in the set (smaller is
more informative). *Alarms (false)* = alarms per run (alarms before step 250). *Drifts found* = share of the
two drifts that got an alarm inside their own segment. *Delay* = steps from the drift point to the first alarm,
averaged over drifts that were found. *Labels* = number of queried labels out of 750.

---

## 2. The benchmarks (what each configuration is)

The 11 configurations are grouped into three figures. "AL only" is repeated in the detector figures as the control.

**Reference points**

| Configuration | What it is | What it tells us |
|---|---|---|
| **Lower Baseline** | Frozen pretrained model, no labels, no detector, static conformal calibration on clean data. | The floor: damage caused by the drift when nothing adapts. |
| **Baseline + AL (No Drift Detector)** | Active learning at 5% labels with online updates and recalibration, but no detector and no bursts. | Control: what the background labeling alone achieves. Any detector must beat this to be useful. |
| **Upper Bound: Oracle** | Alarms exactly at steps 250 and 500 (delay 0), then a label burst. | Ceiling for "perfect detection" under this label protocol. |

**Supervised detectors** (signal: 0/1 prediction error, available **only on queried steps**)

| Configuration | Mechanism |
|---|---|
| **Baseline Z-Score** | Learns mean/std of the error over the first 10 labeled samples, alarms after 2 consecutive samples more than 2.5 std away. |
| **ADWIN** | Keeps an adaptive window of recent errors and alarms when two sub-windows differ in mean more than a Hoeffding-type bound (δ = 0.01). |
| **DDM** | Tracks error rate p and its std s, alarms when p + s exceeds the observed minimum by 2.5 s (after 10 warm-up samples). |
| **Page-Hinkley** | Cumulative sum of deviations from the running error mean, alarms when it rises λ = 5 above its minimum. |

**Label-free detectors** (signal available on **every** step, no labels needed)

| Configuration | Mechanism |
|---|---|
| **KS Test (Confidence)** | Two-sample Kolmogorov–Smirnov test of the model's max-probability in the last 50 steps vs. a clean reference, every 20 steps, α = 0.05. |
| **CUSUM (Confidence)** | Averages confidence in blocks of 20, standardizes with clean statistics, and alarms when the two-sided cumulative sum exceeds h = 4. |
| **MMD (Embedding)** | Kernel two-sample test (RBF, median bandwidth, PCA to 16 dims, permutation threshold, α = 0.05) between the last 50 embeddings and a clean reference, every 20 steps. |
| **Adaptive MMD** | As MMD, but after an alarm it rebuilds the reference from the next 100 stream samples, so the same shift does not keep alarming. |

---

## 3. Results

Corrupted-segment means; "±" is the std over 5 seeds. Rows are in figure order.

| Configuration | Acc overall | Acc clean | Acc gauss. | Acc shot | Acc corrupted | Coverage corrupted | Set size corrupted | Alarms (false) | Drifts found | Mean delay (steps) | Labels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lower Baseline (Frozen, No AL, No Drift) | 0.722 ± 0.014 | 0.808 | 0.638 | 0.719 | 0.678 ± 0.013 | 0.769 ± 0.011 | 1.32 | 0.0 (0.0) | 0% | n/a | 0 ± 0 |
| Baseline + AL (No Drift Detector) | 0.718 ± 0.019 | 0.802 | 0.626 | 0.726 | 0.676 ± 0.021 | 0.888 ± 0.033 | 2.77 | 0.0 (0.0) | 0% | n/a | 39 ± 3 |
| Upper Bound: Oracle | 0.747 ± 0.025 | 0.802 | 0.694 | 0.744 | 0.719 ± 0.027 | 0.919 ± 0.019 | 2.38 | 2.0 (0.0) | 100% | 0 (n=5/5) | 219 ± 3 |
| Supervised: Baseline Z-Score | 0.724 ± 0.028 | 0.802 | 0.634 | 0.735 | 0.685 ± 0.034 | 0.884 ± 0.034 | 2.50 | 0.2 (0.0) | 10% | 172 (n=1/5) | 58 ± 40 |
| Supervised: ADWIN | 0.718 ± 0.019 | 0.802 | 0.626 | 0.726 | 0.676 ± 0.021 | 0.888 ± 0.033 | 2.77 | 0.0 (0.0) | 0% | n/a | 39 ± 3 |
| Supervised: DDM | 0.728 ± 0.018 | 0.802 | 0.633 | 0.749 | 0.691 ± 0.021 | 0.866 ± 0.032 | 2.17 | 1.4 (0.0) | 40% | 125 (n=4/5) | 122 ± 54 |
| Supervised: Page-Hinkley | 0.720 ± 0.017 | 0.802 | 0.626 | 0.733 | 0.679 ± 0.021 | 0.879 ± 0.023 | 2.51 | 0.6 (0.0) | 20% | 138 (n=1/5) | 81 ± 94 |
| Label-Free: KS Test (Confidence) | 0.738 ± 0.018 | 0.799 | 0.668 | 0.748 | 0.708 ± 0.020 | 0.904 ± 0.038 | 2.27 | 2.6 (0.6) | 40% | 66 (n=3/5) | 214 ± 110 |
| Label-Free: CUSUM (Confidence) | 0.727 ± 0.016 | 0.802 | 0.635 | 0.742 | 0.689 ± 0.013 | 0.891 ± 0.040 | 2.50 | 1.0 (0.0) | 40% | 144 (n=3/5) | 125 ± 103 |
| Label-Free: MMD (Embedding) | 0.738 ± 0.015 | 0.797 | 0.674 | 0.742 | 0.708 ± 0.012 | 0.879 ± 0.009 | 2.26 | 10.4 (1.2) | 100% | 36 (n=5/5) | 536 ± 94 |
| Label-Free: Adaptive MMD | 0.744 ± 0.020 | 0.795 | 0.689 | 0.748 | 0.718 ± 0.018 | 0.918 ± 0.018 | 2.43 | 3.2 (0.8) | 80% | 112 (n=5/5) | 321 ± 79 |

Delay `n=k/5` is the number of seeds in which at least one drift was found; a mean over few seeds is weak evidence.

### Figure group 1: reference points

![Reference trajectories](../artifacts/plots/cifar10c_v2_trajectories_reference.png)
![Reference summary](../artifacts/plots/cifar10c_v2_summary_reference.png)

### Figure group 2: supervised detectors

![Supervised trajectories](../artifacts/plots/cifar10c_v2_trajectories_supervised.png)
![Supervised summary](../artifacts/plots/cifar10c_v2_summary_supervised.png)

### Figure group 3: label-free detectors

![Label-free trajectories](../artifacts/plots/cifar10c_v2_trajectories_label_free.png)
![Label-free summary](../artifacts/plots/cifar10c_v2_summary_label_free.png)

### What the numbers say

1. **The drift hurts, and recalibration fixes coverage, not accuracy.** The frozen model loses about 13 points of accuracy on the
   corrupted segments (0.808 clean → 0.678) and its coverage falls to 0.769. Active learning at 5% labels leaves accuracy unchanged
   (0.676; per-seed differences to the frozen model: −0.010, −0.006, −0.028, +0.010, +0.022). It lifts coverage to 0.888 because the
   calibrator re-estimates the threshold, and it pays for this with larger sets (2.77 vs. 1.32 classes).
2. **Detection helps through the label burst.** An alarm triggers about 80 extra labels. The Oracle (2 alarms, 219 labels) gains +4.3
   accuracy points and reaches 0.919 coverage with smaller sets than AL only. Adaptive MMD, MMD and KS gain +4.2, +3.2 and +3.2 points.
   For Adaptive MMD, MMD and the Oracle the gain over AL only is positive in each of the 5 seeds (Adaptive MMD: +0.018 to +0.082).
3. **Supervised detectors almost never fire when labels are scarce.** With 5% labels they see about 12 labeled samples (2–3 errors) in the clean segment.
   Alarms per run: ADWIN 0.0, Z-Score 0.2, Page-Hinkley 0.6, DDM 1.4. Their results are therefore close to AL only. The ablation
   below shows this is caused by label scarcity, not by the detectors being weak.
4. **Label-free detectors differ in label cost.** MMD finds every drift fastest (36 steps) but alarms 10.4 times per run and uses 536 labels
   (72% of the stream). Adaptive MMD has the best accuracy of all detectors (0.718), 0.918 coverage and 3.2 alarms with 321 labels.
   CUSUM and KS miss a drift in about 60% of the cases.
5. **Differences between detectors are mostly within noise.** With 5 seeds and stds of 0.01–0.03, only the gaps between
   {Oracle, Adaptive MMD, MMD, KS} and {AL only, ADWIN, Page-Hinkley, Z-Score} are clear. Do not read a ranking inside these two sets from this data.

### Ablation: what the label fix changed for supervised detectors

Same runs with `--detector-sees-all-labels` (detectors read the true error of every step, as the old benchmark did). Arrows: queried-only → free labels.

| Detector | Acc corrupted | Coverage corrupted | Alarms | Drifts found | Labels |
|---|---|---|---|---|---|
| Baseline Z-Score | 0.685 → 0.672 | 0.884 → 0.841 | 0.2 → 1.0 (false 1.0) | 10% → 0% | 58 → 129 |
| ADWIN | 0.676 → 0.682 | 0.888 → 0.878 | 0.0 → 0.4 (false 0.0) | 0% → 20% | 39 → 75 |
| DDM | 0.691 → 0.704 | 0.866 → 0.907 | 1.4 → 2.0 (false 0.6) | 40% → 50% | 122 → 210 |
| Page-Hinkley | 0.679 → 0.710 | 0.879 → 0.903 | 0.6 → 8.2 (false 2.2) | 20% → 100% | 81 → 561 |

Free labels mostly help DDM and Page-Hinkley, but Page-Hinkley then fires 8.2 times per run and spends 561 labels. Figures for this
ablation are in `artifacts/plots/*_free_labels.png`.

---

## 4. Online-update tuning (validation stream, test images 8000–9999)

Validation: 2 windows × 2 policy seeds, label rate 5%, no detector; metric = accuracy over the whole 750-step stream.
The frozen model scores **0.736**.

- **Stage 1** (40 variants of whole-network updates: Adam/SGD, lr, dropout on/off, replay, 1 vs. 3 steps): the best scored 0.726,
  **none beat the frozen model**. Higher learning rates, SGD, and 3 steps per label were much worse (down to 0.12).
- **Stage 2** (72 variants adding "train only head / dense layers / all" and mixing clean training images into each update):
  40 beat the frozen model. The best, **Adam lr 3e-4 on the two dense layers, no replay, no anchors, dropout off**, scored 0.754 (+1.8 points;
  clean accuracy preserved: 0.840 vs. 0.844). The next eight variants are within 0.6 points of it, so the choice is the simplest top option, not a sharp optimum.


---

## 5. Limitations

- **Label budget is not equalized.** Detectors that alarm also receive more labels (39 for AL only vs. 219–536 for the detector-based setups),
  so part of the gain is "more labels", not "better detection". The comparison answers "what does an alarm buy in this protocol", not
  "which detector is the best per label". An equal-budget comparison (e.g. matching the background rate to the label count) is not done.
- **One mild drift.** Severity 1, two corruption types, 250 images per segment. The drift is small (13 points), and the alarm
  parameters of the detectors were set by hand for the sparse-label regime (warm-up of 10 labels, Page-Hinkley λ = 5) and not tuned.
- **5 seeds.** The seeds change the stream window, the query decisions and the MMD/KS randomness, but the model checkpoint is the same.
- **Online learning gives little here.** On the validation stream the best update rule adds 1.8 points over frozen; on the benchmark windows, AL only
  equals the frozen model (0.676 vs. 0.678). With 5% labels there are about 12 updates per segment.
- **Changed defaults.** `supervised_signals_require_query=True` changes engine behavior for every config that uses `error`/`loss` detectors.
  Set it to `False` to get the previous behavior. Overlapping CIFAR-10-C segments now raise an error unless `enforce_disjoint: false`.

---
