# dacal-stream-learning
Modular research framework for drift-adaptive conformal active learning in streaming ML. Combines concept drift detection, conformal calibration, and active query policies with interchangeable models, datasets, and detectors. Built on PyTorch Lightning for rapid experimentation with classification, regression, and time series streams.

## Synthetic drift dataset generator
The project now includes `SyntheticDriftStreamDataset` in `src/data/synthetic_drift_dataset.py`.
It produces a deterministic binary stream with abrupt phase-wise concept drift.

Minimal config example for `DatasetFactory`:
```yaml
task:
  type: classification

dataset:
  name: synthetic_drift
  seed: 42
  n_per_phase: 1000
  n_phases: 4
  n_features: 10
  noise_std: 0.6
  drift_angle: 1.5707963267948966
```

## Protocol comparison runner
Use `playground/compare_conformal_protocols.py` to compare ROC-AUC under two evaluation protocols:
- notebook-style (`phase 0` training, drift-window labels),
- original-like (`before/after` fitting on full stream, post-drift labels).

Run:
```powershell
python playground/compare_conformal_protocols.py
```

## Observed protocol differences
During the initial development of the framework, we observed that evaluation protocol choices can lead to significant
differences in reported model performance, even when using the same underlying localizer and dataset. For example:

- We constructed our own protocols and compared them with the experiment in the
[original work](https://github.com/FabianHinder/Advanced-Drift-Localization/blob/main/EXP--Drift-Localization-using-Conformal-Predictions.py)

- On the same synthetic `L Sudden 3` stream, ROC-AUC changes noticeably across protocols.
- Example run (`seed=42`): notebook + drift-window labels `0.5330`, notebook + all post-drift labels `0.5081`, original-like `0.6621`.

- `notebook + drift-window labels`:
  train on `phase 0` only, score the full stream, and set positive labels only in short windows right after each drift point.
  This measures transition detection (catching immediate drift onsets), which is typically a strict and imbalanced task.

- `notebook + all post-drift labels`:
  use the same notebook scores, but label all samples after `phase 0` as positive.
  This measures broad pre-drift vs post-drift separation, not specifically immediate transition detection.

- `original-like`:
  fit using before/after-style supervision on the full stream, then evaluate against post-drift positives.
  This setup gives the localizer access to a different supervision signal than phase-0-only training.

- Main driver is target definition and fitting regime: short drift windows vs all post-drift labels are different tasks,
  and phase-0-only training vs full-stream supervised fitting changes what signal is learned.
- These numbers are not directly interchangeable as model-quality rankings unless label semantics and training protocol are matched.
- Use the comparison runner to align protocol choices first, then compare localizers.

| Protocol | Training Data | Positive Label (GT) | Primary Metric Focus |
| --- | --- | --- | --- |
| Drift-Window | Phase 0 only | Short window at onset | Reaction Time / Precision |
| All Post-Drift | Phase 0 only | Everything after `t_drift` | Distribution Robustness |
| Original-Like | Full Stream (Supervised) | Everything after `t_drift` | Discriminative Power |
