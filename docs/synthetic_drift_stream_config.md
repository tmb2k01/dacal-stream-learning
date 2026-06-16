# Synthetic Drift Stream Dataset Config

`SyntheticDriftStreamDataset` creates a binary classification stream with abrupt
phase-wise concept drift. Each phase has balanced class labels and rotates the
class means relative to the previous phase.

Use it through `DatasetFactory` with `dataset.name: synthetic_drift`, or
directly:

```python
from data import SyntheticDriftStreamDataset

dataset = SyntheticDriftStreamDataset.from_yaml("configs/synthetic_drift_stream.yaml")
```

## Example YAML

```yaml
model:
  name: sgd_classifier
  classes: [0, 1]
  loss: log_loss
  alpha: 0.0001
  random_state: 42

conformal:
  enabled: false

dataset:
  name: synthetic_drift
  seed: 42
  n_per_phase: 1000
  n_phases: 4
  n_features: 10
  noise_std: 0.6
  drift_angle: 1.5707963267948966
```

## Dataset Settings

`name`: must be `synthetic_drift`.

`seed`: random seed used to generate every phase. Defaults to `42`.

`n_per_phase`: number of samples in each drift phase. Must be even and at least
`2`. Defaults to `1000`.

`n_phases`: number of phases in the stream. Must be at least `2`. Defaults to
`4`.

`n_features`: number of feature dimensions. Must be at least `2` because drift
is implemented as a rotation in the first two dimensions. Defaults to `10`.

`noise_std`: standard deviation around each class mean. Larger values make the
classes overlap more. Defaults to `0.6`.

`drift_angle`: rotation angle in radians applied between consecutive phases.
Defaults to pi/2, written in YAML as `1.5707963267948966`.

## Stream Items

Iteration yields dictionaries:

```python
{
    "x": feature_vector,
    "y": label,
    "metadata": {
        "index": 0,
        "phase": 0,
        "is_drift_point": False,
    },
}
```

`is_drift_point` is `true` on the first sample of every phase after the first.
`drift_positions` contains those stream indices.

The first generated phase is also exposed as `X_train` and `y_train`, which can
be used for initial model fitting or calibration before replaying the full
stream.
