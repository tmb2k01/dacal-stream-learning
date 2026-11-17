# Drift Detectors and Visualiser - Developer Notes

This document explains the currently implemented drift detectors, the reusable visualiser, and how both are wired into stream simulation.

## Scope and current status

- Implemented detector interface: `BaseDriftDetector` + `DriftEvent` in `src/drift/base.py`.
- Implemented detectors:
  - `RunningStatsBaselineDetector`
  - `ADWINDetector`
  - `DDMDetector`
  - `PageHinkleyDetector`
- Detector creation path: config -> `DriftDetectorFactory` -> `StreamSimulationEngine`.
- Reusable visualiser is implemented in `src/drift/detector_visualizer.py` and can be enabled from config via `drift_detector.visualizer`.

## Key files

- `src/drift/base.py`: detector contract (`DriftEvent`, `BaseDriftDetector`).
- `src/drift/baseline_detector.py`: baseline z-score detector.
- `src/drift/river_detectors.py`: River-backed wrappers (`ADWIN`, `DDM`, `PageHinkley`).
- `src/drift/detector_visualizer.py`: plotting API and detector replay helper.
- `src/factories/detector_factory.py`: config-based detector creation.
- `src/engine/stream_engine.py`: detector update and drift response hooks.
- `src/main.py`: optional runtime visualisation (`drift_detector.visualizer`).

## Detector contract

### `DriftEvent`

Defined in `src/drift/base.py`:

- `drift: bool`: hard drift trigger.
- `warning: bool`: soft warning state.
- `score: float`: detector-specific numeric diagnostic.

### `BaseDriftDetector`

All detectors implement:

- `update(value: float) -> DriftEvent`
- `reset() -> None`

In engine integration, `value` is prediction error (`0.0` for correct, `1.0` for incorrect).

## Implemented detectors

### 1) `RunningStatsBaselineDetector`

Location: `src/drift/baseline_detector.py`

- Warmup phase buffers `warmup_size` values and freezes baseline mean/std.
- Score:

  `abs(value - baseline_mean) / max(baseline_std, min_std)`

- Warning when score crosses `threshold_std`.
- Drift when exceedance streak reaches `consecutive`.
- On drift, detector self-resets and re-enters warmup.

How it works (step-by-step):

1. **Warmup collection**: the detector stores the first `warmup_size` values in a buffer and emits neutral events (`drift=False`, `warning=False`, `score=0.0`).
2. **Baseline freeze**: once warmup is complete, it computes a fixed baseline mean and population standard deviation from the warmup buffer.
3. **Deviation scoring**: for each new value, it computes a standardized distance to the baseline:
   - `abs(value - baseline_mean) / max(baseline_std, min_std)`
4. **Warning stage**: if the score crosses `threshold_std`, the exceedance streak increments and `warning=True` is emitted.
5. **Drift stage**: if exceedances continue for `consecutive` updates, `drift=True` is emitted.
6. **Automatic recovery**: when drift triggers, internal baseline/streak state is reset so a fresh warmup begins.

Practical behavior:

- A single outlier typically creates only a warning.
- Sustained shift is required for drift.
- Because the baseline is fixed after warmup, this detector is interpretable but less adaptive than window-based detectors.

Main parameters:

- `warmup_size` (default `50`)
- `threshold_std` (default `3.0`)
- `consecutive` (default `3`)
- `min_std` (default `1e-8`)

### 2) `ADWINDetector`

Location: `src/drift/river_detectors.py`

- Wraps `river.drift.ADWIN`.
- Drift is emitted from `inner.drift_detected`.
- Score is `inner.variance` when width is positive, else `0.0`.
- On drift, wrapper self-resets.

How it works (step-by-step):

1. **Adaptive window maintenance**: ADWIN maintains a variable-length window of recent values.
2. **Online split test**: it repeatedly checks whether two subwindows inside the current window have significantly different means.
3. **Change decision**: if the mean gap is too large under the confidence set by `delta`, ADWIN flags drift.
4. **Event mapping**: wrapper emits `DriftEvent(drift=..., warning=False, score=...)` where score is ADWIN's current variance estimate.
5. **Automatic recovery**: on drift, wrapper reinitializes ADWIN state.

Practical behavior:

- Smaller `delta` is more conservative (fewer false alarms, slower reaction).
- Larger `delta` is more sensitive (faster reaction, potentially noisier alarms).
- No separate warning mode is exposed in this wrapper.

Main parameter:

- `delta` (default `0.002`, must be in `(0, 1)`).

### 3) `DDMDetector`

Location: `src/drift/river_detectors.py`

- Wraps `river.drift.binary.DDM`.
- Inputs are thresholded to binary (`value >= 0.5 -> 1`, else `0`).
- `warning` and `drift` are exposed from inner detector flags.
- Score is the thresholded binary input (`0.0` or `1.0`).
- On drift, wrapper self-resets.

How it works (step-by-step):

1. **Binary conversion**: wrapper thresholds incoming values to the binary error stream expected by DDM:
   - `value < 0.5 -> 0` (correct), `value >= 0.5 -> 1` (error)
2. **Error-stat tracking**: DDM tracks error probability and uncertainty over time after `warm_start`.
3. **Warning decision**: if deterioration crosses `warning_threshold`, DDM sets warning.
4. **Drift decision**: if deterioration crosses `drift_threshold`, DDM flags drift.
5. **Event mapping**: wrapper forwards warning/drift flags and uses binary input (`0.0` or `1.0`) as score.
6. **Automatic recovery**: on drift, wrapper reinitializes DDM state.

Practical behavior:

- Designed specifically for classification-style error streams.
- Supports an explicit two-stage alert (`warning` then `drift`).
- If your signal is continuous but not naturally binary error, thresholding may hide useful magnitude information.

Main parameters:

- `warm_start` (default `30`)
- `warning_threshold` (default `2.0`)
- `drift_threshold` (default `3.0`, must exceed warning threshold)

### 4) `PageHinkleyDetector`

Location: `src/drift/river_detectors.py`

- Wraps `river.drift.PageHinkley`.
- Emits drift from `inner.drift_detected`.
- Score is the raw `value` forwarded to the test.
- On drift, wrapper self-resets.

How it works (step-by-step):

1. **Sequential mean monitoring**: Page-Hinkley accumulates evidence that the stream mean has shifted.
2. **Forgetting control**: `alpha` controls how strongly older observations are discounted.
3. **Direction control**: `mode` limits detection to upward, downward, or both change directions.
4. **Minimum data gate**: no drift is declared before `min_instances` observations.
5. **Threshold test**: when accumulated change exceeds `threshold`, drift is flagged.
6. **Event mapping**: wrapper emits drift flag with score equal to the raw input value.
7. **Automatic recovery**: on drift, wrapper reinitializes Page-Hinkley state.

Practical behavior:

- Better for persistent shifts than isolated spikes.
- `threshold` is the main sensitivity knob (lower -> more sensitive).
- `mode` is useful when domain knowledge says drift is mostly one-sided.

Main parameters:

- `delta` (default `0.005`)
- `threshold` (default `50.0`)
- `min_instances` (default `30`)
- `alpha` (default `0.9999`)
- `mode` (`up`, `down`, `both`; default `both`)

## Factory and config

`DriftDetectorFactory.create(config)` in `src/factories/detector_factory.py` supports:

- `baseline` or `running_stats_baseline`
- `adwin`
- `ddm`
- `page_hinkley`

Disable modes:

- missing `drift_detector` block,
- `drift_detector.enabled: false`,
- `drift_detector.name: none` (or `null`).

Unknown names raise `ValueError`.

Example (baseline + visualiser):

```yaml
drift_detector:
  enabled: true
  name: baseline
  warmup_size: 50
  threshold_std: 3.0
  consecutive: 3
  min_std: 1e-8
  visualizer: true
```

## Engine integration details

Integration points in `src/engine/stream_engine.py`:

- `step(...)` computes model correctness against ground truth (`correct`) for metrics.
- For detector updates, label visibility depends on active learning:
  - with active policy: detector updates only when label was queried,
  - without active policy: detector updates on every labeled item.
- `_update_drift_detector(correct)` maps correctness to error (`0.0`/`1.0`) and calls `detector.update(...)`.
- If `drift_event.drift` is true, `_handle_drift()` is called.

Current drift response in `_handle_drift()`:

- `state.drift_mode = True`
- `calibrator.reset()` when calibrator exists
- `predictor.reset_adaptation_state()`

Note: detector reset is handled by detector implementations themselves on drift, avoiding double-reset behavior.

## Visualiser module

Location: `src/drift/detector_visualizer.py`

### Public API

- `replay_detector(detector, values) -> list[DriftEvent]`
  - Replays values through a detector and returns all events.
  - Does not auto-reset detector before/after replay.
- `DetectorVisualizer.plot(records, title=..., save_path=...)`
  - Builds diagnostics from engine `result.records`.
- `DetectorVisualizer.plot_events(events, steps=None, title=..., save_path=...)`
  - Builds diagnostics directly from event lists.

### Figure layout

Base panels (always):

1. detector score timeline (optional threshold line + warmup shading)
2. warning/drift event flags
3. cumulative drift count

Optional panel (only when `correct` exists in records):

4. rolling error rate (`window` argument, default `50`)

### Runtime usage

`src/main.py` behavior:

- reads `drift_detector.visualizer` from config,
- validates that visualizer cannot run without an enabled detector,
- after `engine.run(...)`, calls `DetectorVisualizer(detector=engine.drift_detector).plot(result.records)`,
- finally runs `plt.show()`.

## Tests and expected behavior

Detector tests:

- `tests/drift/test_baseline_detector.py`: warmup, warning-to-drift transition, reset behavior.
- `tests/drift/test_river_detectors.py`: shift detection and reset behavior for ADWIN/DDM/Page-Hinkley, plus DDM input thresholding.

Factory tests:

- `tests/drift/test_detector_factory.py`: detector creation, disable modes, unknown detector errors, invalid parameter validation.

Visualizer tests:

- `tests/drift/test_detector_visualizer.py`: figure type, panel count logic, and input validation.

Engine integration tests:

- `tests/engine/test_stream_engine.py`: queried-label detector updates, fully labeled updates without active policy, and no double detector reset after drift.

## Recommended developer workflow

1. Configure detector under `drift_detector` in YAML.
2. Run simulation via `src/main.py`.
3. Inspect `result.records[*]["drift_event"]` and `result.records[*]["correct"]`.
4. Use `DetectorVisualizer.plot(...)` for end-to-end diagnostics or `plot_events(...)` for custom replays.

## Extension guide

To add a new detector:

1. Implement `BaseDriftDetector` in `src/drift/`.
2. Export it in `src/drift/__init__.py`.
3. Add a branch in `DriftDetectorFactory.create(...)`.
4. Add focused tests under `tests/drift/` and at least one integration test in `tests/engine/` if runtime behavior changes.
5. Ensure `score` semantics are documented so visual diagnostics are interpretable.



