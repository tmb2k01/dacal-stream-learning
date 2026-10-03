from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from drift import DetectorVisualizer, RunningStatsBaselineDetector
from drift.base import DriftEvent
from drift.detector_visualizer import replay_detector


def test_plot_events_returns_figure() -> None:
    detector = RunningStatsBaselineDetector(warmup_size=3)
    events = replay_detector(detector, [0.0, 0.1, -0.1, 2.0])

    figure = DetectorVisualizer(detector).plot_events(events)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 3


def test_plot_records_includes_rolling_error_panel() -> None:
    records = [
        {"step": 0, "drift_event": DriftEvent(False, False, 0.0), "correct": True},
        {"step": 1, "drift_event": DriftEvent(False, True, 2.0), "correct": False},
    ]

    figure = DetectorVisualizer(window=2).plot(records)

    assert len(figure.axes) == 4


def test_plot_events_rejects_mismatched_steps() -> None:
    events = [DriftEvent(False, False, 0.0)]

    with pytest.raises(ValueError, match="same length"):
        DetectorVisualizer().plot_events(events, steps=[0, 1])


def test_visualizer_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window"):
        DetectorVisualizer(window=0)
