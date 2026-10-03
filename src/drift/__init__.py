from drift.base import BaseDriftDetector
from drift.baseline_detector import RunningStatsBaselineDetector
from drift.detector_visualizer import DetectorVisualizer, replay_detector
from drift.river_detectors import (
    ADWINDetector,
    DDMDetector,
    PageHinkleyDetector,
)

__all__ = [
    "BaseDriftDetector",
    "RunningStatsBaselineDetector",
    "ADWINDetector",
    "DDMDetector",
    "PageHinkleyDetector",
    "DetectorVisualizer",
    "replay_detector",
]


def __getattr__(name: str):
    if name in {"DetectorVisualizer", "replay_detector"}:
        from drift.detector_visualizer import DetectorVisualizer, replay_detector

        return {"DetectorVisualizer": DetectorVisualizer, "replay_detector": replay_detector}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
