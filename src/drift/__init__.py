from drift.base import BaseDriftDetector, DriftEvent
from drift.baseline_detector import RunningStatsBaselineDetector
from drift.embedding_detectors import (
    AdaptiveMMDDriftDetector,
    C2STDriftDetector,
    CUSUMDriftDetector,
    KSDriftDetector,
    MMDDriftDetector,
    OnlinePCA,
)
from drift.oracle_detector import OracleDriftDetector
from drift.river_detectors import (
    ADWINDetector,
    DDMDetector,
    PageHinkleyDetector,
)

__all__ = [
    "BaseDriftDetector",
    "DriftEvent",
    "RunningStatsBaselineDetector",
    "ADWINDetector",
    "DDMDetector",
    "PageHinkleyDetector",
    "MMDDriftDetector",
    "AdaptiveMMDDriftDetector",
    "KSDriftDetector",
    "C2STDriftDetector",
    "CUSUMDriftDetector",
    "OnlinePCA",
    "OracleDriftDetector",
    "DetectorVisualizer",
    "replay_detector",
]


def __getattr__(name: str):
    # Lazy import keeps matplotlib out of the import path of headless runs.
    if name in {"DetectorVisualizer", "replay_detector"}:
        from drift.detector_visualizer import DetectorVisualizer, replay_detector

        return {"DetectorVisualizer": DetectorVisualizer, "replay_detector": replay_detector}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
