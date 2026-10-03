import pytest

from drift import ADWINDetector, DDMDetector, PageHinkleyDetector, RunningStatsBaselineDetector
from factories.detector_factory import DriftDetectorFactory


@pytest.mark.parametrize(
    ("name", "options", "detector_type", "attribute", "expected"),
    [
        ("baseline", {"warmup_size": 8}, RunningStatsBaselineDetector, "warmup_size", 8),
        ("running_stats_baseline", {}, RunningStatsBaselineDetector, "warmup_size", 50),
        ("adwin", {"delta": 0.01}, ADWINDetector, "delta", 0.01),
        ("ddm", {"warm_start": 12}, DDMDetector, "warm_start", 12),
        ("page_hinkley", {"threshold": 10.0}, PageHinkleyDetector, "threshold", 10.0),
    ],
)
def test_factory_creates_configured_detectors(name, options, detector_type, attribute, expected):
    detector = DriftDetectorFactory.create({"drift_detector": {"name": name, **options}})

    assert isinstance(detector, detector_type)
    assert getattr(detector, attribute) == expected


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"drift_detector": {"enabled": False, "name": "adwin"}},
        {"drift_detector": {"name": "none"}},
    ],
)
def test_factory_allows_disabling_detection(config):
    assert DriftDetectorFactory.create(config) is None


def test_factory_rejects_unknown_detector():
    with pytest.raises(ValueError, match="Unsupported drift detector"):
        DriftDetectorFactory.create({"drift_detector": {"name": "unknown"}})


@pytest.mark.parametrize(
    "name, options",
    [
        ("baseline", {"warmup_size": 1}),
        ("adwin", {"delta": 0}),
        ("ddm", {"drift_threshold": 1, "warning_threshold": 2}),
        ("page_hinkley", {"mode": "sideways"}),
    ],
)
def test_factory_rejects_invalid_detector_settings(name, options):
    with pytest.raises(ValueError):
        DriftDetectorFactory.create({"drift_detector": {"name": name, **options}})
