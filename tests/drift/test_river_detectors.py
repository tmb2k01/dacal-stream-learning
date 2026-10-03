import math

import pytest

from drift import ADWINDetector, DDMDetector, PageHinkleyDetector
from drift.base import DriftEvent
from factories.detector_factory import DriftDetectorFactory


@pytest.mark.parametrize(
    ("name", "options", "before", "after", "max_delay"),
    [
        ("adwin", {"delta": 0.01}, [0.0] * 200, [1.0] * 300, 100),
        ("ddm", {"warm_start": 30}, [0.0, 1.0] * 100, [1.0] * 300, 100),
        ("page_hinkley", {"threshold": 10, "min_instances": 30}, [0.0] * 100, [1.0] * 300, 50),
    ],
)
def test_detector_reports_drift_after_error_shift_and_resets(
    name, options, before, after, max_delay
):
    detector = DriftDetectorFactory.create({"drift_detector": {"name": name, **options}})
    assert isinstance(detector, (ADWINDetector, DDMDetector, PageHinkleyDetector))
    original_inner = detector._inner

    before_events = [detector.update(value) for value in before]
    assert all(isinstance(event, DriftEvent) for event in before_events)
    assert all(not event.drift for event in before_events)

    shift_events = []
    for value in after:
        event = detector.update(value)
        shift_events.append(event)
        assert isinstance(event, DriftEvent)
        assert isinstance(event.warning, bool)
        assert math.isfinite(event.score)
        if event.drift:
            break

    assert shift_events[-1].drift
    assert len(shift_events) <= max_delay
    assert detector._inner is not original_inner

    first_after_reset = detector.update(0.0)
    assert first_after_reset.drift is False
    assert first_after_reset.warning is False


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("adwin", {"delta": 0.01}),
        ("ddm", {"warm_start": 30}),
        ("page_hinkley", {"threshold": 10, "min_instances": 30}),
    ],
)
def test_manual_reset_restarts_detector(name, options):
    config = {"drift_detector": {"name": name, **options}}
    detector = DriftDetectorFactory.create(config)
    fresh = DriftDetectorFactory.create(config)
    for value in [0.0, 1.0] * 20:
        detector.update(value)

    detector.reset()

    assert detector.update(0.0) == fresh.update(0.0)


def test_ddm_thresholds_continuous_input_to_binary_error():
    detector = DDMDetector(warm_start=30)

    assert detector.update(0.49).score == 0.0
    assert detector.update(0.5).score == 1.0
