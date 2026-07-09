from drift import RunningStatsBaselineDetector


def test_baseline_detector_warmup_then_alerts_on_shift() -> None:
    detector = RunningStatsBaselineDetector(warmup_size=8, threshold_std=2.0, consecutive=2)

    # Warmup: no detection while the baseline is being estimated.
    for value in [0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1]:
        event = detector.update(value)
        assert event.drift is False

    first_spike = detector.update(3.0)
    assert first_spike.warning is True
    assert first_spike.drift is False

    second_spike = detector.update(3.2)
    assert second_spike.drift is True
    assert second_spike.warning is False


def test_baseline_detector_reset_clears_state() -> None:
    detector = RunningStatsBaselineDetector(warmup_size=4)

    for value in [0.0, 0.1, 0.0, -0.1, 2.5]:
        detector.update(value)

    detector.reset()

    event = detector.update(10.0)
    assert event.drift is False
    assert event.warning is False
    assert event.score == 0.0
