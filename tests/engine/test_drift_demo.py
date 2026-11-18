from __future__ import annotations

import pytest

from experiments.cifar10c_suite import UPPER_BASELINE_NAME, upper_baseline
from experiments.drift_demo import (
    NOISES,
    RAMP_LEVELS,
    all_configurations,
    build_run_config,
    build_scenarios,
)


def test_scenarios_cover_both_kinds_for_every_noise():
    scenarios = build_scenarios(list(NOISES))
    assert len(scenarios) == 2 * len(NOISES)
    assert {sc.kind for sc in scenarios} == {"sudden", "gradual"}
    assert len({sc.key for sc in scenarios}) == len(scenarios)


@pytest.mark.parametrize("noise", list(NOISES))
def test_sudden_and_gradual_have_same_length_and_end_severity(noise):
    sudden, gradual = build_scenarios([noise], clean_len=300, stage_len=100)
    assert sudden.length == gradual.length == 800
    assert sudden.segments[-1][2] == gradual.segments[-1][2] == RAMP_LEVELS[-1]
    assert [seg[2] for seg in gradual.segments[1:]] == list(RAMP_LEVELS)
    assert len(sudden.segments) == 2 and len(gradual.segments) == 1 + len(RAMP_LEVELS)


def test_run_config_has_disjoint_consecutive_segment_starts():
    (sc,) = build_scenarios(["gaussian_noise"])[1:]
    cfg = build_run_config(
        sc, upper_baseline(1), 1, 800, "data/CIFAR-10-C", "data", "ckpt.pt", stage_len=100
    )
    segments = cfg["dataset"]["segments"]
    assert segments[0]["start"] == 800
    for prev, nxt in zip(segments, segments[1:], strict=False):
        assert nxt["start"] == prev["start"] + prev["count"]
    assert "severity" not in segments[0]


def test_upper_baseline_labels_whole_segment_after_every_shift():
    (sc,) = build_scenarios(["shot_noise"])[1:]
    cfg = build_run_config(sc, upper_baseline(0), 0, 0, "d/CIFAR-10-C", "d", "c.pt", 100)
    assert cfg["drift_detector"]["name"] == "oracle"
    assert cfg["active_policy"]["rate"] == 0.0
    assert cfg["active_policy"]["burst_rate"] == 1.0
    assert cfg["engine"]["burst_rate"] == 1.0
    assert cfg["engine"]["drift_burst"] > 100


def test_all_configurations_order_and_uniqueness():
    names = [item["name"] for item in all_configurations(0)]
    assert names[0].startswith("Lower Baseline")
    assert names[1] == UPPER_BASELINE_NAME
    assert len(names) == len(set(names)) == 12
