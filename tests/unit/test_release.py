"""Canary weights for nginx and the watcher's rollback rule."""

import pytest

from legalrag.release import CanaryHealth, Limits, render_split, should_roll_back


def test_split_sends_the_requested_share_to_the_canary():
    conf = render_split(5)
    assert "5% canary;" in conf and "* stable;" in conf
    assert conf.startswith('split_clients "${request_id}" $release {')


def test_zero_and_hundred_percent_are_single_targets():
    assert "canary" not in render_split(0)
    assert "* canary;" in render_split(100) and "stable" not in render_split(100)


@pytest.mark.parametrize("percent", [-1, 101])
def test_percent_out_of_range_is_refused(percent):
    with pytest.raises(ValueError):
        render_split(percent)


def test_a_healthy_canary_stays():
    health = CanaryHealth(requests=50, error_ratio=0.0, p95_seconds=1.2)
    assert should_roll_back(health, Limits()) == []


def test_too_little_traffic_is_not_judged():
    health = CanaryHealth(requests=3, error_ratio=1.0, p95_seconds=30.0)
    assert should_roll_back(health, Limits(min_requests=10)) == []


def test_errors_or_latency_over_the_limits_roll_back():
    errors = CanaryHealth(requests=40, error_ratio=0.25, p95_seconds=1.0)
    slow = CanaryHealth(requests=40, error_ratio=0.0, p95_seconds=9.0)
    assert should_roll_back(errors, Limits()) == ["5xx ratio 0.250 > 0.05"]
    assert should_roll_back(slow, Limits()) == ["p95 9.00 s > 5.0 s"]


def test_missing_latency_does_not_count_as_slow():
    health = CanaryHealth(requests=40, error_ratio=0.0, p95_seconds=None)
    assert should_roll_back(health, Limits()) == []
