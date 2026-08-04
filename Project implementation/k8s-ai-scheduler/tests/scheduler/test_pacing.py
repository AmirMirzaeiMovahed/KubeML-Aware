from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scheduler.pacing import (
    ClusterMetricsFeedback,
    FeedbackUnavailable,
    MetricsSample,
    Pacer,
    PacingError,
    PacingInterrupted,
    RealClusterFeedback,
)


class FakeClock:
    def __init__(self, wall=1_000.0):
        self.mono = 0.0
        self.wall = wall
        self.sleeps = []

    def monotonic(self):
        return self.mono

    def wall_time(self):
        return self.wall + self.mono

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.mono += duration


class SequenceFeedback:
    def __init__(self, samples):
        self.samples = iter(samples)

    def sample(self):
        value = next(self.samples)
        if isinstance(value, Exception):
            raise value
        return value


def test_none_and_fixed_pacing_use_monotonic_deadline():
    clock = FakeClock()
    Pacer(
        "none", monotonic=clock.monotonic, wall_time=clock.wall_time, sleep=clock.sleep
    ).wait()
    assert clock.mono == 0
    Pacer(
        "fixed",
        fixed_delay=1.5,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    ).wait()
    assert clock.mono == pytest.approx(1.5)


def test_pacing_stops_promptly_without_turning_shutdown_into_failure():
    clock = FakeClock()
    pacer = Pacer(
        "fixed",
        fixed_delay=5,
        poll_interval=0.25,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
        stop_requested=lambda: clock.mono >= 0.5,
    )
    with pytest.raises(PacingInterrupted, match="shutdown"):
        pacer.wait()
    assert clock.mono == pytest.approx(0.5)


def test_adaptive_requires_two_fresh_samples_below_hysteresis_watermark():
    clock = FakeClock()
    feedback = SequenceFeedback(
        [
            MetricsSample(0.81, 1_000.0),  # in hysteresis band: do not release
            MetricsSample(0.79, 1_000.5),
            MetricsSample(0.78, 1_001.0),
        ]
    )
    observed = []
    Pacer(
        "adaptive",
        feedback=feedback,
        cpu_threshold=0.85,
        hysteresis=0.05,
        max_wait=5,
        metrics_max_age=2,
        poll_interval=0.5,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
        on_sample=lambda sample, age: observed.append((sample.utilization, age)),
    ).wait()
    assert clock.mono == pytest.approx(1.0)
    assert len(observed) == 3


def test_adaptive_is_fail_closed_for_stale_or_unavailable_metrics():
    clock = FakeClock()

    class AlwaysUnavailable:
        def sample(self):
            raise FeedbackUnavailable("metrics API absent")

    pacer = Pacer(
        "adaptive",
        feedback=AlwaysUnavailable(),
        max_wait=1,
        metrics_max_age=1,
        poll_interval=0.25,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    with pytest.raises(PacingError, match="metrics unavailable"):
        pacer.wait()
    assert clock.mono == pytest.approx(1.0)

    clock = FakeClock()

    class Stale:
        def sample(self):
            return MetricsSample(0.0, 1.0)

    with pytest.raises(PacingError, match="stale"):
        Pacer(
            "adaptive",
            feedback=Stale(),
            max_wait=0.5,
            metrics_max_age=1,
            poll_interval=0.25,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        ).wait()


def test_adaptive_does_not_count_the_same_metrics_collection_twice():
    clock = FakeClock()

    class RepeatedTimestamp:
        def sample(self):
            return MetricsSample(0.1, 1_000.0)

    with pytest.raises(PacingError, match="timestamp has not advanced"):
        Pacer(
            "adaptive",
            feedback=RepeatedTimestamp(),
            max_wait=0.75,
            metrics_max_age=5,
            poll_interval=0.25,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        ).wait()


def ready_node(name, cpu="4"):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(unschedulable=False),
        status=SimpleNamespace(
            allocatable={"cpu": cpu},
            conditions=[SimpleNamespace(type="Ready", status="True")],
        ),
    )


def test_real_node_feedback_parses_usage_and_timestamp():
    timestamp = "2026-07-11T00:00:00Z"
    core = SimpleNamespace(read_node=lambda *_args, **_kwargs: ready_node("node-a", "4"))
    custom = SimpleNamespace(
        get_cluster_custom_object=lambda *_args, **_kwargs: {
            "usage": {"cpu": "1000m"},
            "timestamp": timestamp,
        }
    )
    feedback = RealClusterFeedback(core, custom, "node-a", api_retries=0)
    sample = feedback.sample()
    assert sample.utilization == pytest.approx(0.25)
    assert sample.observed_at == datetime(2026, 7, 11, tzinfo=timezone.utc).timestamp()


def test_cluster_feedback_is_aggregate_and_rejects_missing_nodes():
    core = SimpleNamespace(
        list_node=lambda **_kwargs: SimpleNamespace(
            items=[ready_node("a", "2"), ready_node("b", "2")]
        )
    )
    timestamp = "2026-07-11T00:00:00Z"
    metrics = {
        "items": [
            {"metadata": {"name": "a"}, "usage": {"cpu": "1"}, "timestamp": timestamp},
            {"metadata": {"name": "b"}, "usage": {"cpu": "500m"}, "timestamp": timestamp},
        ]
    }
    custom = SimpleNamespace(list_cluster_custom_object=lambda *_args, **_kwargs: metrics)
    feedback = ClusterMetricsFeedback(core, custom, api_retries=0)
    assert feedback.sample().utilization == pytest.approx(0.375)

    metrics["items"].pop()
    with pytest.raises(FeedbackUnavailable, match="metrics missing"):
        feedback.sample()
