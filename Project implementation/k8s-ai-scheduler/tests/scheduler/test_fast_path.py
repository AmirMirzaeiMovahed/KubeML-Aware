from types import SimpleNamespace

import pytest

from scheduler.fast_path import (
    TrainingFastPathPolicy,
    balance_training_tail,
    pod_peak_cpu_cores,
)
from scheduler.pacing import MetricsSample
from scheduler.rank import InferenceFeatures, JobFeatures


def container(name="train", *, request="100m", limit="500m"):
    return SimpleNamespace(
        name=name,
        resources=SimpleNamespace(
            requests={"cpu": request} if request is not None else {},
            limits={"cpu": limit} if limit is not None else {},
        ),
    )


def pod(*containers, init_containers=(), overhead=None):
    return SimpleNamespace(
        spec=SimpleNamespace(
            containers=list(containers),
            init_containers=list(init_containers),
            overhead=overhead or {},
        )
    )


def training(job_id="train-a"):
    return JobFeatures(job_id, T=10, R=0.1, M=128, G=5, C=50, P=1)


class Feedback:
    def __init__(self, sample):
        self.value = sample

    def sample(self):
        return self.value


def policy(*, now=100.0):
    return TrainingFastPathPolicy(
        enabled=True,
        cpu_threshold=0.8,
        metrics_max_age=30,
        wall_time=lambda: now,
    )


def test_peak_cpu_uses_limits_and_kubernetes_init_semantics():
    workload = pod(
        container(limit="500m"),
        container("sidecar", request="100m", limit=None),
        init_containers=[container("init", request="900m", limit=None)],
        overhead={"cpu": "50m"},
    )
    assert pod_peak_cpu_cores(workload) == pytest.approx(0.95)


def test_training_tail_balance_preserves_prefix_and_reduces_makespan():
    jobs = [
        JobFeatures(f"job-{index}", duration, 1, 1, 1, 1, 1)
        for index, duration in enumerate((1, 2, 3, 4, 12, 20, 30))
    ]

    balanced, evidence = balance_training_tail(
        jobs,
        parallelism=2,
        protected_prefix=3,
    )

    assert balanced[:3] == jobs[:3]
    assert {job.job_id for job in balanced[3:]} == {
        job.job_id for job in jobs[3:]
    }
    assert evidence["predicted_makespan_after"] <= evidence["predicted_makespan_before"]


def test_training_fast_path_selects_only_fresh_bounded_headroom():
    pods = [pod(container(limit="200m")), pod(container(limit="300m"))]
    sample = MetricsSample(utilization=0.2, observed_at=95, allocatable_cores=4)
    decision = policy().evaluate(
        pods,
        [training("a"), training("b")],
        pacing_mode="none",
        reverse=False,
        feedback=Feedback(sample),
    )
    assert decision.selected is True
    assert decision.reason == "fresh_cpu_headroom"
    assert decision.projected_cpu_utilization == pytest.approx(0.325)
    assert decision.initial_release_count == 2
    assert decision.headroom_release_count == 2

    saturated = MetricsSample(utilization=0.79, observed_at=95, allocatable_cores=4)
    decision = policy().evaluate(
        pods,
        [training("a"), training("b")],
        pacing_mode="none",
        reverse=False,
        feedback=Feedback(saturated),
    )
    assert decision.selected is False
    assert decision.reason == "ranked_capacity_window"
    assert decision.initial_release_count == 1
    assert decision.headroom_release_count == 1


def test_contended_training_prefills_ranked_scheduler_queue():
    pods = [pod(container(limit="1")) for _ in range(4)]
    decision = policy().evaluate(
        pods,
        [training(str(index)) for index in range(4)],
        pacing_mode="none",
        reverse=False,
        feedback=Feedback(MetricsSample(0.3, 100, 4)),
    )
    assert decision.selected is True
    assert decision.reason == "ranked_queue_prefill"
    assert decision.available_cpu_headroom_cores == pytest.approx(2.0)
    assert decision.initial_release_count == 4
    assert decision.headroom_release_count == 2


def test_fast_path_fails_closed_for_nontraining_stale_or_unbounded_work():
    bounded = [pod(container(limit="200m"))]
    fresh = Feedback(MetricsSample(0.1, 100, 4))
    inference = InferenceFeatures(
        "serve",
        latency_slo_ms=100,
        predicted_latency_ms=40,
        request_rate_rps=5,
        memory_mib=128,
        cold_start_ms=200,
        priority=1,
    )
    assert policy().evaluate(
        bounded,
        [inference],
        pacing_mode="none",
        reverse=False,
        feedback=fresh,
    ).reason == "training_only"
    assert policy(now=200).evaluate(
        bounded,
        [training()],
        pacing_mode="none",
        reverse=False,
        feedback=fresh,
    ).reason == "stale_metrics"
    assert policy().evaluate(
        [pod(container(request=None, limit=None))],
        [training()],
        pacing_mode="none",
        reverse=False,
        feedback=fresh,
    ).reason.startswith("unbounded_cpu_demand")
    assert policy().evaluate(
        bounded,
        [training()],
        pacing_mode="fixed",
        reverse=False,
        feedback=fresh,
    ).reason == "explicit_pacing"
