import math

import pytest

from scheduler.rank import (
    WEIGHTS,
    InferenceFeatures,
    JobFeatures,
    RankValidationError,
    compute_duration_only_ranks,
    compute_inference_ranks,
    compute_ranks,
    compute_training_ranks,
    compute_workload_ranks,
    sort_by_rank,
    sort_by_training_policy,
    sort_workloads_by_rank,
)


def job(job_id, **overrides):
    values = dict(T=10.0, R=0.1, M=128.0, G=5.0, C=50.0, P=1.0)
    values.update(overrides)
    return JobFeatures(job_id=job_id, **values)


def test_exact_paper_weights_and_directions():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    best = job("best", T=1, R=9, M=1, G=1, C=9, P=1)
    worst = job("worst", T=9, R=1, M=9, G=9, C=1, P=9)
    ranks = compute_ranks([best, worst])
    assert ranks == {"best": pytest.approx(1), "worst": pytest.approx(0.0)}


def test_duration_only_policy_is_an_explicit_spt_baseline():
    jobs = [job("long", T=30), job("short", T=5), job("middle", T=10)]
    ranks = compute_duration_only_ranks(jobs)
    assert ranks["short"] == pytest.approx(1.0)
    assert ranks["long"] == pytest.approx(0.0)
    assert [item.job_id for item in sort_by_training_policy(jobs, policy="duration_only")] == [
        "short",
        "middle",
        "long",
    ]
    assert compute_training_ranks(jobs, policy="six_feature") == compute_ranks(jobs)


def test_unknown_training_policy_fails_closed():
    with pytest.raises(RankValidationError, match="unknown training rank policy"):
        compute_training_ranks([job("a")], policy="oracle")


def test_identical_features_are_neutral_and_ties_are_deterministic():
    ranks = compute_ranks([job("z"), job("a")])
    assert ranks["z"] == pytest.approx(0.5)
    assert ranks["a"] == pytest.approx(0.5)
    assert [item.job_id for item in sort_by_rank([job("z"), job("a")])] == ["a", "z"]
    assert [item.job_id for item in sort_by_rank([job("z"), job("a")], reverse_order=True)] == [
        "a",
        "z",
    ]


def test_explicit_tie_breaker_is_used_without_reversing_ties():
    jobs = [job("a"), job("b")]
    keys = {"a": (2, "a"), "b": (1, "b")}
    assert [
        item.job_id for item in sort_by_rank(jobs, tie_breaker=lambda item: keys[item.job_id])
    ] == ["b", "a"]


@pytest.mark.parametrize(
    "jobs, message",
    [
        ([job("duplicate"), job("duplicate")], "duplicate job_id"),
        ([job("bad", T=math.nan)], "finite and > 0"),
        ([job("bad", R=0)], "finite and > 0"),
        ([job("bad", P=1.5)], "positive integer"),
        ([job("")], "non-empty string"),
    ],
)
def test_invalid_rank_inputs_are_rejected(jobs, message):
    with pytest.raises(RankValidationError, match=message):
        compute_ranks(jobs)


def test_empty_burst_is_valid_and_empty():
    assert compute_ranks([]) == {}
    assert sort_by_rank([]) == []


def inference(job_id, **overrides):
    values = dict(
        latency_slo_ms=100.0,
        predicted_latency_ms=50.0,
        request_rate_rps=10.0,
        memory_mib=256.0,
        cold_start_ms=100.0,
        priority=1.0,
    )
    values.update(overrides)
    return InferenceFeatures(job_id=job_id, **values)


def test_inference_policy_prioritizes_slo_pressure_and_demand():
    urgent = inference(
        "urgent",
        latency_slo_ms=50,
        predicted_latency_ms=45,
        request_rate_rps=100,
        cold_start_ms=500,
        priority=10,
    )
    background = inference(
        "background",
        latency_slo_ms=500,
        predicted_latency_ms=50,
        request_rate_rps=1,
        cold_start_ms=10,
        priority=1,
    )
    ranks = compute_inference_ranks([background, urgent])
    assert ranks["urgent"] > ranks["background"]
    assert [item.job_id for item in sort_workloads_by_rank([background, urgent])] == [
        "urgent",
        "background",
    ]


def test_workload_policy_rejects_mixed_training_and_inference_burst():
    with pytest.raises(RankValidationError, match="must not mix"):
        compute_workload_ranks([job("train"), inference("serve")])


def test_invalid_inference_features_are_rejected():
    with pytest.raises(RankValidationError, match="finite and > 0"):
        compute_inference_ranks([inference("bad", latency_slo_ms=0)])
