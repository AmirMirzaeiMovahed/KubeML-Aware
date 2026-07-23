import math

import pytest

from scheduler.rank import (
    JobFeatures,
    RankValidationError,
    compute_ranks,
    sort_by_rank,
)


def job(job_id, **overrides):
    values = dict(T=10.0, R=0.1, M=128.0, G=5.0, C=50.0, P=1.0)
    values.update(overrides)
    return JobFeatures(job_id=job_id, **values)


def test_exact_paper_weights_and_directions():
    best = job("best", T=1, R=9, M=1, G=1, C=9, P=1)
    worst = job("worst", T=9, R=1, M=9, G=9, C=1, P=9)
    ranks = compute_ranks([best, worst])
    assert ranks == {"best": pytest.approx(1.25), "worst": pytest.approx(0.0)}


def test_identical_features_are_neutral_and_ties_are_deterministic():
    ranks = compute_ranks([job("z"), job("a")])
    assert ranks["z"] == pytest.approx(0.625)
    assert ranks["a"] == pytest.approx(0.625)
    assert [item.job_id for item in sort_by_rank([job("z"), job("a")])] == ["a", "z"]
    assert [
        item.job_id for item in sort_by_rank([job("z"), job("a")], reverse_order=True)
    ] == ["a", "z"]


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

