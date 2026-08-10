"""Validated implementation of the ranking function from the article."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

FEATURES: Tuple[str, ...] = ("T", "R", "M", "G", "C", "P")
TRAINING_RANK_POLICIES: Tuple[str, ...] = ("six_feature", "duration_only")
WEIGHTS: Dict[str, float] = {
    "T": 0.32,
    "R": 0.28,
    "M": 0.16,
    "G": 0.12,
    "C": 0.08,
    "P": 0.04,
}
LARGER_IS_BETTER = frozenset({"R", "C"})
SMALLER_IS_BETTER = frozenset({"T", "M", "G", "P"})

INFERENCE_FEATURES: Tuple[str, ...] = (
    "slo_pressure",
    "request_rate_rps",
    "predicted_latency_ms",
    "memory_mib",
    "cold_start_ms",
    "priority",
)
INFERENCE_WEIGHTS: Dict[str, float] = {
    "slo_pressure": 0.30,
    "request_rate_rps": 0.25,
    "predicted_latency_ms": 0.15,
    "memory_mib": 0.10,
    "cold_start_ms": 0.10,
    "priority": 0.10,
}
INFERENCE_LARGER_IS_BETTER = frozenset(
    {"slo_pressure", "request_rate_rps", "cold_start_ms", "priority"}
)


class RankValidationError(ValueError):
    """Raised when a burst cannot be ranked without ambiguous input."""


@dataclass
class JobFeatures:
    job_id: str
    T: float
    R: float
    M: float
    G: float
    C: float
    P: float


@dataclass(frozen=True)
class InferenceFeatures:
    """Measured or profiled inputs for latency-sensitive inference ordering."""

    job_id: str
    latency_slo_ms: float
    predicted_latency_ms: float
    request_rate_rps: float
    memory_mib: float
    cold_start_ms: float
    priority: float

    @property
    def slo_pressure(self) -> float:
        return self.predicted_latency_ms / self.latency_slo_ms


WorkloadFeatures = Union[JobFeatures, InferenceFeatures]


def validate_jobs(jobs: Iterable[JobFeatures]) -> List[JobFeatures]:
    """Return ``jobs`` as a list after strict, deterministic validation.

    All six inputs describe positive quantities in the article. Missing,
    non-numeric, non-finite, zero and negative values are rejected rather than
    silently receiving an advantageous normalized score.  ``P`` is a count and
    therefore must be an integer, though it remains a float-compatible field to
    preserve the original public dataclass API.
    """

    validated = list(jobs)
    seen = set()
    for index, job in enumerate(validated):
        if not isinstance(job, JobFeatures):
            raise RankValidationError(f"jobs[{index}] is not a JobFeatures instance")
        if not isinstance(job.job_id, str) or not job.job_id.strip():
            raise RankValidationError(f"jobs[{index}].job_id must be a non-empty string")
        if job.job_id in seen:
            raise RankValidationError(f"duplicate job_id: {job.job_id!r}")
        seen.add(job.job_id)

        for feature in FEATURES:
            value = getattr(job, feature)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RankValidationError(f"job {job.job_id!r} feature {feature} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise RankValidationError(
                    f"job {job.job_id!r} feature {feature} must be finite and > 0"
                )
        if not float(job.P).is_integer():
            raise RankValidationError(f"job {job.job_id!r} feature P must be a positive integer")
    return validated


def validate_inference_jobs(
    jobs: Iterable[InferenceFeatures],
) -> List[InferenceFeatures]:
    """Validate inference inputs before burst-relative normalization."""

    validated = list(jobs)
    seen = set()
    fields = (
        "latency_slo_ms",
        "predicted_latency_ms",
        "request_rate_rps",
        "memory_mib",
        "cold_start_ms",
        "priority",
    )
    for index, job in enumerate(validated):
        if not isinstance(job, InferenceFeatures):
            raise RankValidationError(
                f"inference jobs[{index}] is not an InferenceFeatures instance"
            )
        if not isinstance(job.job_id, str) or not job.job_id.strip():
            raise RankValidationError(f"inference jobs[{index}].job_id must be a non-empty string")
        if job.job_id in seen:
            raise RankValidationError(f"duplicate job_id: {job.job_id!r}")
        seen.add(job.job_id)
        for feature in fields:
            value = getattr(job, feature)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RankValidationError(
                    f"inference job {job.job_id!r} feature {feature} must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise RankValidationError(
                    f"inference job {job.job_id!r} feature {feature} must be finite and > 0"
                )
    return validated


def _minmax_normalize(values: Mapping[str, float], larger_is_better: bool) -> Dict[str, float]:
    """Normalize one feature to ``[0, 1]`` with 1 always most desirable."""

    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {job_id: 0.5 for job_id in values}
    if larger_is_better:
        return {job_id: (value - lo) / (hi - lo) for job_id, value in values.items()}
    return {job_id: (hi - value) / (hi - lo) for job_id, value in values.items()}


def compute_ranks(jobs: Iterable[JobFeatures]) -> Dict[str, float]:
    """Compute the exact burst-relative weighted rank from Eq. (1) and (2)."""

    validated = validate_jobs(jobs)
    if not validated:
        return {}

    normalized: Dict[str, Dict[str, float]] = {}
    for feature in FEATURES:
        raw = {job.job_id: float(getattr(job, feature)) for job in validated}
        normalized[feature] = _minmax_normalize(raw, larger_is_better=feature in LARGER_IS_BETTER)

    return {
        job.job_id: sum(WEIGHTS[feature] * normalized[feature][job.job_id] for feature in FEATURES)
        for job in validated
    }


def compute_duration_only_ranks(jobs: Iterable[JobFeatures]) -> Dict[str, float]:
    """Return the burst-relative shortest-estimated-duration baseline.

    This policy intentionally uses only ``T`` and exists as an explicit SPT/
    weighted-SJF-style ablation.  It shares validation, normalization and tie
    semantics with the six-feature policy so a live experiment can isolate
    what the other five terms add instead of comparing unlike controllers.
    """

    validated = validate_jobs(jobs)
    if not validated:
        return {}
    raw = {job.job_id: float(job.T) for job in validated}
    return _minmax_normalize(raw, larger_is_better=False)


def compute_training_ranks(
    jobs: Iterable[JobFeatures], *, policy: str = "six_feature"
) -> Dict[str, float]:
    """Compute one of the registered training-ranking policies."""

    materialized = list(jobs)
    if policy == "six_feature":
        return compute_ranks(materialized)
    if policy == "duration_only":
        return compute_duration_only_ranks(materialized)
    raise RankValidationError(
        f"unknown training rank policy {policy!r}; expected one of {TRAINING_RANK_POLICIES}"
    )


def compute_inference_ranks(
    jobs: Iterable[InferenceFeatures],
) -> Dict[str, float]:
    """Rank one inference burst using an explicit latency/SLO policy.

    SLO pressure, demand, cold-start cost and operator priority are urgent when
    larger. Predicted latency and memory are efficiency terms and therefore
    prefer smaller values. All terms are burst-relative and weights sum to one.
    """

    validated = validate_inference_jobs(jobs)
    if not validated:
        return {}
    normalized: Dict[str, Dict[str, float]] = {}
    for feature in INFERENCE_FEATURES:
        raw = {job.job_id: float(getattr(job, feature)) for job in validated}
        normalized[feature] = _minmax_normalize(
            raw, larger_is_better=feature in INFERENCE_LARGER_IS_BETTER
        )
    return {
        job.job_id: sum(
            INFERENCE_WEIGHTS[feature] * normalized[feature][job.job_id]
            for feature in INFERENCE_FEATURES
        )
        for job in validated
    }


def compute_workload_ranks(
    jobs: Iterable[WorkloadFeatures],
    *,
    training_policy: str = "six_feature",
) -> Dict[str, float]:
    """Dispatch to the article or inference policy for a homogeneous burst."""

    materialized = list(jobs)
    if not materialized:
        return {}
    kinds = {type(job) for job in materialized}
    if kinds == {JobFeatures}:
        return compute_training_ranks(  # type: ignore[arg-type]
            materialized, policy=training_policy
        )
    if kinds == {InferenceFeatures}:
        return compute_inference_ranks(materialized)  # type: ignore[arg-type]
    raise RankValidationError("a scheduling burst must not mix training and inference workloads")


TieBreaker = Callable[[JobFeatures], object]


def sort_by_rank(
    jobs: Iterable[JobFeatures],
    reverse_order: bool = False,
    tie_breaker: Optional[TieBreaker] = None,
) -> List[JobFeatures]:
    """Return a deterministic best-first (or ablation worst-first) ordering.

    Rank direction changes for the reversed-order ablation, but tie ordering
    deliberately remains ascending.  The default tie-breaker is ``job_id``;
    Kubernetes callers pass ``(creation_timestamp, uid, name)`` so equal ranks
    do not depend on API list order.
    """

    validated = validate_jobs(jobs)
    ranks = compute_ranks(validated)
    tie_breaker = tie_breaker or (lambda job: job.job_id)

    decorated = []
    for job in validated:
        tie = tie_breaker(job)
        if not isinstance(tie, tuple):
            tie = (tie,)
        rank_key = ranks[job.job_id] if reverse_order else -ranks[job.job_id]
        decorated.append((rank_key, tie, job))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]


def sort_by_training_policy(
    jobs: Iterable[JobFeatures],
    *,
    policy: str = "six_feature",
    reverse_order: bool = False,
    tie_breaker: Optional[TieBreaker] = None,
) -> List[JobFeatures]:
    """Sort a training burst under a named, reproducible ranking policy."""

    validated = validate_jobs(jobs)
    ranks = compute_training_ranks(validated, policy=policy)
    tie_breaker = tie_breaker or (lambda job: job.job_id)
    decorated = []
    for job in validated:
        tie = tie_breaker(job)
        if not isinstance(tie, tuple):
            tie = (tie,)
        rank_key = ranks[job.job_id] if reverse_order else -ranks[job.job_id]
        decorated.append((rank_key, tie, job))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]


WorkloadTieBreaker = Callable[[WorkloadFeatures], object]


def sort_workloads_by_rank(
    jobs: Iterable[WorkloadFeatures],
    reverse_order: bool = False,
    tie_breaker: Optional[WorkloadTieBreaker] = None,
    training_policy: str = "six_feature",
) -> List[WorkloadFeatures]:
    """Return deterministic policy-specific ordering for one homogeneous burst."""

    materialized = list(jobs)
    ranks = compute_workload_ranks(materialized, training_policy=training_policy)
    tie_breaker = tie_breaker or (lambda job: job.job_id)
    decorated = []
    for job in materialized:
        tie = tie_breaker(job)
        if not isinstance(tie, tuple):
            tie = (tie,)
        rank_key = ranks[job.job_id] if reverse_order else -ranks[job.job_id]
        decorated.append((rank_key, tie, job))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]
