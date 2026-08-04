"""Validated implementation of the ranking function from the paper.

The paper intentionally uses weights whose sum is 1.25.  They are kept
verbatim because multiplying all scores by a constant does not change the
ordering and changing them would no longer reproduce the published method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

FEATURES: Tuple[str, ...] = ("T", "R", "M", "G", "C", "P")
WEIGHTS: Dict[str, float] = {
    "T": 0.40,
    "R": 0.35,
    "M": 0.20,
    "G": 0.15,
    "C": 0.10,
    "P": 0.05,
}
LARGER_IS_BETTER = frozenset({"R", "C"})
SMALLER_IS_BETTER = frozenset({"T", "M", "G", "P"})


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


def validate_jobs(jobs: Iterable[JobFeatures]) -> List[JobFeatures]:
    """Return ``jobs`` as a list after strict, deterministic validation.

    All six inputs describe positive quantities in the paper.  Missing,
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
                raise RankValidationError(
                    f"job {job.job_id!r} feature {feature} must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise RankValidationError(
                    f"job {job.job_id!r} feature {feature} must be finite and > 0"
                )
        if not float(job.P).is_integer():
            raise RankValidationError(
                f"job {job.job_id!r} feature P must be a positive integer"
            )
    return validated


def _minmax_normalize(
    values: Mapping[str, float], larger_is_better: bool
) -> Dict[str, float]:
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
        normalized[feature] = _minmax_normalize(
            raw, larger_is_better=feature in LARGER_IS_BETTER
        )

    return {
        job.job_id: sum(
            WEIGHTS[feature] * normalized[feature][job.job_id]
            for feature in FEATURES
        )
        for job in validated
    }


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
