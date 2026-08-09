"""Pure, dependency-free tail balancing for ranked training bursts.

These helpers minimize predicted makespan (and, secondarily, completion-time
sum) by reordering only a bounded ranked *tail*, leaving the high-priority
prefix untouched.  They intentionally depend on nothing beyond
:class:`~scheduler.rank.JobFeatures` and the standard library so both the
Kubernetes gate controller (:mod:`scheduler.fast_path`) and the offline
simulator (:mod:`sim.simulate`) can share one validated implementation without
importing the Kubernetes client.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

from scheduler.rank import JobFeatures


def _predicted_schedule(
    jobs: Sequence[JobFeatures], parallelism: int
) -> tuple[float, float]:
    """Return predicted completion-time sum and makespan for list scheduling."""

    lanes = [0.0] * parallelism
    completion_sum = 0.0
    for job in jobs:
        lane = min(range(parallelism), key=lambda index: (lanes[index], index))
        lanes[lane] += job.T
        completion_sum += lanes[lane]
    return completion_sum, max(lanes, default=0.0)


def balance_training_tail(
    jobs: Sequence[JobFeatures],
    *,
    parallelism: int,
    protected_prefix: int,
    window: int = 4,
) -> tuple[list[JobFeatures], dict[str, object]]:
    """Balance a small ranked tail without disturbing its priority prefix."""

    ordered = list(jobs)
    if parallelism < 2 or protected_prefix < 1 or window < 2:
        return ordered, {"selected": False, "reason": "not_applicable"}
    tail_size = min(window, len(ordered) - protected_prefix)
    if tail_size < 2:
        return ordered, {"selected": False, "reason": "tail_too_small"}
    prefix = ordered[:-tail_size]
    original_tail = ordered[-tail_size:]
    before_sum, before_makespan = _predicted_schedule(ordered, parallelism)
    candidates = ([*prefix, *candidate] for candidate in permutations(original_tail))
    balanced = min(
        candidates,
        key=lambda candidate: (
            _predicted_schedule(candidate, parallelism)[1],
            _predicted_schedule(candidate, parallelism)[0],
            tuple(job.job_id for job in candidate[-tail_size:]),
        ),
    )
    after_sum, after_makespan = _predicted_schedule(balanced, parallelism)
    changed = [job.job_id for job in balanced] != [job.job_id for job in ordered]
    return balanced, {
        "selected": changed,
        "reason": "balanced" if changed else "already_balanced",
        "parallelism": parallelism,
        "protected_prefix": len(prefix),
        "original_tail": [job.job_id for job in original_tail],
        "balanced_tail": [job.job_id for job in balanced[-tail_size:]],
        "predicted_completion_sum_before": before_sum,
        "predicted_completion_sum_after": after_sum,
        "predicted_makespan_before": before_makespan,
        "predicted_makespan_after": after_makespan,
    }
