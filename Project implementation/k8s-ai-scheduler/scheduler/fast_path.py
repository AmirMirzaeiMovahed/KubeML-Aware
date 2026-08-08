"""Fail-closed fast-path admission for ML training bursts.

The production gate controller normally confirms useful execution before it
releases the next ranked Pod. That preserves rank under resource contention,
but it can leave kube-scheduler's queue empty between completions. This module
uses a fresh Metrics API sample and declared peak CPU demand to decide whether
the complete ranked queue may be safely prefilled. Gate removal never binds a
Pod; kube-scheduler remains the capacity and placement authority.

Only article-style :class:`~scheduler.rank.JobFeatures` are eligible.  Explicit
pacing, reversed ablations, inference workloads, missing resource declarations
and unavailable/stale metrics all keep the conservative ranked path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any, Callable, Optional, Sequence

from scheduler.kube import parse_cpu_quantity
from scheduler.pacing import FeedbackUnavailable, MetricsSample
from scheduler.rank import JobFeatures, WorkloadFeatures


class FastPathResourceError(ValueError):
    """Raised when Pod CPU demand cannot be bounded safely."""


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


@dataclass(frozen=True)
class FastPathDecision:
    selected: bool
    reason: str
    cpu_threshold: float
    current_cpu_utilization: Optional[float] = None
    burst_cpu_demand_cores: Optional[float] = None
    allocatable_cpu_cores: Optional[float] = None
    projected_cpu_utilization: Optional[float] = None
    metrics_age_seconds: Optional[float] = None
    available_cpu_headroom_cores: Optional[float] = None
    initial_release_count: int = 1
    headroom_release_count: int = 1

    def event(self, *, timestamp: float) -> dict[str, object]:
        return {
            "event": "fast_path_decision",
            "timestamp": timestamp,
            **asdict(self),
        }


def _resource_map(container: Any, field: str) -> dict[str, str]:
    resources = getattr(container, "resources", None)
    value = getattr(resources, field, None) if resources is not None else None
    if value is None and isinstance(resources, dict):
        value = resources.get(field)
    return dict(value or {})


def _container_peak_cpu(container: Any) -> float:
    limits = _resource_map(container, "limits")
    requests = _resource_map(container, "requests")
    raw = limits.get("cpu") or requests.get("cpu")
    if raw is None:
        name = getattr(container, "name", "unknown")
        raise FastPathResourceError(
            f"container {name!r} needs a CPU limit or request for fast-path admission"
        )
    cores = parse_cpu_quantity(str(raw))
    if cores <= 0:
        name = getattr(container, "name", "unknown")
        raise FastPathResourceError(
            f"container {name!r} declares non-positive peak CPU"
        )
    return cores


def pod_peak_cpu_cores(pod: Any) -> float:
    """Return a conservative Kubernetes-style peak CPU bound for one Pod."""

    spec = getattr(pod, "spec", None)
    containers = list(getattr(spec, "containers", None) or [])
    if not containers:
        raise FastPathResourceError("Pod has no application containers")
    application = sum(_container_peak_cpu(container) for container in containers)
    init_containers = list(getattr(spec, "init_containers", None) or [])
    initialization = max(
        (_container_peak_cpu(container) for container in init_containers),
        default=0.0,
    )
    overhead = getattr(spec, "overhead", None) or {}
    overhead_cpu = parse_cpu_quantity(str(overhead["cpu"])) if "cpu" in overhead else 0.0
    return max(application, initialization) + overhead_cpu


class TrainingFastPathPolicy:
    """Admit a bounded training queue when fresh CPU headroom proves progress."""

    def __init__(
        self,
        *,
        enabled: bool,
        cpu_threshold: float,
        metrics_max_age: float,
        wall_time: Callable[[], float],
    ):
        if not 0 < cpu_threshold <= 1:
            raise ValueError("fast-path CPU threshold must be in (0, 1]")
        if metrics_max_age <= 0:
            raise ValueError("metrics_max_age must be > 0")
        self.enabled = bool(enabled)
        self.cpu_threshold = float(cpu_threshold)
        self.metrics_max_age = float(metrics_max_age)
        self.wall_time = wall_time

    def _decision(self, selected: bool, reason: str, **values: object) -> FastPathDecision:
        return FastPathDecision(
            selected=selected,
            reason=reason,
            cpu_threshold=self.cpu_threshold,
            **values,
        )

    def evaluate(
        self,
        pods: Sequence[Any],
        jobs: Sequence[WorkloadFeatures],
        *,
        pacing_mode: str,
        reverse: bool,
        feedback: Optional[Any],
    ) -> FastPathDecision:
        if not self.enabled:
            return self._decision(False, "disabled")
        if pacing_mode != "none":
            return self._decision(False, "explicit_pacing")
        if reverse:
            return self._decision(False, "reversed_ablation")
        if not jobs or not all(isinstance(job, JobFeatures) for job in jobs):
            return self._decision(False, "training_only")
        if feedback is None:
            return self._decision(False, "metrics_feedback_unavailable")
        try:
            pod_demands = [pod_peak_cpu_cores(pod) for pod in pods]
            demand = sum(pod_demands)
        except (FastPathResourceError, ValueError, TypeError, KeyError) as exc:
            return self._decision(False, f"unbounded_cpu_demand: {exc}")
        try:
            sample: MetricsSample = feedback.sample()
        except FeedbackUnavailable as exc:
            return self._decision(False, f"metrics_unavailable: {exc}")
        age = max(0.0, self.wall_time() - sample.observed_at)
        capacity = sample.allocatable_cores
        common = {
            "current_cpu_utilization": sample.utilization,
            "burst_cpu_demand_cores": demand,
            "allocatable_cpu_cores": capacity,
            "metrics_age_seconds": age,
        }
        if age > self.metrics_max_age:
            return self._decision(False, "stale_metrics", **common)
        if capacity is None or capacity <= 0:
            return self._decision(False, "metrics_missing_capacity", **common)
        projected = sample.utilization + demand / capacity
        common["projected_cpu_utilization"] = projected
        available = max(0.0, (self.cpu_threshold - sample.utilization) * capacity)
        common["available_cpu_headroom_cores"] = available
        cumulative = 0.0
        release_count = 0
        for pod_demand in pod_demands:
            if cumulative + pod_demand > available:
                break
            cumulative += pod_demand
            release_count += 1
        # The conservative path always makes progress with the top-ranked Job,
        # even when no fresh headroom window can be proven.
        # Removing a scheduling gate only admits a Pod to kube-scheduler's
        # queue; it does not allocate CPU.  When at least one bounded training
        # Pod fits fresh headroom, prefill the complete ranked queue and let
        # kube-scheduler's normal feasibility checks protect node capacity.
        # This removes controller-induced completion-to-admission bubbles.
        headroom_release_count = max(1, release_count)
        initial_release_count = (
            len(pod_demands) if release_count > 0 else headroom_release_count
        )
        common["headroom_release_count"] = headroom_release_count
        common["initial_release_count"] = initial_release_count
        if projected <= self.cpu_threshold:
            return self._decision(True, "fresh_cpu_headroom", **common)
        if release_count > 0:
            return self._decision(True, "ranked_queue_prefill", **common)
        return self._decision(False, "ranked_capacity_window", **common)
