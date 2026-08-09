"""Calibratable single-node proxy simulator for scheduler experiments.

This remains a proxy, not a substitute for Kubernetes measurements.  Its work
model delegates step, gradient, partition and checkpoint semantics to the same
versioned module as ``k8s/train.py`` and workload generation.  Hardware scales
can be calibrated, while ``T`` is only blended in when explicitly requested.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from k8s.work_model import (  # noqa: E402
    CONVERGENCE_THRESHOLD,
    MATMUL_SECONDS_AT_128,
    WORK_MODEL_VERSION,
    estimate_work,
)
from scheduler.rank import JobFeatures, compute_ranks, sort_by_rank  # noqa: E402
from scheduler.tail_balance import balance_training_tail  # noqa: E402


@dataclass(frozen=True)
class TrainerWorkModel:
    """Hardware calibration parameters for the trainer-derived proxy."""

    model_version: str = WORK_MODEL_VERSION
    calibrated: bool = False
    calibration_id: str = "uncalibrated-defaults"
    matrix_reference: float = 128.0
    matmul_seconds_at_reference: float = MATMUL_SECONDS_AT_128
    gradient_scale: float = 1.0
    synchronization_scale: float = 1.0
    checkpoint_scale: float = 1.0
    estimated_time_weight: float = 0.0
    convergence_threshold: float = CONVERGENCE_THRESHOLD

    def validate(self) -> None:
        if self.model_version != WORK_MODEL_VERSION:
            raise ValueError(
                f"model_version must be {WORK_MODEL_VERSION!r}; "
                f"got {self.model_version!r}"
            )
        if not isinstance(self.calibrated, bool):
            raise ValueError("calibrated must be a boolean")
        if not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise ValueError("calibration_id must be a non-empty string")
        if self.calibrated and not self.calibration_id.startswith("sha256:"):
            raise ValueError("a calibrated model requires a sha256 calibration_id")
        for name in (
            "matrix_reference",
            "matmul_seconds_at_reference",
            "gradient_scale",
            "synchronization_scale",
            "checkpoint_scale",
            "convergence_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not 0 <= self.estimated_time_weight <= 1:
            raise ValueError("estimated_time_weight must be in [0, 1]")
        if not 0 < self.convergence_threshold < 1:
            raise ValueError("convergence_threshold must be in (0, 1)")


DEFAULT_WORK_MODEL = TrainerWorkModel()


class SimulationIncompleteError(RuntimeError):
    def __init__(self, message: str, unfinished_job_ids: Iterable[str]):
        super().__init__(message)
        self.unfinished_job_ids = tuple(sorted(unfinished_job_ids))


@dataclass(frozen=True)
class SimResult:
    job_id: str
    submission_t: float
    exec_start_t: float
    completion_t: float
    features: Optional[Mapping[str, float]] = None
    rank: Optional[float] = None

    @property
    def jct(self) -> float:
        return self.completion_t - self.submission_t


def trainer_step_count(
    job: JobFeatures, threshold: float = CONVERGENCE_THRESHOLD
) -> int:
    return estimate_work(
        R=job.R,
        M=int(job.M),
        G=job.G,
        C=int(job.C),
        P=int(job.P),
        threshold=threshold,
    ).planned_steps


def estimate_trainer_work(job: JobFeatures, model: TrainerWorkModel = DEFAULT_WORK_MODEL) -> float:
    """Estimate trainer wall work in seconds before CPU contention."""
    model.validate()
    for name in ("T", "R", "M", "G", "C", "P"):
        value = float(getattr(job, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"job {job.job_id}: {name} must be finite and > 0")
    for count_name in ("M", "C", "P"):
        if int(getattr(job, count_name)) != getattr(job, count_name):
            raise ValueError(f"job {job.job_id}: {count_name} must be an integer")
    estimate = estimate_work(
        R=job.R,
        M=int(job.M),
        G=job.G,
        C=int(job.C),
        P=int(job.P),
        threshold=model.convergence_threshold,
    )
    matmul = (
        estimate.planned_steps
        * model.matmul_seconds_at_reference
        * (job.M / model.matrix_reference) ** 3
    )
    gradient = estimate.gradient_update_seconds * model.gradient_scale
    synchronization = estimate.partition_sync_seconds * model.synchronization_scale
    checkpoint = estimate.checkpoint_seconds * model.checkpoint_scale
    trainer_estimate = matmul + gradient + synchronization + checkpoint
    blended = (
        (1.0 - model.estimated_time_weight) * trainer_estimate
        + model.estimated_time_weight * job.T
    )
    return max(blended, 1e-9)


def _validate_simulation_inputs(
    jobs: List[JobFeatures], *, n_cores: int, dt: float, alpha: float, max_time: float
) -> None:
    if not jobs:
        raise ValueError("jobs must not be empty")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("job IDs must be unique")
    if not isinstance(n_cores, int) or isinstance(n_cores, bool) or n_cores <= 0:
        raise ValueError("n_cores must be a positive integer")
    for name, value, allow_zero in (
        ("dt", dt, False),
        ("alpha", alpha, True),
        ("max_time", max_time, False),
    ):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            comparator = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{name} must be finite and {comparator}")


def _contention_efficiency(k: int, n_cores: int, alpha: float) -> float:
    if k <= n_cores:
        return 1.0
    return 1.0 / (1.0 + alpha * (k / n_cores - 1.0))


def _fair_share_step(
    active: Dict[str, float],
    n_cores: int,
    dt: float,
    t: float,
    completions: List[SimResult],
    submission_t: Mapping[str, float],
    exec_start: Mapping[str, float],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
) -> None:
    """Advance jobs and interpolate completion inside this time slice."""
    if not active:
        return
    k = len(active)
    rate = min(1.0, n_cores / k) * _contention_efficiency(k, n_cores, alpha)
    finished: List[tuple[str, float]] = []
    for job_id, remaining_before in list(active.items()):
        progress = rate * dt
        if remaining_before <= progress + 1e-12:
            completion = t + min(dt, max(0.0, remaining_before / rate))
            finished.append((job_id, completion))
        else:
            active[job_id] = remaining_before - progress
    for job_id, completion in finished:
        del active[job_id]
        info = metadata[job_id]
        completions.append(SimResult(
            job_id,
            submission_t[job_id],
            exec_start[job_id],
            completion,
            features=info["features"],
            rank=info["rank"],
        ))


def _metadata(jobs: List[JobFeatures]) -> Dict[str, Dict[str, Any]]:
    ranks = compute_ranks(jobs)
    return {
        job.job_id: {
            "features": {name: float(getattr(job, name)) for name in ("T", "R", "M", "G", "C", "P")},
            "rank": float(ranks[job.job_id]),
        }
        for job in jobs
    }


def run_default(
    jobs: List[JobFeatures],
    n_cores: int = 4,
    dt: float = 0.05,
    jitter: float = 0.67,
    seed: int = 0,
    *,
    mean_ilt: Optional[float] = None,
    launch_jitter_fraction: float = 0.0,
    alpha: float = 0.05,
    max_time: float = 100000.0,
    work_model: TrainerWorkModel = DEFAULT_WORK_MODEL,
) -> List[SimResult]:
    """Launch in creation order with a calibrated *inter-launch* interval.

    ``jitter`` is retained as the legacy parameter name, but now denotes mean
    ILT instead of a total launch window.  Set ``mean_ilt`` explicitly in new
    code.  ``launch_jitter_fraction`` adds bounded zero-mean variation while
    preserving positive gaps.
    """
    _validate_simulation_inputs(jobs, n_cores=n_cores, dt=dt, alpha=alpha, max_time=max_time)
    interval = jitter if mean_ilt is None else mean_ilt
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("mean_ilt must be finite and >= 0")
    if not 0 <= launch_jitter_fraction < 1:
        raise ValueError("launch_jitter_fraction must be in [0, 1)")
    work_model.validate()
    rng = random.Random(seed)

    launch_t: Dict[str, float] = {}
    clock = 0.0
    for index, job in enumerate(jobs):
        if index:
            variation = rng.uniform(-launch_jitter_fraction, launch_jitter_fraction)
            clock += interval * (1.0 + variation)
        launch_t[job.job_id] = clock
    submission_t = {job.job_id: 0.0 for job in jobs}
    pending = {job.job_id: estimate_trainer_work(job, work_model) for job in jobs}
    exec_start: Dict[str, float] = {}
    active: Dict[str, float] = {}
    completions: List[SimResult] = []
    metadata = _metadata(jobs)

    t = 0.0
    while pending or active:
        if t >= max_time:
            raise SimulationIncompleteError(
                f"simulation reached max_time={max_time}",
                set(pending) | set(active),
            )
        for job_id in list(pending):
            if launch_t[job_id] <= t + 1e-12:
                active[job_id] = pending.pop(job_id)
                exec_start[job_id] = t
        step_duration = min(dt, max_time - t)
        _fair_share_step(
            active, n_cores, step_duration, t, completions, submission_t, exec_start,
            metadata, alpha=alpha,
        )
        t += step_duration
    return sorted(completions, key=lambda result: result.completion_t)


def run_paced(
    jobs: List[JobFeatures],
    n_cores: int = 4,
    dt: float = 0.05,
    inherent_gap: float = 1.0,
    extra_delay: float = 0.0,
    mode: str = "fixed",
    cpu_threshold: float = 0.85,
    max_wait: float = 6.0,
    reverse: bool = False,
    *,
    initial_delay: float = 0.0,
    alpha: float = 0.05,
    max_time: float = 100000.0,
    work_model: TrainerWorkModel = DEFAULT_WORK_MODEL,
    balance_tail: bool = False,
    tail_window: int = 4,
    on_tail_plan: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> List[SimResult]:
    """Launch rank-ordered jobs using fixed or contention-feedback pacing.

    ``balance_tail`` reuses the production :func:`balance_training_tail` policy
    to reorder only the low-priority tail of the ranked burst so the longest
    Jobs no longer start last.  The high-priority prefix is frozen, so mean JCT
    is preserved while makespan and tail (p95) JCT improve.  It is disabled by
    default and skipped for reversed ablations so ``custom-baseline`` and the
    reversed control keep their exact article semantics.  When balancing runs,
    ``on_tail_plan`` (if given) receives the balancer's evidence dict so callers
    can persist an audit record of how the tail was reordered.
    """
    _validate_simulation_inputs(jobs, n_cores=n_cores, dt=dt, alpha=alpha, max_time=max_time)
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("mode must be 'fixed' or 'adaptive'")
    for name, value in (
        ("inherent_gap", inherent_gap),
        ("extra_delay", extra_delay),
        ("max_wait", max_wait),
        ("initial_delay", initial_delay),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0")
    if not 0 < cpu_threshold <= 1:
        raise ValueError("cpu_threshold must be in (0, 1]")
    if not isinstance(tail_window, int) or isinstance(tail_window, bool) or tail_window < 2:
        raise ValueError("tail_window must be an integer >= 2")
    work_model.validate()

    ordered = sort_by_rank(jobs, reverse_order=reverse)
    if balance_tail and not reverse:
        # Freeze the ``n_cores`` highest-priority Jobs (mean-JCT protection) and
        # rebalance only the ranked tail for makespan, exactly as the gate
        # controller does on its ``ranked_queue_prefill`` fast path.
        ordered, tail_plan = balance_training_tail(
            ordered,
            parallelism=n_cores,
            protected_prefix=n_cores,
            window=tail_window,
        )
        if on_tail_plan is not None:
            on_tail_plan(tail_plan)
    queue = [job.job_id for job in ordered]
    remaining = {job.job_id: estimate_trainer_work(job, work_model) for job in ordered}
    submission_t = {job.job_id: 0.0 for job in ordered}
    exec_start: Dict[str, float] = {}
    active: Dict[str, float] = {}
    completions: List[SimResult] = []
    metadata = _metadata(jobs)

    t = 0.0
    next_ready_t = initial_delay
    waiting_since: Optional[float] = None
    while queue or active:
        if t >= max_time:
            raise SimulationIncompleteError(
                f"simulation reached max_time={max_time}",
                set(queue) | set(active),
            )
        can_launch = bool(queue) and t >= next_ready_t - 1e-12
        if can_launch and mode == "adaptive":
            utilization = min(1.0, len(active) / n_cores)
            if waiting_since is None:
                waiting_since = t
            waited = t - waiting_since
            if utilization >= cpu_threshold and waited < max_wait:
                can_launch = False

        if can_launch:
            job_id = queue.pop(0)
            active[job_id] = remaining[job_id]
            exec_start[job_id] = t
            waiting_since = None
            gap = inherent_gap if mode == "adaptive" else inherent_gap + extra_delay
            next_ready_t = t + gap

        step_duration = min(dt, max_time - t)
        _fair_share_step(
            active, n_cores, step_duration, t, completions, submission_t, exec_start,
            metadata, alpha=alpha,
        )
        t += step_duration
    return sorted(completions, key=lambda result: result.completion_t)


def results_to_dicts(results: List[SimResult]) -> List[Dict[str, Any]]:
    return [
        {
            "job_id": result.job_id,
            "submission_time": result.submission_t,
            "execution_start_time": result.exec_start_t,
            "completion_time": result.completion_t,
            "jct_s": result.jct,
            "features": dict(result.features or {}),
            "rank": result.rank,
            "status": "completed",
            "node_name": "simulated-node",
            "error": None,
        }
        for result in results
    ]
