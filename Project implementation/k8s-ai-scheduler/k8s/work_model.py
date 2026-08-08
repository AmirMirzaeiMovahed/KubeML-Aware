"""Versioned synthetic-training model shared by generation and execution.

The article specifies feature meaning but not its original constants. These
constants are therefore explicit reproduction assumptions, not attributed to
the authors. Keeping the estimator and trainer budget in one dependency-free
module prevents generator/trainer/simulator drift.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

WORK_MODEL_VERSION = "2.0"
WORK_MODEL_ANNOTATION = "ml.scheduler/work-model-version"
WORK_MODEL_ENV = "ML_WORK_MODEL_VERSION"
BLAS_THREADS_ENV = "ML_NUM_THREADS"
BLAS_THREADS_ANNOTATION = "ml.scheduler/blas-threads"
REPRODUCTION_BLAS_THREADS = 1
MIB = 1024**2
FLOAT32_BYTES = 4
INITIAL_LOSS = 1.0
CONVERGENCE_THRESHOLD = 0.02

# Training-budget assumptions.  The scale lets R affect light, heavy and most
# I/O jobs while preserving a finite budget for deliberately slow convergence.
MIN_STEP_BUDGET = 64
MAX_STEP_BUDGET = 512
STEP_BUDGET_SCALE = 4.0

# Estimator calibration defaults.  Stage-specific hardware calibration can
# replace these values without changing the equations or termination logic.
MATMUL_SECONDS_AT_128 = 0.0005
GRADIENT_BANDWIDTH_MIB_PER_SECOND = 5000.0
CHECKPOINT_BANDWIDTH_MIB_PER_SECOND = 250.0
PARTITION_SYNC_LATENCY_SECONDS = 0.0005
MAX_CHECKPOINT_BYTES = 16 * MIB


@dataclass(frozen=True)
class WorkModelEstimate:
    model_version: str
    step_budget: int
    convergence_steps: int
    planned_steps: int
    termination_reason: str
    gradient_bytes: int
    checkpoint_bytes: int
    checkpoint_count: int
    matrix_compute_seconds: float
    gradient_update_seconds: float
    partition_sync_seconds: float
    checkpoint_seconds: float
    estimated_training_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return numeric


def validate_characteristics(*, R: float, M: int, G: float, C: int, P: int) -> None:
    _positive_finite("R", R)
    _positive_finite("G", G)
    for name, value in (("M", M), ("C", C), ("P", P)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if P > M:
        raise ValueError(f"P must not exceed matrix rows M; got P={P}, M={M}")


def convergence_step_count(R: float, threshold: float = CONVERGENCE_THRESHOLD) -> int:
    rate = _positive_finite("R", R)
    if not math.isfinite(threshold) or not 0 < threshold < INITIAL_LOSS:
        raise ValueError("threshold must be finite and in (0, 1)")
    # The trainer stops on loss < threshold, so exact equality needs one more step.
    return math.floor(math.log(INITIAL_LOSS / threshold) / rate) + 1


def step_budget(M: int, G: float) -> int:
    if isinstance(M, bool) or not isinstance(M, int) or M <= 0:
        raise ValueError(f"M must be a positive integer; got {M!r}")
    gradient_mib = _positive_finite("G", G)
    raw = math.ceil(STEP_BUDGET_SCALE * (M / 32.0) * (1.0 + gradient_mib / 10.0))
    return min(MAX_STEP_BUDGET, max(MIN_STEP_BUDGET, raw))


def gradient_buffer_bytes(G: float) -> int:
    gradient_mib = _positive_finite("G", G)
    return max(
        FLOAT32_BYTES,
        math.ceil(gradient_mib * MIB / FLOAT32_BYTES) * FLOAT32_BYTES,
    )


def checkpoint_payload_bytes(M: int) -> int:
    if isinstance(M, bool) or not isinstance(M, int) or M <= 0:
        raise ValueError(f"M must be a positive integer; got {M!r}")
    model_matrix_bytes = M * M * FLOAT32_BYTES
    return min(MAX_CHECKPOINT_BYTES, max(4096, model_matrix_bytes))


def partition_row_ranges(M: int, P: int) -> tuple[tuple[int, int], ...]:
    validate_characteristics(R=1.0, M=M, G=1.0, C=1, P=P)
    base, remainder = divmod(M, P)
    ranges = []
    start = 0
    for index in range(P):
        width = base + (1 if index < remainder else 0)
        ranges.append((start, start + width))
        start += width
    return tuple(ranges)


def estimate_work(
    *,
    R: float,
    M: int,
    G: float,
    C: int,
    P: int,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> WorkModelEstimate:
    validate_characteristics(R=R, M=M, G=G, C=C, P=P)
    budget = step_budget(M, G)
    convergence = convergence_step_count(R, threshold)
    planned = min(budget, convergence)
    reason = "converged" if convergence <= budget else "max_steps"
    gradient_bytes = gradient_buffer_bytes(G)
    checkpoint_bytes = checkpoint_payload_bytes(M)
    checkpoint_count = planned // C

    matrix_seconds = planned * MATMUL_SECONDS_AT_128 * (M / 128.0) ** 3
    # np.add reads and writes the gradient buffer once per step.
    gradient_seconds = (
        planned * 2.0 * (gradient_bytes / MIB) / GRADIENT_BANDWIDTH_MIB_PER_SECOND
    )
    partition_seconds = (
        planned
        * (P - 1)
        * (
            (gradient_bytes / MIB) / GRADIENT_BANDWIDTH_MIB_PER_SECOND
            + PARTITION_SYNC_LATENCY_SECONDS
        )
        if P > 1
        else 0.0
    )
    checkpoint_seconds = (
        checkpoint_count
        * (checkpoint_bytes / MIB)
        / CHECKPOINT_BANDWIDTH_MIB_PER_SECOND
    )
    total = matrix_seconds + gradient_seconds + partition_seconds + checkpoint_seconds
    return WorkModelEstimate(
        model_version=WORK_MODEL_VERSION,
        step_budget=budget,
        convergence_steps=convergence,
        planned_steps=planned,
        termination_reason=reason,
        gradient_bytes=gradient_bytes,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_count=checkpoint_count,
        matrix_compute_seconds=matrix_seconds,
        gradient_update_seconds=gradient_seconds,
        partition_sync_seconds=partition_seconds,
        checkpoint_seconds=checkpoint_seconds,
        estimated_training_seconds=max(total, 1e-6),
    )


def model_assumptions() -> dict[str, Any]:
    return {
        "model_version": WORK_MODEL_VERSION,
        "work_model_annotation": WORK_MODEL_ANNOTATION,
        "work_model_env": WORK_MODEL_ENV,
        "blas_threads_env": BLAS_THREADS_ENV,
        "blas_threads_annotation": BLAS_THREADS_ANNOTATION,
        "reproduction_blas_threads": REPRODUCTION_BLAS_THREADS,
        "initial_loss": INITIAL_LOSS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "min_step_budget": MIN_STEP_BUDGET,
        "max_step_budget": MAX_STEP_BUDGET,
        "step_budget_scale": STEP_BUDGET_SCALE,
        "matmul_seconds_at_128": MATMUL_SECONDS_AT_128,
        "gradient_bandwidth_mib_per_second": GRADIENT_BANDWIDTH_MIB_PER_SECOND,
        "checkpoint_bandwidth_mib_per_second": CHECKPOINT_BANDWIDTH_MIB_PER_SECOND,
        "partition_sync_latency_seconds": PARTITION_SYNC_LATENCY_SECONDS,
        "max_checkpoint_bytes": MAX_CHECKPOINT_BYTES,
    }
