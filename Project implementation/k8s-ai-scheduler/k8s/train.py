"""Deterministic CPU training-workload simulator used by experiment Pods.

All lifecycle records are one-line JSON.  In particular, the scheduler and
collector consume records of the form::

    {"event":"EXECUTION_STARTED","timestamp":1700000000.0,"job_id":"..."}

``EXECUTION_STARTED`` is emitted *after* configuration validation and matrix
allocation, immediately before the first compute step.  This makes the marker
an honest useful-work boundary rather than a container-start proxy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from k8s.work_model import (
    CONVERGENCE_THRESHOLD,
    INITIAL_LOSS,
    PARTITION_SYNC_LATENCY_SECONDS,
    WORK_MODEL_ENV,
    WORK_MODEL_VERSION,
    estimate_work,
    partition_row_ranges,
    step_budget,
    validate_characteristics,
)


# NumPy/BLAS reads these variables during import.  Force a reproducible default
# while still allowing operators to choose a different count through the single
# documented ML_NUM_THREADS setting.
_THREADS = os.environ.get("ML_NUM_THREADS", "1")
if not _THREADS.isdigit() or int(_THREADS) < 1:
    print("ML_NUM_THREADS must be a positive integer", file=sys.stderr, flush=True)
    raise SystemExit(2)
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = _THREADS
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np  # noqa: E402  (must follow BLAS environment setup)


MAX_MATRIX_DIMENSION = 4096
MAX_GRADIENT_MIB = 256.0
MAX_PARTITIONS = 64
_shutdown_signal: Optional[int] = None


@dataclass(frozen=True)
class TrainingConfig:
    job_id: str
    seed: int
    T: float
    R: float
    M: int
    G: float
    C: int
    P: int


def emit_event(event: str, *, timestamp: Optional[float] = None, **fields: Any) -> None:
    """Emit one machine-readable record, rejecting non-finite JSON values."""
    record: Dict[str, Any] = {
        "event": event,
        "timestamp": time.time() if timestamp is None else float(timestamp),
        **fields,
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def _read_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric; got {raw!r}") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]; got {value!r}")
    return value


def _read_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        numeric = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc
    if not math.isfinite(numeric) or int(numeric) != numeric:
        raise ValueError(f"{name} must be an integer; got {raw!r}")
    value = int(numeric)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]; got {value!r}")
    return value


def _stable_seed(job_id: str) -> int:
    digest = hashlib.sha256(job_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def load_config() -> TrainingConfig:
    job_id = os.environ.get("JOB_ID", "unknown-job").strip()
    if not job_id or len(job_id) > 253:
        raise ValueError("JOB_ID must contain 1..253 characters")
    default_seed = _stable_seed(job_id)
    model_version = os.environ.get(WORK_MODEL_ENV, WORK_MODEL_VERSION)
    if model_version != WORK_MODEL_VERSION:
        raise ValueError(
            f"{WORK_MODEL_ENV}={model_version!r} does not match trainer "
            f"model {WORK_MODEL_VERSION!r}"
        )
    config = TrainingConfig(
        job_id=job_id,
        seed=_read_int("JOB_SEED", default_seed, minimum=0, maximum=2**63 - 1),
        # T is an estimate used by ranking; as in the article it does not stop training.
        T=_read_float("JOB_T", 10.0, minimum=0.000001, maximum=86400.0),
        R=_read_float("JOB_R", 0.05, minimum=0.000001, maximum=10.0),
        M=_read_int("JOB_M", 128, minimum=1, maximum=MAX_MATRIX_DIMENSION),
        G=_read_float("JOB_G", 5.0, minimum=0.000001, maximum=MAX_GRADIENT_MIB),
        C=_read_int("JOB_C", 50, minimum=1, maximum=10_000_000),
        P=_read_int("JOB_P", 1, minimum=1, maximum=MAX_PARTITIONS),
    )
    validate_characteristics(
        R=config.R, M=config.M, G=config.G, C=config.C, P=config.P
    )
    return config


def compute_max_steps(config: TrainingConfig) -> int:
    return step_budget(config.M, config.G)


def _checkpoint_path(job_id: str) -> Path:
    root = Path(os.environ.get("ML_CHECKPOINT_DIR", "/tmp"))
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16]
    return root / f"ml-checkpoint-{digest}.bin"


def _write_checkpoint(path: Path, matrix: np.ndarray, payload_bytes: int) -> float:
    """Write and fsync a bounded model-state payload, returning wall duration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = memoryview(matrix).cast("B")
    if not payload:
        raise ValueError("checkpoint source matrix is empty")
    started = time.perf_counter()
    with path.open("wb", buffering=0) as checkpoint:
        remaining = payload_bytes
        while remaining:
            chunk = payload[: min(remaining, len(payload))]
            written = checkpoint.write(chunk)
            if written <= 0:
                raise OSError("checkpoint write made no progress")
            remaining -= written
        os.fsync(checkpoint.fileno())
    return time.perf_counter() - started


def _handle_signal(signum: int, _frame: Any) -> None:
    global _shutdown_signal
    _shutdown_signal = signum


def run_training(config: TrainingConfig) -> Dict[str, Any]:
    """Execute physical compute, gradient, synchronization and checkpoint work."""
    global _shutdown_signal
    _shutdown_signal = None
    rng = np.random.default_rng(config.seed)
    a = rng.random((config.M, config.M), dtype=np.float32)
    b = rng.random((config.M, config.M), dtype=np.float32)
    result_matrix = np.empty_like(a)
    model = estimate_work(
        R=config.R, M=config.M, G=config.G, C=config.C, P=config.P
    )
    gradient_elements = model.gradient_bytes // np.dtype(np.float32).itemsize
    gradient = rng.random(gradient_elements, dtype=np.float32)
    synchronization_buffer = np.empty_like(gradient) if config.P > 1 else None
    partitions = partition_row_ranges(config.M, config.P)
    checkpoint_path = _checkpoint_path(config.job_id)
    loss = INITIAL_LOSS

    emit_event(
        "INITIALIZATION_COMPLETED",
        job_id=config.job_id,
        seed=config.seed,
        matrix_shape=[config.M, config.M],
        work_model=model.to_dict(),
        estimated_T=config.T,
        estimated_T_error_seconds=config.T - model.estimated_training_seconds,
        partition_rows=[list(item) for item in partitions],
        blas_threads=int(_THREADS),
    )
    started_at = time.time()
    emit_event("EXECUTION_STARTED", timestamp=started_at, job_id=config.job_id)

    step = 0
    checkpoint_count = 0
    checkpoint_seconds = 0.0
    while step < model.step_budget:
        if _shutdown_signal is not None:
            raise InterruptedError(f"received signal {_shutdown_signal}")
        for start, end in partitions:
            np.matmul(a[start:end], b, out=result_matrix[start:end])

        np.add(gradient, np.float32(config.R * 1e-4), out=gradient)
        if synchronization_buffer is not None:
            # P partitions imply P-1 peer synchronizations.  Copying G MiB for
            # each peer makes both G and P observable physical work instead of
            # merely multiplying a sleep constant.
            for _peer in range(config.P - 1):
                np.copyto(synchronization_buffer, gradient)
            time.sleep(PARTITION_SYNC_LATENCY_SECONDS * (config.P - 1))

        loss *= math.exp(-config.R)
        step += 1

        if step % config.C == 0:
            io_duration = _write_checkpoint(
                checkpoint_path, result_matrix, model.checkpoint_bytes
            )
            checkpoint_count += 1
            checkpoint_seconds += io_duration
            emit_event(
                "CHECKPOINT",
                job_id=config.job_id,
                step=step,
                loss=loss,
                bytes_written=model.checkpoint_bytes,
                io_seconds=io_duration,
                path=str(checkpoint_path),
            )

        if loss < CONVERGENCE_THRESHOLD:
            emit_event("CONVERGED", job_id=config.job_id, step=step, loss=loss)
            break

    completed_at = time.time()
    termination_reason = "converged" if loss < CONVERGENCE_THRESHOLD else "max_steps"
    result = {
        "job_id": config.job_id,
        "steps": step,
        "final_loss": loss,
        "termination_reason": termination_reason,
        "work_model_version": WORK_MODEL_VERSION,
        "step_budget": model.step_budget,
        "convergence_steps": model.convergence_steps,
        "gradient_bytes": model.gradient_bytes,
        "checkpoint_bytes": model.checkpoint_bytes,
        "checkpoint_count": checkpoint_count,
        "checkpoint_seconds": checkpoint_seconds,
        "duration_seconds": completed_at - started_at,
    }
    emit_event("EXECUTION_COMPLETED", timestamp=completed_at, **result)
    return result


def main() -> int:
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _handle_signal)
    try:
        config = load_config()
        emit_event("JOB_CONFIG", job_id=config.job_id, config=asdict(config))
        run_training(config)
        return 0
    except (ValueError, MemoryError, InterruptedError) as exc:
        emit_event(
            "EXECUTION_FAILED",
            job_id=os.environ.get("JOB_ID", "unknown-job"),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    except Exception as exc:  # preserve a machine-readable terminal marker
        emit_event(
            "EXECUTION_FAILED",
            job_id=os.environ.get("JOB_ID", "unknown-job"),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
