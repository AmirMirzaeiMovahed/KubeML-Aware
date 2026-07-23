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
from typing import Any, Dict, Optional


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


CONVERGENCE_THRESHOLD = 0.02
MAX_MATRIX_DIMENSION = 4096
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
    return TrainingConfig(
        job_id=job_id,
        seed=_read_int("JOB_SEED", default_seed, minimum=0, maximum=2**63 - 1),
        # T is an estimate used by ranking; as in the article it does not stop training.
        T=_read_float("JOB_T", 10.0, minimum=0.000001, maximum=86400.0),
        R=_read_float("JOB_R", 0.05, minimum=0.000001, maximum=10.0),
        M=_read_int("JOB_M", 128, minimum=1, maximum=MAX_MATRIX_DIMENSION),
        G=_read_float("JOB_G", 5.0, minimum=0.000001, maximum=100000.0),
        C=_read_int("JOB_C", 50, minimum=1, maximum=10_000_000),
        P=_read_int("JOB_P", 1, minimum=1, maximum=1024),
    )


def compute_max_steps(config: TrainingConfig) -> int:
    return max(20, int((config.M / 32.0) * (1.0 + config.G / 10.0)))


def _handle_signal(signum: int, _frame: Any) -> None:
    global _shutdown_signal
    _shutdown_signal = signum


def run_training(config: TrainingConfig) -> Dict[str, Any]:
    """Allocate deterministic matrices and execute the article workload."""
    global _shutdown_signal
    _shutdown_signal = None
    rng = np.random.default_rng(config.seed)
    a = rng.random((config.M, config.M), dtype=np.float32)
    b = rng.random((config.M, config.M), dtype=np.float32)
    max_steps = compute_max_steps(config)
    loss = 1.0

    emit_event(
        "INITIALIZATION_COMPLETED",
        job_id=config.job_id,
        seed=config.seed,
        matrix_shape=[config.M, config.M],
        max_steps=max_steps,
        blas_threads=int(_THREADS),
    )
    started_at = time.time()
    emit_event("EXECUTION_STARTED", timestamp=started_at, job_id=config.job_id)

    step = 0
    while step < max_steps:
        if _shutdown_signal is not None:
            raise InterruptedError(f"received signal {_shutdown_signal}")
        for _ in range(config.P):
            _ = a @ b
            if config.P > 1:
                time.sleep(0.001 * config.P)

        loss *= math.exp(-config.R)
        step += 1

        if step % config.C == 0:
            io_delay = 0.002 * config.M / 100.0
            time.sleep(io_delay)
            emit_event(
                "CHECKPOINT",
                job_id=config.job_id,
                step=step,
                loss=loss,
                simulated_io_seconds=io_delay,
            )

        if loss < CONVERGENCE_THRESHOLD:
            emit_event("CONVERGED", job_id=config.job_id, step=step, loss=loss)
            break

    completed_at = time.time()
    result = {
        "job_id": config.job_id,
        "steps": step,
        "final_loss": loss,
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
