"""Benchmark and validate hardware calibration for the proxy simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from k8s.work_model import (
    CHECKPOINT_BANDWIDTH_MIB_PER_SECOND,
    CONVERGENCE_THRESHOLD,
    GRADIENT_BANDWIDTH_MIB_PER_SECOND,
    MIB,
    PARTITION_SYNC_LATENCY_SECONDS,
    WORK_MODEL_VERSION,
)
from sim.simulate import TrainerWorkModel

CALIBRATION_SCHEMA_VERSION = "1.0"
CALIBRATION_KIND = "ml-trainer-hardware-calibration"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return numeric


def _median_duration(
    operation: Callable[[], None], *, repeats: int, warmups: int = 2
) -> tuple[float, list[float]]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(max(time.perf_counter() - started, 1e-9))
    return statistics.median(samples), samples


def benchmark_calibration(
    *,
    repeats: int = 7,
    matrix_reference: int = 128,
    gradient_mib: float = 16.0,
    checkpoint_mib: float = 1.0,
    checkpoint_parent: Path | None = None,
) -> dict[str, Any]:
    """Measure trainer primitives and return a self-identifying calibration."""

    if (
        isinstance(matrix_reference, bool)
        or not isinstance(matrix_reference, int)
        or matrix_reference <= 0
    ):
        raise ValueError("matrix_reference must be a positive integer")
    gradient_mib = _positive_number("gradient_mib", gradient_mib)
    checkpoint_mib = _positive_number("checkpoint_mib", checkpoint_mib)

    # Import only after callers have configured their BLAS environment.
    if "numpy" in sys.modules:
        raise RuntimeError(
            "NumPy is already loaded; run calibration in a fresh process so "
            "BLAS thread controls are auditable"
        )
    threads = os.environ.get("ML_NUM_THREADS", "1")
    if not threads.isdigit() or int(threads) <= 0:
        raise ValueError("ML_NUM_THREADS must be a positive integer")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = threads
    os.environ["OMP_DYNAMIC"] = "FALSE"
    import numpy as np

    from k8s.train import _write_checkpoint

    rng = np.random.default_rng(20250711)
    a = rng.random((matrix_reference, matrix_reference), dtype=np.float32)
    b = rng.random((matrix_reference, matrix_reference), dtype=np.float32)
    matrix_output = np.empty_like(a)
    matmul_median, matmul_samples = _median_duration(
        lambda: np.matmul(a, b, out=matrix_output), repeats=repeats
    )

    gradient_elements = max(1, math.ceil(gradient_mib * MIB / 4))
    gradient = rng.random(gradient_elements, dtype=np.float32)
    sync_buffer = np.empty_like(gradient)
    gradient_median, gradient_samples = _median_duration(
        lambda: np.add(gradient, np.float32(1e-5), out=gradient), repeats=repeats
    )

    def synchronize() -> None:
        np.copyto(sync_buffer, gradient)
        time.sleep(PARTITION_SYNC_LATENCY_SECONDS)

    sync_median, sync_samples = _median_duration(synchronize, repeats=repeats)

    checkpoint_bytes = max(4096, math.ceil(checkpoint_mib * MIB))
    checkpoint_dimension = max(1, math.ceil(math.sqrt(checkpoint_bytes / 4)))
    checkpoint_matrix = rng.random(
        (checkpoint_dimension, checkpoint_dimension), dtype=np.float32
    )
    temporary_parent = str(checkpoint_parent) if checkpoint_parent else None
    with tempfile.TemporaryDirectory(
        prefix="ml-calibration-", dir=temporary_parent
    ) as temporary:
        checkpoint_path = Path(temporary) / "checkpoint.bin"
        checkpoint_median, checkpoint_samples = _median_duration(
            lambda: _write_checkpoint(
                checkpoint_path, checkpoint_matrix, checkpoint_bytes
            ),
            repeats=repeats,
            warmups=1,
        )

    modeled_gradient_seconds = 2.0 * gradient_mib / GRADIENT_BANDWIDTH_MIB_PER_SECOND
    modeled_sync_seconds = (
        gradient_mib / GRADIENT_BANDWIDTH_MIB_PER_SECOND
        + PARTITION_SYNC_LATENCY_SECONDS
    )
    modeled_checkpoint_seconds = (
        checkpoint_bytes / MIB / CHECKPOINT_BANDWIDTH_MIB_PER_SECOND
    )
    parameters = {
        "matrix_reference": float(matrix_reference),
        "matmul_seconds_at_reference": matmul_median,
        "gradient_scale": gradient_median / modeled_gradient_seconds,
        "synchronization_scale": sync_median / modeled_sync_seconds,
        "checkpoint_scale": checkpoint_median / modeled_checkpoint_seconds,
        "estimated_time_weight": 0.0,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
    }
    evidence = {
        "work_model_version": WORK_MODEL_VERSION,
        "parameters": parameters,
        "benchmark": {
            "repeats": repeats,
            "matrix_reference": matrix_reference,
            "gradient_mib": gradient_mib,
            "checkpoint_bytes": checkpoint_bytes,
            "samples_seconds": {
                "matmul": matmul_samples,
                "gradient_update": gradient_samples,
                "partition_sync": sync_samples,
                "checkpoint_fsync": checkpoint_samples,
            },
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "ml_num_threads": os.environ.get("ML_NUM_THREADS", "1"),
            "numpy_version": np.__version__,
        },
    }
    calibration_id = f"sha256:{_canonical_sha256(evidence)}"
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "calibration_id": calibration_id,
        **evidence,
    }


def validate_calibration_document(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema_version")
    if document.get("kind") != CALIBRATION_KIND:
        raise ValueError("invalid calibration kind")
    if document.get("work_model_version") != WORK_MODEL_VERSION:
        raise ValueError("calibration work model version is stale")
    calibration_id = document.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.startswith("sha256:"):
        raise ValueError("calibration_id must be sha256-qualified")
    evidence = {
        key: document.get(key)
        for key in ("work_model_version", "parameters", "benchmark", "environment")
    }
    expected_id = f"sha256:{_canonical_sha256(evidence)}"
    if calibration_id != expected_id:
        raise ValueError("calibration_id does not match calibration contents")
    parameters = document.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TypeError("calibration parameters are missing")
    for name in (
        "matrix_reference",
        "matmul_seconds_at_reference",
        "gradient_scale",
        "synchronization_scale",
        "checkpoint_scale",
        "convergence_threshold",
    ):
        _positive_number(f"parameters.{name}", parameters.get(name))
    estimated_weight = parameters.get("estimated_time_weight")
    if (
        isinstance(estimated_weight, bool)
        or not isinstance(estimated_weight, (int, float))
        or not math.isfinite(float(estimated_weight))
        or not 0 <= float(estimated_weight) <= 1
    ):
        raise ValueError("parameters.estimated_time_weight must be in [0, 1]")


def calibrated_model(document: Mapping[str, Any]) -> TrainerWorkModel:
    validate_calibration_document(document)
    parameters = document["parameters"]
    return TrainerWorkModel(
        calibrated=True,
        calibration_id=str(document["calibration_id"]),
        matrix_reference=float(parameters["matrix_reference"]),
        matmul_seconds_at_reference=float(parameters["matmul_seconds_at_reference"]),
        gradient_scale=float(parameters["gradient_scale"]),
        synchronization_scale=float(parameters["synchronization_scale"]),
        checkpoint_scale=float(parameters["checkpoint_scale"]),
        estimated_time_weight=float(parameters["estimated_time_weight"]),
        convergence_threshold=float(parameters["convergence_threshold"]),
    )


def load_calibrated_model(path: Path) -> TrainerWorkModel:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError("calibration document must be a JSON object")
    return calibrated_model(document)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--matrix-reference", type=int, default=128)
    parser.add_argument("--gradient-mib", type=float, default=16.0)
    parser.add_argument("--checkpoint-mib", type=float, default=1.0)
    parser.add_argument("--checkpoint-parent", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    document = benchmark_calibration(
        repeats=args.repeats,
        matrix_reference=args.matrix_reference,
        gradient_mib=args.gradient_mib,
        checkpoint_mib=args.checkpoint_mib,
        checkpoint_parent=args.checkpoint_parent,
    )
    validate_calibration_document(document)
    _atomic_json(args.output, document)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "calibration_id": document["calibration_id"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
