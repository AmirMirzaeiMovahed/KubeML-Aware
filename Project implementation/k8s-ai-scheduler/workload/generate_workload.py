"""Deterministic synthetic workload generation for the paper reproduction.

The article defines four workload categories but does not publish the exact
sampling distributions.  ``CATEGORY_RANGES`` therefore remains an explicit,
versioned project assumption.  The article's half-load scenario is represented
by halving only matrix dimension ``M``; the duration estimate ``T`` is then
recomputed from the versioned work model.  The selected profile and every
assumption are written to ``jobs.json`` so a run can be audited later.

The command produces two byte-stable manifest sets from one sampled burst:
``pods_default`` and ``pods_custom``.  Existing non-empty output directories
are rejected.  ``--overwrite`` is intentionally allowed only for directories
that contain this generator's sentinel file, preventing accidental deletion of
unrelated user data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from k8s.work_model import (  # noqa: E402
    BLAS_THREADS_ANNOTATION,
    BLAS_THREADS_ENV,
    REPRODUCTION_BLAS_THREADS,
    WORK_MODEL_ANNOTATION,
    WORK_MODEL_ENV,
    WORK_MODEL_VERSION,
    estimate_work,
    model_assumptions,
)
from scheduler.constants import EXECUTION_CONTAINER_ANNOTATION  # noqa: E402
from scheduler.rank import JobFeatures, compute_ranks  # noqa: E402


WORKLOAD_SCHEMA_VERSION = "1.1"
OUTPUT_SENTINEL = ".ml-scheduler-workload"
FEATURE_NAMES = ("T", "R", "M", "G", "C", "P")
PRODUCTION_RESOURCE_REQUIREMENTS = {
    "requests": {"cpu": "100m", "memory": "128Mi"},
    "limits": {"cpu": "1", "memory": "1Gi"},
}

CATEGORY_RANGES: Dict[str, Dict[str, tuple[float, float]]] = {
    # T is derived from R/M/G/C/P by the versioned work model rather than
    # sampled independently. R is the per-step loss reduction rate,
    # M is square matrix dimension, G is gradient size (MiB),
    # C: checkpoint interval (steps), P: model partition count.
    "light": {
        "R": (0.08, 0.15), "M": (64, 128),
        "G": (1, 5), "C": (50, 100), "P": (1, 1),
    },
    "heavy": {
        "R": (0.03, 0.08), "M": (512, 1024),
        "G": (20, 50), "C": (20, 50), "P": (2, 4),
    },
    "io_intensive": {
        "R": (0.05, 0.10), "M": (128, 256),
        "G": (5, 15), "C": (5, 15), "P": (1, 2),
    },
    "slow_converging": {
        "R": (0.01, 0.03), "M": (128, 256),
        "G": (5, 15), "C": (30, 60), "P": (1, 2),
    },
}

CATEGORY_WEIGHTS = {
    "light": 0.30,
    "heavy": 0.25,
    "io_intensive": 0.25,
    "slow_converging": 0.20,
}

LOAD_PROFILES: Dict[str, Dict[str, Any]] = {
    "normal": {
        "feature_scales": {},
        "target_estimated_work_ratio": 1.0,
        "calibration_feature": None,
        "calibration_method": "identity",
        "description": "Article normal-load profile; sampled features are unchanged.",
    },
    "half": {
        "feature_scales": {},
        "target_estimated_work_ratio": 0.5,
        "calibration_feature": "M",
        "calibration_method": "deterministic-global-bisection-v1",
        "description": (
            "Article half-load profile: one deterministic global M scale targets "
            "50% aggregate estimated physical work; R, G, C, and P are unchanged "
            "and T is re-estimated."
        ),
    },
}

HALF_LOAD_TARGET_RATIO = 0.5
HALF_LOAD_RATIO_TOLERANCE = 0.01
HALF_LOAD_CALIBRATION_ITERATIONS = 80

LABEL_KEYS = {
    "run_id": "ml.scheduler/run-id",
    "scenario": "ml.scheduler/scenario",
    "config": "ml.scheduler/config",
    "repetition": "ml.scheduler/repetition",
}

RUN_ANNOTATIONS = {
    "run_id": "ml.scheduler/run-id",
    "pacing_mode": "ml.scheduler/pacing-mode",
    "fixed_delay_seconds": "ml.scheduler/fixed-delay-seconds",
    "reverse": "ml.scheduler/reverse",
    "expected_jobs": "ml.scheduler/expected-jobs",
}

FEATURE_ANNOTATIONS = {
    "T": "ml.scheduler/estimated-training-time",
    "R": "ml.scheduler/loss-reduction-rate",
    "M": "ml.scheduler/matrix-size",
    "G": "ml.scheduler/gradient-update-size",
    "C": "ml.scheduler/checkpoint-interval",
    "P": "ml.scheduler/model-partitions",
}

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
_K8S_LABEL_VALUE_RE = re.compile(r"^[A-Za-z0-9](?:[-A-Za-z0-9_.]*[A-Za-z0-9])?$")


def _require_finite(name: str, value: float, *, minimum: float = 0.0) -> float:
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}; got {value!r}")
    return value


def _validate_job(job: JobFeatures) -> None:
    if not job.job_id or len(job.job_id) > 63 or not _DNS_LABEL_RE.fullmatch(job.job_id):
        raise ValueError(f"job_id is not a valid Kubernetes name: {job.job_id!r}")
    _require_finite("T", job.T, minimum=0.000001)
    _require_finite("R", job.R, minimum=0.000001)
    _require_finite("M", job.M, minimum=1)
    _require_finite("G", job.G, minimum=0.000001)
    _require_finite("C", job.C, minimum=1)
    _require_finite("P", job.P, minimum=1)
    if int(job.M) != job.M:
        raise ValueError(f"M must be an integer matrix dimension; got {job.M!r}")
    if int(job.C) != job.C:
        raise ValueError(f"C must be an integer checkpoint interval; got {job.C!r}")
    if int(job.P) != job.P:
        raise ValueError(f"P must be an integer count; got {job.P!r}")


def _validate_label_value(name: str, value: str) -> str:
    value = str(value)
    if not value or len(value) > 63 or not _K8S_LABEL_VALUE_RE.fullmatch(value):
        raise ValueError(f"{name} is not a valid Kubernetes label value: {value!r}")
    return value


def deterministic_job_id(category: str, seed: int, index: int) -> str:
    """Return a stable Kubernetes-safe ID derived from seed and job index."""
    if category not in CATEGORY_RANGES:
        raise ValueError(f"unknown category: {category!r}")
    if index < 0:
        raise ValueError("index must be >= 0")
    # Include a short digest so negative/large seeds remain compact and safe.
    digest = hashlib.sha256(f"{seed}:{index}:{category}".encode("utf-8")).hexdigest()[:8]
    prefix = category.replace("_", "-")[:32]
    return f"{prefix}-{index:03d}-{digest}"


def deterministic_job_seed(run_seed: int, job_id: str) -> int:
    """Derive a stable, distinct NumPy seed for each job in one run."""
    digest = hashlib.sha256(f"{run_seed}:{job_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def _sample_category(category: str, rng: random.Random, *, seed: int, index: int) -> JobFeatures:
    ranges = CATEGORY_RANGES[category]
    p_low, p_high = (int(ranges["P"][0]), int(ranges["P"][1]))
    R = round(rng.uniform(*ranges["R"]), 4)
    M = rng.randint(math.ceil(ranges["M"][0]), math.floor(ranges["M"][1]))
    G = round(rng.uniform(*ranges["G"]), 2)
    C = rng.randint(math.ceil(ranges["C"][0]), math.floor(ranges["C"][1]))
    P = rng.randint(p_low, p_high)
    estimate = estimate_work(R=R, M=M, G=G, C=C, P=P)
    job = JobFeatures(
        job_id=deterministic_job_id(category, seed, index),
        T=round(estimate.estimated_training_seconds, 6),
        R=R,
        M=M,
        G=G,
        C=C,
        P=P,
    )
    _validate_job(job)
    return job


def category_for_job(job: JobFeatures) -> str:
    prefix = job.job_id.rsplit("-", 2)[0].replace("-", "_")
    if prefix not in CATEGORY_RANGES:
        raise ValueError(f"cannot infer category from deterministic job id {job.job_id!r}")
    return prefix


def estimated_burst_seconds(jobs: Sequence[JobFeatures]) -> float:
    if not jobs:
        raise ValueError("jobs must not be empty")
    return sum(
        estimate_work(
            R=job.R,
            M=int(job.M),
            G=job.G,
            C=int(job.C),
            P=int(job.P),
        ).estimated_training_seconds
        for job in jobs
    )


def calibrate_half_load(
    jobs: Sequence[JobFeatures],
    *,
    target_ratio: float = HALF_LOAD_TARGET_RATIO,
) -> tuple[List[JobFeatures], Dict[str, Any]]:
    """Scale M globally so aggregate versioned work is as close as possible to 50%."""

    if not jobs:
        raise ValueError("jobs must not be empty")
    if not math.isfinite(target_ratio) or not 0 < target_ratio < 1:
        raise ValueError("target_ratio must be finite and in (0, 1)")
    baseline_seconds = estimated_burst_seconds(jobs)
    target_seconds = baseline_seconds * target_ratio
    minimum_scale = max(float(job.P) / float(job.M) for job in jobs)

    def candidate(scale: float) -> tuple[List[JobFeatures], float]:
        scaled_jobs: List[JobFeatures] = []
        for job in jobs:
            matrix_dimension = max(int(job.P), round(float(job.M) * scale))
            estimate = estimate_work(
                R=job.R,
                M=matrix_dimension,
                G=job.G,
                C=int(job.C),
                P=int(job.P),
            )
            scaled_jobs.append(
                replace(
                    job,
                    M=matrix_dimension,
                    T=round(estimate.estimated_training_seconds, 6),
                )
            )
        return scaled_jobs, estimated_burst_seconds(scaled_jobs)

    low, high = minimum_scale, 1.0
    candidates: List[tuple[float, List[JobFeatures], float]] = []
    for scale in (low, high):
        scaled_jobs, seconds = candidate(scale)
        candidates.append((scale, scaled_jobs, seconds))
    for _ in range(HALF_LOAD_CALIBRATION_ITERATIONS):
        middle = (low + high) / 2.0
        scaled_jobs, seconds = candidate(middle)
        candidates.append((middle, scaled_jobs, seconds))
        if seconds < target_seconds:
            low = middle
        else:
            high = middle

    selected_scale, scaled_jobs, adjusted_seconds = min(
        candidates,
        key=lambda item: (abs(item[2] - target_seconds), -item[0]),
    )
    achieved_ratio = adjusted_seconds / baseline_seconds
    evidence = {
        "method": LOAD_PROFILES["half"]["calibration_method"],
        "work_model_version": WORK_MODEL_VERSION,
        "scaled_feature": "M",
        "selected_scale": selected_scale,
        "target_estimated_work_ratio": target_ratio,
        "achieved_estimated_work_ratio": achieved_ratio,
        "absolute_ratio_error": abs(achieved_ratio - target_ratio),
        "ratio_tolerance": HALF_LOAD_RATIO_TOLERANCE,
        "within_tolerance": abs(achieved_ratio - target_ratio)
        <= HALF_LOAD_RATIO_TOLERANCE,
        "baseline_estimated_seconds": baseline_seconds,
        "adjusted_estimated_seconds": adjusted_seconds,
    }
    return scaled_jobs, evidence


def generate_burst(n_jobs: int, seed: Optional[int] = None, load: str = "normal") -> List[JobFeatures]:
    """Generate a deterministic category-weighted burst.

    ``seed=None`` is retained for API compatibility, but the generated seed is
    then intentionally non-reproducible.  Experiment runners always supply an
    explicit integer seed.
    """
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs <= 0:
        raise ValueError(f"n_jobs must be a positive integer; got {n_jobs!r}")
    if load not in LOAD_PROFILES:
        raise ValueError(f"load must be one of {sorted(LOAD_PROFILES)}; got {load!r}")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError(f"seed must be an integer or None; got {seed!r}")

    actual_seed = seed if seed is not None else random.SystemRandom().randrange(0, 2**63)
    rng = random.Random(actual_seed)
    categories = list(CATEGORY_WEIGHTS)
    weights = [CATEGORY_WEIGHTS[name] for name in categories]
    jobs: List[JobFeatures] = []
    for index in range(n_jobs):
        category = rng.choices(categories, weights=weights, k=1)[0]
        job = _sample_category(category, rng, seed=actual_seed, index=index)
        _validate_job(job)
        jobs.append(job)

    if load == "half":
        jobs, _evidence = calibrate_half_load(jobs)

    if len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("deterministic job ID collision")
    return jobs


def load_profile_evidence(
    jobs: Sequence[JobFeatures], *, seed: int, load: str
) -> Dict[str, Any]:
    if load == "normal":
        total = estimated_burst_seconds(jobs)
        return {
            "method": "identity",
            "work_model_version": WORK_MODEL_VERSION,
            "selected_scale": 1.0,
            "target_estimated_work_ratio": 1.0,
            "achieved_estimated_work_ratio": 1.0,
            "absolute_ratio_error": 0.0,
            "ratio_tolerance": 0.0,
            "within_tolerance": True,
            "baseline_estimated_seconds": total,
            "adjusted_estimated_seconds": total,
        }

    normal_jobs = generate_burst(len(jobs), seed=seed, load="normal")
    expected_half, evidence = calibrate_half_load(normal_jobs)
    observed = [asdict(job) for job in jobs]
    expected = [asdict(job) for job in expected_half]
    if observed != expected:
        raise ValueError("half-load jobs do not match deterministic calibration evidence")
    return evidence


def build_workload_document(
    jobs: List[JobFeatures],
    *,
    seed: int,
    load: str,
    run_id: str,
    scenario: str,
    repetition: int,
) -> Dict[str, Any]:
    if load not in LOAD_PROFILES:
        raise ValueError(f"unknown load profile {load!r}")
    ranks = compute_ranks(jobs)
    return {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "kind": "ml-scheduler-workload-burst",
        "run": {
            "run_id": run_id,
            "scenario": scenario,
            "repetition": repetition,
            "seed": seed,
            "expected_jobs": len(jobs),
            "load_profile": load,
        },
        "profile": LOAD_PROFILES[load],
        "load_calibration": load_profile_evidence(jobs, seed=seed, load=load),
        "sampling_assumptions": {
            "category_ranges": CATEGORY_RANGES,
            "category_weights": CATEGORY_WEIGHTS,
            "note": "The paper does not publish exact category distributions; these are project assumptions.",
        },
        "work_model_assumptions": model_assumptions(),
        "jobs": [
            {
                "job_id": job.job_id,
                "job_seed": deterministic_job_seed(seed, job.job_id),
                "category": category_for_job(job),
                "features": {name: getattr(job, name) for name in FEATURE_NAMES},
                "rank": ranks[job.job_id],
            }
            for job in jobs
        ],
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def save_json(
    jobs: List[JobFeatures],
    path: str | os.PathLike[str],
    *,
    document: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a workload document; legacy list output remains opt-in by omission."""
    payload: Any = document if document is not None else [asdict(job) for job in jobs]
    _atomic_write_text(Path(path), json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _pod_manifest(
    job: JobFeatures,
    scheduler_name: str,
    image: str,
    *,
    namespace: str,
    run_id: str,
    scenario: str,
    config: str,
    repetition: int,
    expected_jobs: int,
    seed: int,
    pacing_mode: str,
    fixed_delay_seconds: float,
    reverse: bool,
    scheduling_gate: Optional[str],
    load_profile: str,
    image_pull_secrets: Optional[Sequence[str]],
) -> Dict[str, Any]:
    _validate_job(job)
    if not scheduler_name or not image or not namespace:
        raise ValueError("scheduler_name, image, and namespace must be non-empty")
    if pacing_mode not in {"none", "fixed", "adaptive"}:
        raise ValueError("pacing_mode must be none, fixed, or adaptive")
    if not math.isfinite(fixed_delay_seconds) or fixed_delay_seconds < 0:
        raise ValueError("fixed_delay_seconds must be finite and >= 0")
    if not isinstance(expected_jobs, int) or expected_jobs <= 0:
        raise ValueError("expected_jobs must be a positive integer")
    if repetition < 0 or seed < 0:
        raise ValueError("repetition and seed must be >= 0")
    if load_profile not in LOAD_PROFILES:
        raise ValueError(f"unknown load_profile {load_profile!r}")
    pull_secrets = list(dict.fromkeys(image_pull_secrets or []))
    for secret_name in pull_secrets:
        if (
            not isinstance(secret_name, str)
            or not secret_name
            or len(secret_name) > 253
            or not _DNS_LABEL_RE.fullmatch(secret_name)
        ):
            raise ValueError(f"invalid image pull secret name: {secret_name!r}")
    labels = {
        "app": "ml-sim-job",  # backwards-compatible selector
        "app.kubernetes.io/name": "ml-sim-job",
        LABEL_KEYS["run_id"]: _validate_label_value("run_id", run_id),
        LABEL_KEYS["scenario"]: _validate_label_value("scenario", scenario),
        LABEL_KEYS["config"]: _validate_label_value("config", config),
        LABEL_KEYS["repetition"]: _validate_label_value("repetition", str(repetition)),
    }
    annotations = {
        FEATURE_ANNOTATIONS[name]: str(getattr(job, name)) for name in FEATURE_NAMES
    }
    annotations.update({
        RUN_ANNOTATIONS["run_id"]: run_id,
        RUN_ANNOTATIONS["pacing_mode"]: pacing_mode,
        RUN_ANNOTATIONS["fixed_delay_seconds"]: str(float(fixed_delay_seconds)),
        RUN_ANNOTATIONS["reverse"]: str(bool(reverse)).lower(),
        RUN_ANNOTATIONS["expected_jobs"]: str(expected_jobs),
        "ml.scheduler/seed": str(seed),
        "ml.scheduler/load-profile": load_profile,
        WORK_MODEL_ANNOTATION: WORK_MODEL_VERSION,
        BLAS_THREADS_ANNOTATION: str(REPRODUCTION_BLAS_THREADS),
    })

    spec: Dict[str, Any] = {
        "schedulerName": scheduler_name,
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 30,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "volumes": [{
            "name": "tmp",
            "emptyDir": {"sizeLimit": "64Mi"},
        }],
        "containers": [{
            "name": "train",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
            "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
            "env": [
                {
                    "name": "JOB_ID",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                },
                {"name": "JOB_SEED", "value": str(deterministic_job_seed(seed, job.job_id))},
                {"name": WORK_MODEL_ENV, "value": WORK_MODEL_VERSION},
                {
                    "name": BLAS_THREADS_ENV,
                    "value": str(REPRODUCTION_BLAS_THREADS),
                },
                *[
                    {"name": f"JOB_{name}", "value": str(getattr(job, name))}
                    for name in FEATURE_NAMES
                ],
            ],
        }],
    }
    if pull_secrets:
        spec["imagePullSecrets"] = [{"name": name} for name in pull_secrets]
    if scheduling_gate:
        spec["schedulingGates"] = [{"name": scheduling_gate}]
        annotations[EXECUTION_CONTAINER_ANNOTATION] = "train"
        spec["containers"][0]["resources"] = {
            section: dict(values)
            for section, values in PRODUCTION_RESOURCE_REQUIREMENTS.items()
        }

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": job.job_id,
            "namespace": namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": spec,
    }


def to_pod_yaml(
    job: JobFeatures,
    scheduler_name: str,
    image: str,
    namespace: str = "default",
    *,
    run_id: str = "manual",
    scenario: str = "manual",
    config: str = "custom-baseline",
    repetition: int = 0,
    expected_jobs: int = 1,
    seed: int = 0,
    pacing_mode: str = "none",
    fixed_delay_seconds: float = 0.0,
    reverse: bool = False,
    scheduling_gate: Optional[str] = None,
    load_profile: str = "normal",
    image_pull_secrets: Optional[Sequence[str]] = None,
) -> str:
    """Serialize a Pod manifest with ``yaml.safe_dump`` (never interpolation)."""
    manifest = _pod_manifest(
        job,
        scheduler_name,
        image,
        namespace=namespace,
        run_id=run_id,
        scenario=scenario,
        config=config,
        repetition=repetition,
        expected_jobs=expected_jobs,
        seed=seed,
        pacing_mode=pacing_mode,
        fixed_delay_seconds=fixed_delay_seconds,
        reverse=reverse,
        scheduling_gate=scheduling_gate,
        load_profile=load_profile,
        image_pull_secrets=image_pull_secrets,
    )
    return yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)


def write_pod_manifests(
    jobs: List[JobFeatures],
    out_dir: str | os.PathLike[str],
    scheduler_name: str,
    image: str,
    *,
    namespace: str = "default",
    run_id: str = "manual",
    scenario: str = "manual",
    config: str = "custom-baseline",
    repetition: int = 0,
    seed: int = 0,
    pacing_mode: str = "none",
    fixed_delay_seconds: float = 0.0,
    reverse: bool = False,
    scheduling_gate: Optional[str] = None,
    load_profile: str = "normal",
    image_pull_secrets: Optional[Sequence[str]] = None,
) -> None:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        text = to_pod_yaml(
            job,
            scheduler_name,
            image,
            namespace,
            run_id=run_id,
            scenario=scenario,
            config=config,
            repetition=repetition,
            expected_jobs=len(jobs),
            seed=seed,
            pacing_mode=pacing_mode,
            fixed_delay_seconds=fixed_delay_seconds,
            reverse=reverse,
            scheduling_gate=scheduling_gate,
            load_profile=load_profile,
            image_pull_secrets=image_pull_secrets,
        )
        _atomic_write_text(directory / f"{job.job_id}.yaml", text)


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {path}; choose a new --out or pass --overwrite"
            )
        sentinel = path / OUTPUT_SENTINEL
        if not sentinel.is_file():
            raise RuntimeError(
                f"refusing to overwrite {path}: generator sentinel {OUTPUT_SENTINEL!r} is absent"
            )
        # The explicit flag plus sentinel makes removal scoped and auditable.
        for generated in ("pods_custom", "pods_default", "pods"):
            target = path / generated
            if target.exists():
                shutil.rmtree(target)
        for generated in ("jobs.json",):
            target = path / generated
            if target.exists():
                target.unlink()
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path / OUTPUT_SENTINEL, f"schema_version={WORKLOAD_SCHEMA_VERSION}\n")


def _config_to_scheduler_settings(config: str) -> tuple[str, float, bool]:
    if config == "custom-baseline":
        return "none", 0.0, False
    if config == "reversed":
        return "none", 0.0, True
    if config in {"adaptive", "custom-adaptive"}:
        return "adaptive", 0.0, False
    match = re.fullmatch(r"custom-delay-([0-9]+(?:\.[0-9]+)?)s", config)
    if match:
        return "fixed", float(match.group(1)), False
    raise ValueError(
        "custom config must be custom-baseline, reversed, adaptive, or custom-delay-<seconds>s"
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--load", choices=sorted(LOAD_PROFILES), default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./workload_out")
    parser.add_argument("--scheduler-name", default="ml-aware-scheduler")
    parser.add_argument("--image", default="ml-sim-job:0.1.0")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--run-id", default="manual-run")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--custom-config", default="custom-baseline")
    parser.add_argument("--scheduling-gate", default=None)
    parser.add_argument(
        "--image-pull-secret",
        action="append",
        default=[],
        help="registry Secret name to attach to generated Pods; repeat as needed",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.repetition < 0:
        parser.error("--repetition must be >= 0")
    scenario = args.scenario or f"{args.n}-{args.load}"
    pacing_mode, fixed_delay, reverse = _config_to_scheduler_settings(args.custom_config)
    output = Path(args.out).resolve()
    prepare_output_directory(output, overwrite=args.overwrite)

    jobs = generate_burst(args.n, seed=args.seed, load=args.load)
    document = build_workload_document(
        jobs,
        seed=args.seed,
        load=args.load,
        run_id=args.run_id,
        scenario=scenario,
        repetition=args.repetition,
    )
    document["manifest_sets"] = {
        "custom": {
            "scheduler_name": args.scheduler_name,
            "config": args.custom_config,
            "pacing_mode": pacing_mode,
            "fixed_delay_seconds": fixed_delay,
            "reverse": reverse,
            "scheduling_gate": args.scheduling_gate,
        },
        "default": {
            "scheduler_name": "default-scheduler",
            "config": "default",
            "pacing_mode": "none",
            "fixed_delay_seconds": 0.0,
            "reverse": False,
            "scheduling_gate": None,
        },
    }
    document["container"] = {
        "image": args.image,
        "image_pull_secrets": list(dict.fromkeys(args.image_pull_secret)),
        "custom_manifest_resources": (
            PRODUCTION_RESOURCE_REQUIREMENTS if args.scheduling_gate else None
        ),
    }
    save_json(jobs, output / "jobs.json", document=document)
    write_pod_manifests(
        jobs,
        output / "pods_custom",
        args.scheduler_name,
        args.image,
        namespace=args.namespace,
        run_id=args.run_id,
        scenario=scenario,
        config=args.custom_config,
        repetition=args.repetition,
        seed=args.seed,
        pacing_mode=pacing_mode,
        fixed_delay_seconds=fixed_delay,
        reverse=reverse,
        scheduling_gate=args.scheduling_gate,
        load_profile=args.load,
        image_pull_secrets=args.image_pull_secret,
    )
    write_pod_manifests(
        jobs,
        output / "pods_default",
        "default-scheduler",
        args.image,
        namespace=args.namespace,
        run_id=args.run_id,
        scenario=scenario,
        config="default",
        repetition=args.repetition,
        seed=args.seed,
        pacing_mode="none",
        fixed_delay_seconds=0.0,
        reverse=False,
        load_profile=args.load,
        image_pull_secrets=args.image_pull_secret,
    )
    print(f"Generated {len(jobs)} deterministic jobs in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
