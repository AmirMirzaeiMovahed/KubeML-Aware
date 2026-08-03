"""Versioned result contract shared by simulation and Kubernetes runs."""

from __future__ import annotations

import math
import platform
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


RESULT_SCHEMA_VERSION = "1.1"
RESULT_KIND = "ml-scheduler-experiment-run"
FEATURE_NAMES = ("T", "R", "M", "G", "C", "P")
TERMINAL_JOB_STATUSES = {"completed", "failed", "invalid", "missing"}


class SchemaValidationError(ValueError):
    """Raised when a result document violates the common contract."""


class IncompleteRunError(RuntimeError):
    """Carries the partial result so callers can persist failure evidence."""

    def __init__(self, message: str, document: Mapping[str, Any]):
        super().__init__(message)
        self.document = dict(document)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear percentile (Hyndman/Fan type 7; NumPy/Pandas default)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_jobs(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute metrics only from completed jobs; expose completion counts."""
    completed = [job for job in jobs if job.get("status") == "completed"]
    failed = [job for job in jobs if job.get("status") != "completed"]
    summary: Dict[str, Any] = {
        "observed_jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
    }
    if not completed:
        summary.update({
            "avg_jct": None,
            "tail_jct_p95": None,
            "p95_jct": None,
            "max_jct": None,
            "min_jct": None,
            "makespan": None,
            "avg_ilt": None,
        })
        return summary

    jcts = [float(job["jct_s"]) for job in completed]
    starts = sorted(float(job["execution_start_time"]) for job in completed)
    submissions = [float(job["submission_time"]) for job in completed]
    completions = [float(job["completion_time"]) for job in completed]
    inter_launch = [right - left for left, right in zip(starts, starts[1:])]
    p95 = percentile(jcts, 0.95)
    summary.update({
        "avg_jct": statistics.fmean(jcts),
        "tail_jct_p95": p95,  # compatibility with the original CSV column
        "p95_jct": p95,
        "max_jct": max(jcts),
        "min_jct": min(jcts),
        "makespan": max(completions) - min(submissions),
        "avg_ilt": statistics.fmean(inter_launch) if inter_launch else 0.0,
    })
    return summary


def local_environment_metadata() -> Dict[str, Any]:
    return {
        "collector": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
    }


def make_result_document(
    *,
    run_id: str,
    scenario: str,
    config: str,
    repetition: int,
    seed: int,
    expected_jobs: int,
    jobs: Iterable[Mapping[str, Any]],
    environment: Optional[Mapping[str, Any]] = None,
    failures: Optional[Iterable[Mapping[str, Any]]] = None,
    source: str,
) -> Dict[str, Any]:
    job_rows = [dict(job) for job in jobs]
    failure_rows = [dict(failure) for failure in (failures or [])]
    completed = sum(job.get("status") == "completed" for job in job_rows)
    rows_are_valid = not any(validate_job_row(job) for job in job_rows)
    valid = (
        len(job_rows) == expected_jobs
        and completed == expected_jobs
        and not failure_rows
        and rows_are_valid
    )
    document: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "created_at": utc_now(),
        "source": source,
        "run": {
            "run_id": run_id,
            "scenario": scenario,
            "config": config,
            "repetition": repetition,
            "seed": seed,
            "expected_jobs": expected_jobs,
            "status": "completed" if valid else "failed",
        },
        "environment": {**local_environment_metadata(), **dict(environment or {})},
        "summary": summarize_jobs(job_rows),
        "jobs": job_rows,
        "failures": failure_rows,
    }
    validate_result_document(document, strict=False)
    return document


def validate_job_row(job: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        errors.append("job_id must be a non-empty string")
    status = job.get("status")
    if status not in TERMINAL_JOB_STATUSES:
        errors.append(f"invalid status {status!r}")
    features = job.get("features")
    if not isinstance(features, Mapping):
        errors.append("features must be an object")
    else:
        for feature in FEATURE_NAMES:
            value = features.get(feature)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"feature {feature} must be finite")
    rank = job.get("rank")
    if not isinstance(rank, (int, float)) or isinstance(rank, bool) or not math.isfinite(float(rank)):
        errors.append("rank must be finite")
    if status == "completed":
        for field in ("submission_time", "execution_start_time", "completion_time", "jct_s"):
            value = job.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"{field} must be finite for completed jobs")
        if not errors:
            submission = float(job["submission_time"])
            start = float(job["execution_start_time"])
            end = float(job["completion_time"])
            if not submission <= start <= end:
                errors.append("timestamps must satisfy submission <= start <= completion")
            if not math.isclose(float(job["jct_s"]), end - submission, rel_tol=1e-9, abs_tol=1e-6):
                errors.append("jct_s does not equal completion_time - submission_time")
    elif not job.get("error"):
        errors.append("non-completed jobs require an error")
    return errors


def validate_result_document(document: Mapping[str, Any], *, strict: bool = True) -> List[str]:
    errors: List[str] = []
    if document.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RESULT_SCHEMA_VERSION!r}")
    if document.get("kind") != RESULT_KIND:
        errors.append(f"kind must be {RESULT_KIND!r}")
    if document.get("source") not in {"kubernetes", "simulation"}:
        errors.append("source must be 'kubernetes' or 'simulation'")
    run = document.get("run")
    if not isinstance(run, Mapping):
        errors.append("run must be an object")
        run = {}
    for field in ("run_id", "scenario", "config"):
        if not isinstance(run.get(field), str) or not run.get(field):
            errors.append(f"run.{field} must be a non-empty string")
    for field in ("repetition", "seed", "expected_jobs"):
        if not isinstance(run.get(field), int) or isinstance(run.get(field), bool):
            errors.append(f"run.{field} must be an integer")
    expected = run.get("expected_jobs")
    if isinstance(expected, int) and expected <= 0:
        errors.append("run.expected_jobs must be > 0")
    jobs = document.get("jobs")
    if not isinstance(jobs, list):
        errors.append("jobs must be an array")
        jobs = []
    identifiers: List[str] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            errors.append(f"jobs[{index}] must be an object")
            continue
        identifiers.append(str(job.get("job_id")))
        errors.extend(f"jobs[{index}]: {error}" for error in validate_job_row(job))
    if len(set(identifiers)) != len(identifiers):
        errors.append("job_id values must be unique")
    environment = document.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("environment must be an object")
    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("summary must be an object")
    else:
        expected_summary = summarize_jobs(
            [job for job in jobs if isinstance(job, Mapping)]
        )
        for field, expected_value in expected_summary.items():
            observed_value = summary.get(field)
            if isinstance(expected_value, float):
                if not isinstance(observed_value, (int, float)) or isinstance(observed_value, bool) or not math.isclose(
                    float(observed_value), expected_value, rel_tol=1e-9, abs_tol=1e-9
                ):
                    errors.append(f"summary.{field} does not match jobs")
            elif observed_value != expected_value:
                errors.append(f"summary.{field} does not match jobs")
    failures = document.get("failures")
    if not isinstance(failures, list):
        errors.append("failures must be an array")
    if strict and isinstance(expected, int) and len(jobs) != expected:
        errors.append(f"expected {expected} jobs, observed {len(jobs)}")
    if strict and any(job.get("status") != "completed" for job in jobs if isinstance(job, Mapping)):
        errors.append("strict result contains non-completed jobs")
    if strict and document.get("failures"):
        errors.append("strict result contains failures")
    if strict and run.get("status") != "completed":
        errors.append("run.status is not completed")
    if errors and strict:
        raise SchemaValidationError("; ".join(errors))
    return errors
