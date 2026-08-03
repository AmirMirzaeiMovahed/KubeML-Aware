"""Deterministic, category-neutral burst submission for cluster experiments.

The article describes every workload as a burst.  Submitting a directory with
``kubectl apply -f`` does not provide that contract: files are visited in
lexical order and category-prefixed filenames therefore leak workload class
into the default scheduler's arrival order.  This module loads and validates
the complete manifest set, shuffles it with the run seed, and releases one
worker per Pod through a common barrier.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SUBMISSION_SCHEMA_VERSION = "1.0"


class BurstSubmissionError(RuntimeError):
    """Raised when one or more Pod creates fail after the burst barrier opens."""

    def __init__(self, message: str, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


@dataclass(frozen=True)
class PodManifest:
    job_id: str
    namespace: str
    body: Mapping[str, Any]


def _epoch(value: Any) -> float:
    if isinstance(value, datetime):
        return float(value.timestamp())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    if isinstance(value, str) and value:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        return float(datetime.fromisoformat(normalized).timestamp())
    raise ValueError(f"invalid Kubernetes creation timestamp: {value!r}")


def load_pod_manifests(
    directory: Path, *, expected_namespace: str
) -> list[PodManifest]:
    """Load a flat directory of one-Pod YAML files with a strict contract."""

    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValueError(f"manifest directory contains no YAML files: {directory}")
    manifests: list[PodManifest] = []
    seen = set()
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("apiVersion") != "v1":
            raise ValueError(f"manifest is not a core/v1 object: {path}")
        if payload.get("kind") != "Pod":
            raise ValueError(f"manifest is not a Pod: {path}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"manifest metadata is missing: {path}")
        job_id = metadata.get("name")
        namespace = metadata.get("namespace")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"manifest Pod name is missing: {path}")
        if path.stem != job_id:
            raise ValueError(
                f"manifest filename/name mismatch: {path.name} != {job_id!r}"
            )
        if namespace != expected_namespace:
            raise ValueError(
                f"manifest namespace mismatch for {job_id!r}: "
                f"{namespace!r} != {expected_namespace!r}"
            )
        if job_id in seen:
            raise ValueError(f"duplicate Pod name in manifest set: {job_id!r}")
        seen.add(job_id)
        manifests.append(PodManifest(job_id, namespace, payload))
    return manifests


def _metadata_value(response: Any, name: str) -> Any:
    metadata = getattr(response, "metadata", None)
    if metadata is not None:
        value = getattr(metadata, name, None)
        if value is not None:
            return value
    if isinstance(response, Mapping):
        raw_metadata = response.get("metadata")
        if isinstance(raw_metadata, Mapping):
            aliases = {"creation_timestamp": "creationTimestamp"}
            return raw_metadata.get(aliases.get(name, name))
    return None


def submit_burst(
    manifest_directory: Path,
    *,
    expected_namespace: str,
    expected_count: int,
    seed: int,
    create_pod: Callable[[str, Mapping[str, Any]], Any],
    max_workers: int,
    max_creation_spread_seconds: float,
    wall_time: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Create all Pods behind one client-side barrier and return audit evidence.

    Article-mode execution requires at least one worker per Pod.  This avoids a
    hidden queue in the client process and makes API-server admission, rather
    than filename traversal, determine the observed arrival order.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("submission seed must be a non-negative integer")
    if expected_count <= 0:
        raise ValueError("expected_count must be > 0")
    if max_workers < expected_count:
        raise ValueError(
            f"max_workers={max_workers} is smaller than expected_count={expected_count}; "
            "article bursts require one worker per Pod"
        )
    if (
        not math.isfinite(max_creation_spread_seconds)
        or max_creation_spread_seconds <= 0
    ):
        raise ValueError("max_creation_spread_seconds must be finite and > 0")

    manifests = load_pod_manifests(
        manifest_directory, expected_namespace=expected_namespace
    )
    if len(manifests) != expected_count:
        raise ValueError(
            f"manifest count {len(manifests)} does not equal expected_count {expected_count}"
        )
    randomized = list(manifests)
    random.Random(seed).shuffle(randomized)

    release = threading.Event()
    evidence_lock = threading.Lock()
    rows: list[dict[str, Any]] = []
    burst_started_at = wall_time()
    burst_started_monotonic = monotonic()

    def create(index: int, manifest: PodManifest) -> None:
        release.wait()
        requested_at = wall_time()
        requested_monotonic = monotonic()
        row: dict[str, Any] = {
            "submission_index": index,
            "job_id": manifest.job_id,
            "client_requested_at": requested_at,
            "client_requested_offset_seconds": requested_monotonic
            - burst_started_monotonic,
            "status": "creating",
            "error": None,
        }
        try:
            response = create_pod(manifest.namespace, manifest.body)
            row.update(
                {
                    "status": "created",
                    "uid": str(_metadata_value(response, "uid") or ""),
                    "server_creation_timestamp": _epoch(
                        _metadata_value(response, "creation_timestamp")
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve every API failure in evidence
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            row["client_completed_at"] = wall_time()
            row["client_duration_seconds"] = monotonic() - requested_monotonic
            with evidence_lock:
                rows.append(row)

    futures: Sequence[Future[None]]
    with ThreadPoolExecutor(
        max_workers=expected_count, thread_name_prefix="pod-burst"
    ) as executor:
        futures = [
            executor.submit(create, index, manifest)
            for index, manifest in enumerate(randomized)
        ]
        release.set()
        for future in as_completed(futures):
            future.result()

    rows.sort(key=lambda row: int(row["submission_index"]))
    created = [row for row in rows if row["status"] == "created"]
    server_times = [float(row["server_creation_timestamp"]) for row in created]
    request_times = [float(row["client_requested_at"]) for row in rows]
    completed_at = wall_time()
    evidence: dict[str, Any] = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "mode": "concurrent-client-barrier",
        "seed": seed,
        "expected_jobs": expected_count,
        "worker_count": expected_count,
        "configured_worker_ceiling": max_workers,
        "burst_started_at": burst_started_at,
        "burst_completed_at": completed_at,
        "client_request_spread_seconds": (
            max(request_times) - min(request_times) if request_times else None
        ),
        "server_creation_spread_seconds": (
            max(server_times) - min(server_times) if server_times else None
        ),
        "max_creation_spread_seconds": max_creation_spread_seconds,
        "randomized_job_order": [manifest.job_id for manifest in randomized],
        "jobs": rows,
    }
    failures = [row for row in rows if row["status"] != "created"]
    if failures:
        raise BurstSubmissionError(
            f"burst submission failed for {len(failures)}/{expected_count} Pods",
            evidence,
        )
    spread = float(evidence["server_creation_spread_seconds"])
    if spread > max_creation_spread_seconds:
        raise BurstSubmissionError(
            f"server creation spread {spread:.6f}s exceeds "
            f"{max_creation_spread_seconds:.6f}s",
            evidence,
        )
    return evidence
