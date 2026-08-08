"""Strict run grouping and quiet-period burst reconciliation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    ANNOTATION_MAP,
    EXPECTED_COUNT_ANNOTATION,
    EXPECTED_JOBS_ANNOTATION,
    FIXED_DELAY_ANNOTATION,
    INFERENCE_ANNOTATION_MAP,
    PACING_MODE_ANNOTATION,
    REVERSE_ANNOTATION,
    RUN_ID_ANNOTATION,
    RUN_ID_LABEL,
    WORKLOAD_KIND_ANNOTATION,
)
from .rank import (
    InferenceFeatures,
    JobFeatures,
    RankValidationError,
    WorkloadFeatures,
    validate_inference_jobs,
    validate_jobs,
)


class AnnotationValidationError(ValueError):
    pass


class BurstContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSettings:
    run_id: str
    expected_count: int
    pacing_mode: str
    fixed_delay: float
    reverse: bool


def _metadata_map(pod: Any, field: str) -> Dict[str, str]:
    return dict(getattr(getattr(pod, "metadata", None), field, None) or {})


def pod_run_id(pod: Any) -> Optional[str]:
    labels = _metadata_map(pod, "labels")
    annotations = _metadata_map(pod, "annotations")
    label_value = labels.get(RUN_ID_LABEL)
    annotation_value = annotations.get(RUN_ID_ANNOTATION)
    if label_value and annotation_value and label_value != annotation_value:
        name = getattr(getattr(pod, "metadata", None), "name", "unknown")
        raise AnnotationValidationError(
            f"pod {name!r} has conflicting run-id label and annotation"
        )
    value = label_value or annotation_value
    return value.strip() if isinstance(value, str) and value.strip() else None


def extract_features(pod: Any) -> WorkloadFeatures:
    metadata = getattr(pod, "metadata", None)
    name = getattr(metadata, "name", None)
    if not isinstance(name, str) or not name:
        raise AnnotationValidationError("pod metadata.name is missing")
    annotations = _metadata_map(pod, "annotations")
    workload_kind = str(annotations.get(WORKLOAD_KIND_ANNOTATION, "training")).strip().lower()
    if workload_kind == "inference":
        values: Dict[str, float] = {}
        for feature, key in INFERENCE_ANNOTATION_MAP.items():
            if key not in annotations:
                raise AnnotationValidationError(
                    f"inference pod {name!r} is missing required annotation {key!r}"
                )
            try:
                values[feature] = float(annotations[key])
            except (TypeError, ValueError) as exc:
                raise AnnotationValidationError(
                    f"inference pod {name!r} annotation {key!r} is not numeric"
                ) from exc
        job = InferenceFeatures(job_id=name, **values)
        try:
            validate_inference_jobs([job])
        except RankValidationError as exc:
            raise AnnotationValidationError(str(exc)) from exc
        return job
    if workload_kind != "training":
        raise AnnotationValidationError(
            f"pod {name!r} has unsupported workload kind {workload_kind!r}"
        )
    values: Dict[str, float] = {}
    for feature, key in ANNOTATION_MAP.items():
        if key not in annotations:
            raise AnnotationValidationError(
                f"pod {name!r} is missing required annotation {key!r}"
            )
        raw = annotations[key]
        if isinstance(raw, bool):
            raise AnnotationValidationError(
                f"pod {name!r} annotation {key!r} must be numeric"
            )
        try:
            values[feature] = float(raw)
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError(
                f"pod {name!r} annotation {key!r} is not numeric: {raw!r}"
            ) from exc
    job = JobFeatures(job_id=name, **values)
    try:
        validate_jobs([job])
    except RankValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    return job


def _parse_positive_int(raw: object, field: str) -> int:
    try:
        text = str(raw).strip()
        if not text or any(character in text for character in ".eE"):
            raise ValueError
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{field} must be a positive integer") from exc
    if value <= 0:
        raise AnnotationValidationError(f"{field} must be a positive integer")
    return value


def _parse_bool(raw: object, field: str) -> bool:
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise AnnotationValidationError(f"{field} must be true or false")


def run_settings_for_pods(
    pods: Sequence[Any],
    *,
    fallback_run_id: Optional[str],
    fallback_expected_count: Optional[int],
    fallback_pacing_mode: str,
    fallback_fixed_delay: float,
    fallback_reverse: bool,
) -> RunSettings:
    if not pods:
        raise BurstContractError("cannot derive run settings from an empty pod group")
    run_ids = {pod_run_id(pod) or fallback_run_id for pod in pods}
    if None in run_ids or len(run_ids) != 1:
        raise AnnotationValidationError("all pods must have one identical non-empty run-id")
    run_id = str(next(iter(run_ids)))

    settings = set()
    for pod in pods:
        annotations = _metadata_map(pod, "annotations")
        expected_raw = annotations.get(
            EXPECTED_JOBS_ANNOTATION,
            annotations.get(EXPECTED_COUNT_ANNOTATION, fallback_expected_count),
        )
        if expected_raw is None:
            raise AnnotationValidationError(
                f"run {run_id!r} has no expected-jobs annotation or CLI fallback"
            )
        expected = _parse_positive_int(expected_raw, EXPECTED_JOBS_ANNOTATION)
        pacing_mode = str(
            annotations.get(PACING_MODE_ANNOTATION, fallback_pacing_mode)
        ).strip().lower()
        if pacing_mode not in {"none", "fixed", "adaptive"}:
            raise AnnotationValidationError(
                f"{PACING_MODE_ANNOTATION} must be none, fixed, or adaptive"
            )
        fixed_raw = annotations.get(FIXED_DELAY_ANNOTATION, fallback_fixed_delay)
        try:
            fixed_delay = float(fixed_raw)
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError(
                f"{FIXED_DELAY_ANNOTATION} must be numeric"
            ) from exc
        if fixed_delay < 0:
            raise AnnotationValidationError(
                f"{FIXED_DELAY_ANNOTATION} must be >= 0"
            )
        reverse = _parse_bool(
            annotations.get(REVERSE_ANNOTATION, fallback_reverse), REVERSE_ANNOTATION
        )
        settings.add((expected, pacing_mode, fixed_delay, reverse))
    if len(settings) != 1:
        raise AnnotationValidationError(
            f"run {run_id!r} contains inconsistent per-run scheduler annotations"
        )
    expected, pacing_mode, fixed_delay, reverse = next(iter(settings))
    return RunSettings(run_id, expected, pacing_mode, fixed_delay, reverse)


def group_pods_by_run(pods: Iterable[Any]) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = {}
    for pod in pods:
        run_id = pod_run_id(pod)
        if run_id:
            groups.setdefault(run_id, []).append(pod)
    return groups


def pod_fingerprint(pod: Any) -> Tuple[str, str, str]:
    metadata = getattr(pod, "metadata", None)
    return (
        str(getattr(metadata, "name", "")),
        str(getattr(metadata, "uid", "")),
        str(getattr(metadata, "resource_version", "")),
    )


class BurstCollector:
    """Repeatedly relist a run until its exact membership is quiet and complete.

    ``wait_for_change`` may use a Kubernetes watch.  Every wake-up is followed
    by a full relist, so watch disconnects and missed events cannot corrupt the
    final membership.
    """

    def __init__(
        self,
        list_eligible_pods: Callable[[], Sequence[Any]],
        *,
        quiet_period: float,
        timeout: float,
        poll_interval: float,
        wait_for_change: Optional[Callable[[float], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.list_eligible_pods = list_eligible_pods
        self.quiet_period = quiet_period
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.wait_for_change = wait_for_change
        self.monotonic = monotonic
        self.sleep = sleep

    def collect(self, run_id: str, expected_count: int) -> List[Any]:
        deadline = self.monotonic() + self.timeout
        last_fingerprint: Optional[Tuple[Tuple[str, str, str], ...]] = None
        last_change = self.monotonic()
        observed_names: List[str] = []

        while self.monotonic() < deadline:
            matching = [
                pod for pod in self.list_eligible_pods() if pod_run_id(pod) == run_id
            ]
            matching.sort(key=lambda pod: pod_fingerprint(pod)[0])
            fingerprint = tuple(pod_fingerprint(pod) for pod in matching)
            now = self.monotonic()
            if fingerprint != last_fingerprint:
                last_fingerprint = fingerprint
                last_change = now
            observed_names = [pod_fingerprint(pod)[0] for pod in matching]
            if len(matching) > expected_count:
                raise BurstContractError(
                    f"run {run_id!r} expected {expected_count} pods but observed "
                    f"{len(matching)}: {observed_names}"
                )
            if len(matching) == expected_count:
                for pod in matching:
                    extract_features(pod)
                quiet_remaining = self.quiet_period - (now - last_change)
                if quiet_remaining <= 0:
                    return matching
                wait_time = min(self.poll_interval, quiet_remaining, deadline - now)
            else:
                wait_time = min(self.poll_interval, deadline - now)
            if wait_time > 0:
                if self.wait_for_change is not None:
                    try:
                        self.wait_for_change(wait_time)
                    except Exception:
                        # Watches are accelerators, never the source of truth.
                        self.sleep(wait_time)
                else:
                    self.sleep(wait_time)

        raise BurstContractError(
            f"run {run_id!r} did not reach exactly {expected_count} quiet pods "
            f"within {self.timeout:.3f}s; last observed {len(observed_names)}: "
            f"{observed_names}"
        )
