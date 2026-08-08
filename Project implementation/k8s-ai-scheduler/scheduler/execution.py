"""Reliable detection of the trainer's useful-execution marker."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Callable, Optional

from .constants import EXECUTION_CONTAINER_ANNOTATION, EXECUTION_EVENT
from .kube import (
    ApiFailureKind,
    KubernetesOperationError,
    api_timeout,
    call_with_retries,
)


class ExecutionStartError(RuntimeError):
    pass


_LEGACY_MARKER = re.compile(
    r"^\s*\[(?P<timestamp>\d+(?:\.\d+)?)\]\s+EXECUTION_STARTED(?:\s.*)?$"
)


def execution_container_for_pod(pod: Any) -> str:
    """Resolve the application container without assuming a fixed Pod shape."""

    metadata = getattr(pod, "metadata", None)
    annotations = dict(getattr(metadata, "annotations", None) or {})
    requested = annotations.get(EXECUTION_CONTAINER_ANNOTATION)
    containers = getattr(getattr(pod, "spec", None), "containers", None) or []
    names = [getattr(container, "name", None) for container in containers]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ExecutionStartError("pod has no valid application containers")
    if len(names) != len(set(names)):
        raise ExecutionStartError("pod has duplicate application container names")
    if requested:
        if requested not in names:
            raise ExecutionStartError(
                f"{EXECUTION_CONTAINER_ANNOTATION}={requested!r} does not name a container"
            )
        return str(requested)
    if len(names) == 1:
        return str(names[0])
    if "train" in names:
        return "train"
    raise ExecutionStartError(
        f"multi-container pod must set {EXECUTION_CONTAINER_ANNOTATION}"
    )


def parse_execution_marker(
    log_text: str, *, expected_job_id: Optional[str] = None
) -> Optional[float]:
    """Accept the versioned JSON marker and the original ``[epoch]`` marker."""

    for line in (log_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except (TypeError, ValueError):
                payload = None
            if (
                isinstance(payload, dict)
                and str(payload.get("event", "")).upper() == EXECUTION_EVENT
            ):
                marker_job_id = payload.get("job_id")
                if expected_job_id and marker_job_id not in (None, expected_job_id):
                    continue
                value = payload.get(
                    "timestamp", payload.get("ts", payload.get("epoch"))
                )
                try:
                    timestamp = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(timestamp) and timestamp > 0:
                    return timestamp
        match = _LEGACY_MARKER.match(stripped)
        if match:
            timestamp = float(match.group("timestamp"))
            if math.isfinite(timestamp) and timestamp > 0:
                return timestamp
    return None


def _pod_terminal_failure(pod: Any) -> Optional[str]:
    phase = getattr(getattr(pod, "status", None), "phase", None)
    if phase == "Failed":
        reason = getattr(getattr(pod, "status", None), "reason", None) or "unknown"
        message = getattr(getattr(pod, "status", None), "message", None) or ""
        return f"pod entered Failed phase ({reason}): {message}".rstrip()
    statuses = getattr(getattr(pod, "status", None), "container_statuses", None) or []
    for status in statuses:
        terminated = getattr(getattr(status, "state", None), "terminated", None)
        if terminated is not None:
            exit_code = getattr(terminated, "exit_code", None)
            reason = getattr(terminated, "reason", None)
            # Only treat as failure if container terminated with error (non-zero exit code
            # or non-Completed reason). Completed with exit_code=0 is success.
            if exit_code != 0 or (reason is not None and reason != "Completed"):
                return (
                    f"container {getattr(status, 'name', 'unknown')} terminated before marker "
                    f"(exit_code={exit_code}, reason={reason})"
                )
    return None


def wait_for_execution_start(
    core_api: Any,
    pod_name: str,
    namespace: str,
    *,
    timeout: float,
    api_timeout_seconds: float,
    api_retries: int,
    container: str = "train",
    poll_interval: float = 0.3,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> float:
    deadline = monotonic() + timeout
    last_observation = "no logs available"
    while monotonic() < deadline:
        if stop_requested is not None and stop_requested():
            raise InterruptedError("shutdown requested while waiting for execution")
        try:
            logs = call_with_retries(
                lambda: core_api.read_namespaced_pod_log(
                    pod_name,
                    namespace,
                    container=container,
                    timestamps=False,
                    _request_timeout=api_timeout(api_timeout_seconds),
                ),
                operation=f"read execution log for {pod_name}",
                retries=api_retries,
                log_read=True,
            )
            # Handle K8s Python client returning str containing repr(bytes)
            # (literal b'...' with \\n) instead of decoded string
            if isinstance(logs, bytes):
                logs = logs.decode("utf-8", errors="replace")
            elif isinstance(logs, str) and logs.startswith("b'") and logs.endswith("'"):
                try:
                    # Strip b'...' wrapper and decode escaped sequences
                    logs = logs[2:-1].encode().decode("unicode_escape")
                except Exception:
                    pass  # fallback to original string
            marker = parse_execution_marker(logs, expected_job_id=pod_name)
            if marker is not None:
                return marker
            last_observation = "container logs did not contain a valid marker"
        except KubernetesOperationError as exc:
            if exc.kind not in {
                ApiFailureKind.TRANSIENT,
                ApiFailureKind.NOT_FOUND,
                ApiFailureKind.INVALID,
            }:
                raise ExecutionStartError(str(exc)) from exc
            last_observation = str(exc)

        try:
            pod = call_with_retries(
                lambda: core_api.read_namespaced_pod_status(
                    pod_name,
                    namespace,
                    _request_timeout=api_timeout(api_timeout_seconds),
                ),
                operation=f"read pod status for {pod_name}",
                retries=api_retries,
            )
            failure = _pod_terminal_failure(pod)
            if failure:
                raise ExecutionStartError(failure)

            # Check if pod has already completed successfully (Succeeded phase)
            phase = getattr(getattr(pod, "status", None), "phase", None)
            if phase == "Succeeded":
                # Pod completed - read logs one more time to find marker
                try:
                    logs = call_with_retries(
                        lambda: core_api.read_namespaced_pod_log(
                            pod_name,
                            namespace,
                            container=container,
                            timestamps=False,
                            _request_timeout=api_timeout(api_timeout_seconds),
                        ),
                        operation=f"read execution log for completed pod {pod_name}",
                        retries=api_retries,
                        log_read=True,
                    )
                    # Handle K8s Python client returning str containing repr(bytes)
                    # (literal b'...' with \\n) instead of decoded string
                    if isinstance(logs, bytes):
                        logs = logs.decode("utf-8", errors="replace")
                    elif (
                        isinstance(logs, str)
                        and logs.startswith("b'")
                        and logs.endswith("'")
                    ):
                        try:
                            # Strip b'...' wrapper and decode escaped sequences
                            logs = logs[2:-1].encode().decode("unicode_escape")
                        except Exception:
                            pass  # fallback to original string
                    marker = parse_execution_marker(logs, expected_job_id=pod_name)
                    if marker is not None:
                        return marker
                    last_observation = (
                        "completed pod logs did not contain a valid marker"
                    )
                except KubernetesOperationError as exc:
                    if exc.kind not in {
                        ApiFailureKind.TRANSIENT,
                        ApiFailureKind.NOT_FOUND,
                        ApiFailureKind.INVALID,
                    }:
                        raise ExecutionStartError(str(exc)) from exc
                    last_observation = str(exc)
        except KubernetesOperationError as exc:
            if exc.kind not in {ApiFailureKind.TRANSIENT, ApiFailureKind.NOT_FOUND}:
                raise ExecutionStartError(str(exc)) from exc
            last_observation = str(exc)

        remaining = deadline - monotonic()
        if remaining > 0:
            sleep(min(poll_interval, remaining))
    if stop_requested is not None and stop_requested():
        raise InterruptedError("shutdown requested while waiting for execution")
    raise ExecutionStartError(
        f"timed out after {timeout:.3f}s waiting for {EXECUTION_EVENT} from "
        f"pod {namespace}/{pod_name}; last observation: {last_observation}"
    )
