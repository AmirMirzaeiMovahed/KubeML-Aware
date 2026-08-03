"""Fail-closed execution controls for reproducible cluster experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from k8s.work_model import (
    BLAS_THREADS_ENV,
    REPRODUCTION_BLAS_THREADS,
    WORK_MODEL_ENV,
    WORK_MODEL_VERSION,
)


CONTROL_SCHEMA_VERSION = "1.0"
PREWARM_CONTROL_LABEL = "ml.scheduler/control"
PREWARM_CONTROL_VALUE = "trainer-prewarm"
DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_COOLDOWN_CLEAN_POLLS = 3
_IMMUTABLE_IMAGE_RE = re.compile(r"^.+@sha256:[a-f0-9]{64}$")

PREWARM_SCRIPT = """\
import json
import numpy as np
from k8s.train import blas_runtime_evidence
a = np.ones((128, 128), dtype=np.float32)
out = np.empty_like(a)
np.matmul(a, a, out=out)
print(json.dumps({
    "event": "PREWARM_ATTESTATION",
    "blas_runtime": blas_runtime_evidence(),
    "matrix_shape": [128, 128],
}, sort_keys=True, separators=(",", ":")))
"""


def execution_controls_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "trainer_image_prewarm": "required-on-target-node",
        "blas_threads": REPRODUCTION_BLAS_THREADS,
        "cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
        "cooldown_clean_polls": DEFAULT_COOLDOWN_CLEAN_POLLS,
        "cooldown_policy": "fixed-window-clean-node-and-scheduler-continuity",
        "minikube_driver": "docker",
    }


def prewarm_pod_manifest(
    *,
    namespace: str,
    target_node: str,
    image: str,
    image_pull_secrets: Sequence[str] = (),
) -> dict[str, Any]:
    if not namespace or not target_node:
        raise ValueError("namespace and target_node must be non-empty")
    if not _IMMUTABLE_IMAGE_RE.fullmatch(image):
        raise ValueError("prewarm image must be digest-qualified")
    digest = image.rsplit("sha256:", 1)[1]
    name = f"ml-trainer-prewarm-{digest[:12]}"
    spec: dict[str, Any] = {
        "nodeName": target_node,
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "terminationGracePeriodSeconds": 10,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "volumes": [{"name": "tmp", "emptyDir": {"sizeLimit": "32Mi"}}],
        "containers": [{
            "name": "attest",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["python", "-c", PREWARM_SCRIPT],
            "env": [
                {"name": WORK_MODEL_ENV, "value": WORK_MODEL_VERSION},
                {
                    "name": BLAS_THREADS_ENV,
                    "value": str(REPRODUCTION_BLAS_THREADS),
                },
            ],
            "resources": {
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "1", "memory": "512Mi"},
            },
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
            "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
        }],
    }
    secrets = list(dict.fromkeys(image_pull_secrets))
    if secrets:
        spec["imagePullSecrets"] = [{"name": secret} for secret in secrets]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {PREWARM_CONTROL_LABEL: PREWARM_CONTROL_VALUE},
        },
        "spec": spec,
    }


def _parse_attestation(logs: str) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for line in logs.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, Mapping)
            and payload.get("event") == "PREWARM_ATTESTATION"
        ):
            matches.append(payload)
    if len(matches) != 1:
        raise ValueError(f"expected one PREWARM_ATTESTATION; observed {len(matches)}")
    return matches[0]


def validate_prewarm_observation(
    pod: Mapping[str, Any],
    logs: str,
    *,
    expected_image: str,
    target_node: str,
) -> dict[str, Any]:
    metadata = pod.get("metadata") or {}
    spec = pod.get("spec") or {}
    status = pod.get("status") or {}
    containers = spec.get("containers") or []
    statuses = status.get("containerStatuses") or []
    if status.get("phase") != "Succeeded":
        raise ValueError(f"prewarm Pod phase is {status.get('phase')!r}")
    if spec.get("nodeName") != target_node:
        raise ValueError("prewarm Pod ran on the wrong node")
    if len(containers) != 1 or containers[0].get("image") != expected_image:
        raise ValueError("prewarm Pod requested an unexpected image")
    if len(statuses) != 1 or statuses[0].get("restartCount") != 0:
        raise ValueError("prewarm container restarted or status is missing")
    image_id = str(statuses[0].get("imageID") or "")
    if not re.search(r"sha256:[a-f0-9]{64}", image_id):
        raise ValueError("prewarm runtime imageID is not digest-qualified")

    attestation = _parse_attestation(logs)
    runtime = attestation.get("blas_runtime") or {}
    libraries = runtime.get("libraries") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("expected_threads") != REPRODUCTION_BLAS_THREADS
        or not isinstance(libraries, list)
        or not libraries
        or any(
            not isinstance(pool, Mapping)
            or pool.get("num_threads") != REPRODUCTION_BLAS_THREADS
            for pool in (libraries or [])
        )
    ):
        raise ValueError("prewarm BLAS runtime evidence is invalid")
    evidence = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "status": "passed",
        "pod_name": metadata.get("name"),
        "pod_uid": metadata.get("uid"),
        "target_node": target_node,
        "requested_image": expected_image,
        "runtime_image_id": image_id,
        "creation_timestamp": metadata.get("creationTimestamp"),
        "start_timestamp": status.get("startTime"),
        "finished_timestamp": (
            ((statuses[0].get("state") or {}).get("terminated") or {}).get(
                "finishedAt"
            )
        ),
        "attestation": dict(attestation),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return evidence


def validate_cooldown_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    expected_clean_polls: int = DEFAULT_COOLDOWN_CLEAN_POLLS,
) -> list[str]:
    errors: list[str] = []
    elapsed = evidence.get("elapsed_seconds")
    polls = evidence.get("clean_polls")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) + 0.05 < expected_seconds
    ):
        errors.append(f"cooldown elapsed_seconds is below {expected_seconds}")
    if isinstance(polls, bool) or not isinstance(polls, int) or polls < expected_clean_polls:
        errors.append(f"cooldown clean_polls is below {expected_clean_polls}")
    if evidence.get("workload_pods_observed") != 0:
        errors.append("cooldown observed workload Pods")
    if evidence.get("scheduler_continuity") is not True:
        errors.append("cooldown did not prove scheduler continuity")
    if evidence.get("node_pressure_clear") is not True:
        errors.append("cooldown did not prove clear node pressure conditions")
    return errors


def validate_execution_controls_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_image: str,
    target_node: str,
    expected_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    expected_clean_polls: int = DEFAULT_COOLDOWN_CLEAN_POLLS,
) -> list[str]:
    errors: list[str] = []
    if evidence.get("contract") != execution_controls_contract():
        errors.append("execution control contract mismatch")
    prewarm = evidence.get("prewarm")
    if not isinstance(prewarm, Mapping):
        errors.append("prewarm evidence is missing")
    else:
        if prewarm.get("status") != "passed":
            errors.append("prewarm status is not passed")
        if prewarm.get("requested_image") != expected_image:
            errors.append("prewarm requested image mismatch")
        if prewarm.get("target_node") != target_node:
            errors.append("prewarm target node mismatch")
        attestation = prewarm.get("attestation")
        runtime = (
            attestation.get("blas_runtime")
            if isinstance(attestation, Mapping)
            else None
        )
        libraries = runtime.get("libraries") if isinstance(runtime, Mapping) else None
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("expected_threads") != REPRODUCTION_BLAS_THREADS
            or not isinstance(libraries, list)
            or not libraries
            or any(
                not isinstance(pool, Mapping)
                or pool.get("num_threads") != REPRODUCTION_BLAS_THREADS
                for pool in (libraries or [])
            )
        ):
            errors.append("recorded prewarm BLAS evidence is invalid")
        recorded_digest = prewarm.get("evidence_sha256")
        digest_payload = {
            key: value for key, value in prewarm.items() if key != "evidence_sha256"
        }
        expected_digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if recorded_digest != expected_digest:
            errors.append("prewarm evidence digest mismatch")
    cooldown = evidence.get("pre_run_cooldown")
    if not isinstance(cooldown, Mapping):
        errors.append("pre-run cooldown evidence is missing")
    else:
        errors.extend(
            validate_cooldown_evidence(
                cooldown,
                expected_seconds=expected_cooldown_seconds,
                expected_clean_polls=expected_clean_polls,
            )
        )
    return errors


def _get_case_insensitive(mapping: Mapping[str, Any], name: str) -> Any:
    for key, value in mapping.items():
        if str(key).lower() == name.lower():
            return value
    return None


def validate_minikube_attestation(
    profile_list: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    profile: str,
    version: str,
) -> dict[str, Any]:
    valid = _get_case_insensitive(profile_list, "valid")
    if not isinstance(valid, list):
        raise ValueError("minikube profile list has no valid profile array")
    matches = [
        item
        for item in valid
        if isinstance(item, Mapping)
        and str(_get_case_insensitive(item, "name") or "") == profile
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Minikube profile {profile!r}")
    observed = matches[0]
    config = _get_case_insensitive(observed, "config")
    config = config if isinstance(config, Mapping) else {}
    driver = (
        _get_case_insensitive(config, "driver")
        or _get_case_insensitive(observed, "vm driver")
        or _get_case_insensitive(observed, "driver")
    )
    profile_status = _get_case_insensitive(observed, "status")
    if str(driver).lower() != "docker":
        raise ValueError(f"Minikube driver must be docker; observed {driver!r}")
    if str(profile_status).lower() != "running":
        raise ValueError(
            f"Minikube profile must be Running; observed {profile_status!r}"
        )

    status_values: dict[str, list[str]] = {}

    def collect_status_values(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if isinstance(nested, str):
                    status_values.setdefault(str(key).lower(), []).append(
                        nested.lower()
                    )
                else:
                    collect_status_values(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_status_values(nested)

    collect_status_values(status)
    required_statuses = {
        "host": "running",
        "kubelet": "running",
        "apiserver": "running",
        "kubeconfig": "configured",
    }
    invalid_statuses = {
        key: status_values.get(key)
        for key, wanted in required_statuses.items()
        if wanted not in (status_values.get(key) or [])
    }
    if invalid_statuses:
        raise ValueError(
            f"Minikube component status is incomplete or unhealthy: {invalid_statuses}"
        )
    return {
        "profile": profile,
        "driver": "docker",
        "profile_status": profile_status,
        "version": version.strip(),
        "status": dict(status),
        "profile_record": dict(observed),
    }
