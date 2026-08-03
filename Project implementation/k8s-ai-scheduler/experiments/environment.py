"""Fail-closed validation for the article's experimental environment."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

GIB = 1024**3
_QUANTITY_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([EPTGMK]i?|m|u|n)?$")
_DECIMAL_MULTIPLIERS = {
    "": 1.0,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
}
_BINARY_MULTIPLIERS = {
    "Ki": 1024.0,
    "Mi": 1024.0**2,
    "Gi": 1024.0**3,
    "Ti": 1024.0**4,
    "Pi": 1024.0**5,
    "Ei": 1024.0**6,
}


class ArticleEnvironmentError(RuntimeError):
    def __init__(self, errors: list[str], evidence: Mapping[str, Any]):
        super().__init__("article environment validation failed: " + "; ".join(errors))
        self.errors = tuple(errors)
        self.evidence = dict(evidence)


def parse_quantity(value: Any) -> float:
    """Parse the Kubernetes quantity forms used by node capacity fields."""

    text = str(value or "").strip()
    match = _QUANTITY_RE.fullmatch(text)
    if not match:
        raise ValueError(f"unsupported Kubernetes quantity: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2) or ""
    multiplier = _BINARY_MULTIPLIERS.get(suffix, _DECIMAL_MULTIPLIERS.get(suffix))
    if multiplier is None or not math.isfinite(number):
        raise ValueError(f"unsupported Kubernetes quantity: {value!r}")
    return number * multiplier


def _active_non_experiment_pods(
    snapshot: Mapping[str, Any], *, experiment_namespace: str
) -> list[str]:
    scheduler_names = {
        str(pod.get("name"))
        for pod in ((snapshot.get("scheduler_deployment") or {}).get("pods") or [])
        if pod.get("name")
    }
    unexpected: list[str] = []
    for pod in snapshot.get("target_node_pods") or []:
        phase = str(pod.get("phase") or "")
        if phase in {"Succeeded", "Failed"}:
            continue
        namespace = str(pod.get("namespace") or "")
        name = str(pod.get("name") or "")
        if namespace == "kube-system":
            continue
        if namespace == experiment_namespace and name in scheduler_names:
            continue
        unexpected.append(f"{namespace}/{name}({phase or 'Unknown'})")
    return sorted(unexpected)


def validate_article_environment(
    snapshot: Mapping[str, Any],
    *,
    experiment_namespace: str,
    profile: str = "article-exact",
) -> dict[str, Any]:
    """Validate the published 1-node Minikube 4-CPU/8-GiB setup.

    ``record-only`` records every mismatch but deliberately makes the result
    ineligible for an article reproduction claim.  It is useful for smoke
    testing on shared K3s clusters without weakening the article profile.
    """

    if profile not in {"article-exact", "record-only"}:
        raise ValueError("profile must be article-exact or record-only")
    errors: list[str] = []
    nodes = snapshot.get("cluster_nodes") or []
    target = snapshot.get("target_node") or {}
    capacity = target.get("capacity") or {}
    labels = target.get("labels") or {}
    conditions = target.get("conditions") or []

    if len(nodes) != 1:
        errors.append(f"expected exactly one Kubernetes node; observed {len(nodes)}")
    context = str(snapshot.get("kubectl_context") or "").lower()
    minikube_label = any(str(key).startswith("minikube.k8s.io/") for key in labels)
    if "minikube" not in context and not minikube_label:
        errors.append("cluster is not identifiable as Minikube")
    try:
        cpu = parse_quantity(capacity.get("cpu"))
        if not math.isclose(cpu, 4.0, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"expected 4 CPU cores; observed {capacity.get('cpu')!r}")
    except ValueError as exc:
        errors.append(str(exc))
        cpu = None
    try:
        memory_bytes = parse_quantity(capacity.get("memory"))
        if not 7.5 * GIB <= memory_bytes <= 8.5 * GIB:
            errors.append(
                f"expected approximately 8 GiB memory; observed {capacity.get('memory')!r}"
            )
    except ValueError as exc:
        errors.append(str(exc))
        memory_bytes = None

    condition_map = {
        str(condition.get("type")): str(condition.get("status"))
        for condition in conditions
    }
    if condition_map.get("Ready") != "True":
        errors.append("target node is not Ready")
    for pressure in ("MemoryPressure", "DiskPressure", "PIDPressure"):
        if condition_map.get(pressure) != "False":
            errors.append(f"target node {pressure} is not False")

    unexpected = _active_non_experiment_pods(
        snapshot, experiment_namespace=experiment_namespace
    )
    if unexpected:
        errors.append("unexpected active target-node Pods: " + ", ".join(unexpected))

    evidence: dict[str, Any] = {
        "profile": profile,
        "article_claim_eligible": profile == "article-exact" and not errors,
        "expected": {
            "cluster": "Minikube",
            "nodes": 1,
            "cpu_cores": 4,
            "memory_gib_range": [7.5, 8.5],
            "active_non_system_pods": "reproduction scheduler only",
        },
        "observed": {
            "nodes": len(nodes),
            "cpu_cores": cpu,
            "memory_bytes": memory_bytes,
            "unexpected_active_pods": unexpected,
        },
        "errors": errors,
    }
    if errors and profile == "article-exact":
        raise ArticleEnvironmentError(errors, evidence)
    return evidence
