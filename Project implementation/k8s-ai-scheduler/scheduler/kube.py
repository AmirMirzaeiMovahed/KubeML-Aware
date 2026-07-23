"""Kubernetes API boundary: authentication, quantities, retries and nodes."""

from __future__ import annotations

import math
import random
import re
import time
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, Optional, Tuple

from kubernetes import config
from kubernetes.client import exceptions as kube_exceptions
from kubernetes.config.config_exception import ConfigException


class ApiFailureKind(str, Enum):
    TRANSIENT = "transient"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    INVALID = "invalid"
    PERMANENT = "permanent"


class KubernetesOperationError(RuntimeError):
    def __init__(self, operation: str, kind: ApiFailureKind, cause: Exception):
        super().__init__(f"{operation} failed ({kind.value}): {cause}")
        self.operation = operation
        self.kind = kind
        self.cause = cause


class NodeValidationError(RuntimeError):
    pass


_CPU_QUANTITY = re.compile(
    r"^(?P<number>\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>n|u|m|k|K|M|G|T|P|E)?$"
)
_CPU_MULTIPLIERS = {
    "n": Decimal("1e-9"),
    "u": Decimal("1e-6"),
    "m": Decimal("1e-3"),
    "": Decimal(1),
    "k": Decimal("1e3"),
    "K": Decimal("1e3"),
    "M": Decimal("1e6"),
    "G": Decimal("1e9"),
    "T": Decimal("1e12"),
    "P": Decimal("1e15"),
    "E": Decimal("1e18"),
}


def parse_cpu_quantity(quantity: str) -> float:
    """Parse a Kubernetes DecimalSI CPU quantity into cores.

    BinarySI quantities are deliberately rejected: Kubernetes does not permit
    fractional CPU to be expressed with memory suffixes such as ``Mi``.
    """

    if not isinstance(quantity, str):
        raise ValueError("CPU quantity must be a string")
    text = quantity.strip()
    match = _CPU_QUANTITY.fullmatch(text)
    if not match:
        raise ValueError(f"invalid Kubernetes CPU quantity: {quantity!r}")
    try:
        value = Decimal(match.group("number")) * _CPU_MULTIPLIERS[
            match.group("suffix") or ""
        ]
    except (InvalidOperation, KeyError) as exc:
        raise ValueError(f"invalid Kubernetes CPU quantity: {quantity!r}") from exc
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"CPU quantity must be finite and non-negative: {quantity!r}")
    return result


def load_kubernetes_configuration() -> str:
    """Load in-cluster credentials first and kubeconfig only as a local fallback."""

    try:
        config.load_incluster_config()
        return "in-cluster"
    except ConfigException:
        config.load_kube_config()
        return "kubeconfig"


def classify_api_exception(exc: Exception, *, log_read: bool = False) -> ApiFailureKind:
    status = getattr(exc, "status", None)
    if status in (401, 403):
        return ApiFailureKind.FORBIDDEN
    if status == 404:
        return ApiFailureKind.NOT_FOUND
    if status == 409:
        return ApiFailureKind.CONFLICT
    if status in (408, 425, 429) or (isinstance(status, int) and status >= 500):
        return ApiFailureKind.TRANSIENT
    if status == 400:
        body = " ".join(
            str(value).lower()
            for value in (getattr(exc, "reason", ""), getattr(exc, "body", ""))
        )
        if log_read and any(
            token in body
            for token in (
                "waiting to start",
                "containercreating",
                "pod initializing",
                "is not available",
            )
        ):
            return ApiFailureKind.TRANSIENT
        return ApiFailureKind.INVALID
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ApiFailureKind.TRANSIENT
    if isinstance(exc, kube_exceptions.ApiException) and status in (None, 0):
        return ApiFailureKind.TRANSIENT
    return ApiFailureKind.PERMANENT


def call_with_retries(
    function: Callable[[], Any],
    *,
    operation: str,
    retries: int = 4,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    log_read: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
) -> Any:
    """Execute one bounded API operation with exponential-jitter retries."""

    attempt = 0
    while True:
        try:
            return function()
        except Exception as exc:
            kind = classify_api_exception(exc, log_read=log_read)
            if kind is not ApiFailureKind.TRANSIENT or attempt >= retries:
                raise KubernetesOperationError(operation, kind, exc) from exc
            delay = min(max_delay, base_delay * (2**attempt))
            sleep(delay * (0.75 + 0.5 * random_value()))
            attempt += 1


def api_timeout(value: float) -> Tuple[float, float]:
    """Return the connect/read timeout tuple understood by the Python client."""

    return (min(3.0, value), value)


def validate_target_node(core_api: Any, node_name: str, *, timeout: float, retries: int):
    """Validate the explicit manual-bind target and return its Node object."""

    if not node_name or not node_name.strip():
        raise NodeValidationError("an explicit target node is required")
    try:
        node = call_with_retries(
            lambda: core_api.read_node(
                node_name, _request_timeout=api_timeout(timeout)
            ),
            operation=f"read target node {node_name}",
            retries=retries,
        )
    except KubernetesOperationError as exc:
        raise NodeValidationError(str(exc)) from exc

    if getattr(getattr(node, "metadata", None), "deletion_timestamp", None) is not None:
        raise NodeValidationError(f"target node {node_name!r} is being deleted")
    if bool(getattr(getattr(node, "spec", None), "unschedulable", False)):
        raise NodeValidationError(f"target node {node_name!r} is unschedulable")
    conditions = getattr(getattr(node, "status", None), "conditions", None) or []
    ready = next(
        (
            str(getattr(condition, "status", "")).lower() == "true"
            for condition in conditions
            if getattr(condition, "type", None) == "Ready"
        ),
        False,
    )
    if not ready:
        raise NodeValidationError(f"target node {node_name!r} is not Ready")
    allocatable = getattr(getattr(node, "status", None), "allocatable", None) or {}
    try:
        cores = parse_cpu_quantity(allocatable["cpu"])
    except (KeyError, ValueError) as exc:
        raise NodeValidationError(
            f"target node {node_name!r} has invalid allocatable CPU"
        ) from exc
    if cores <= 0:
        raise NodeValidationError(f"target node {node_name!r} has no allocatable CPU")
    return node
