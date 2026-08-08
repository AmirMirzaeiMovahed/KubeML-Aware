"""Fixed and resource-feedback pacing with bounded monotonic waits."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from .kube import api_timeout, call_with_retries, parse_cpu_quantity


class FeedbackUnavailable(RuntimeError):
    pass


class PacingError(RuntimeError):
    pass


class PacingInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class MetricsSample:
    utilization: float
    observed_at: float
    allocatable_cores: Optional[float] = None


def _parse_rfc3339(value: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError("metrics timestamp is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("metrics timestamp must include a timezone")
    return parsed.timestamp()


class RealClusterFeedback:
    """Read one node's Metrics API sample; never invent a zero on failure."""

    def __init__(
        self,
        core_api: Any,
        custom_api: Any,
        node_name: str,
        *,
        api_timeout_seconds: float = 10.0,
        api_retries: int = 2,
        wall_time: Callable[[], float] = time.time,
    ):
        self.core_api = core_api
        self.custom_api = custom_api
        self.node_name = node_name
        self.api_timeout_seconds = api_timeout_seconds
        self.api_retries = api_retries
        self.wall_time = wall_time
        node = call_with_retries(
            lambda: self.core_api.read_node(
                node_name, _request_timeout=api_timeout(api_timeout_seconds)
            ),
            operation=f"read feedback node {node_name}",
            retries=api_retries,
        )
        allocatable = getattr(getattr(node, "status", None), "allocatable", None) or {}
        try:
            self.allocatable_cores = parse_cpu_quantity(allocatable["cpu"])
        except (KeyError, ValueError) as exc:
            raise FeedbackUnavailable(
                f"node {node_name!r} has invalid allocatable CPU"
            ) from exc
        if self.allocatable_cores <= 0:
            raise FeedbackUnavailable(f"node {node_name!r} has no allocatable CPU")

    def sample(self) -> MetricsSample:
        try:
            metrics = call_with_retries(
                lambda: self.custom_api.get_cluster_custom_object(
                    "metrics.k8s.io",
                    "v1beta1",
                    "nodes",
                    self.node_name,
                    _request_timeout=api_timeout(self.api_timeout_seconds),
                ),
                operation=f"read CPU metrics for node {self.node_name}",
                retries=self.api_retries,
            )
            usage = metrics.get("usage") or {}
            used_cores = parse_cpu_quantity(usage["cpu"])
            observed_at = _parse_rfc3339(metrics["timestamp"])
            utilization = used_cores / self.allocatable_cores
            if utilization < 0:
                raise ValueError("negative utilization")
            return MetricsSample(
                utilization=utilization,
                observed_at=observed_at,
                allocatable_cores=self.allocatable_cores,
            )
        except Exception as exc:
            if isinstance(exc, FeedbackUnavailable):
                raise
            raise FeedbackUnavailable(str(exc)) from exc


class ClusterMetricsFeedback:
    """Aggregate CPU usage across every Ready, schedulable cluster node.

    Production gate release cannot predict the node that kube-scheduler will
    choose.  A cluster-wide ratio is therefore safer than watching an arbitrary
    node.  Missing metrics for any included node invalidate the entire sample.
    """

    def __init__(
        self,
        core_api: Any,
        custom_api: Any,
        *,
        api_timeout_seconds: float = 10.0,
        api_retries: int = 2,
    ):
        self.core_api = core_api
        self.custom_api = custom_api
        self.api_timeout_seconds = api_timeout_seconds
        self.api_retries = api_retries

    @staticmethod
    def _ready(node: Any) -> bool:
        if getattr(getattr(node, "spec", None), "unschedulable", False):
            return False
        return any(
            getattr(condition, "type", None) == "Ready"
            and str(getattr(condition, "status", "")).lower() == "true"
            for condition in (
                getattr(getattr(node, "status", None), "conditions", None) or []
            )
        )

    def sample(self) -> MetricsSample:
        try:
            nodes_response = call_with_retries(
                lambda: self.core_api.list_node(
                    _request_timeout=api_timeout(self.api_timeout_seconds)
                ),
                operation="list nodes for aggregate CPU feedback",
                retries=self.api_retries,
            )
            nodes = [
                node
                for node in (getattr(nodes_response, "items", None) or [])
                if self._ready(node)
            ]
            capacities = {}
            for node in nodes:
                name = getattr(getattr(node, "metadata", None), "name", None)
                allocatable = (
                    getattr(getattr(node, "status", None), "allocatable", None) or {}
                )
                if not name:
                    raise ValueError("Ready node is missing metadata.name")
                capacities[name] = parse_cpu_quantity(allocatable["cpu"])
            if not capacities or any(value <= 0 for value in capacities.values()):
                raise ValueError("no Ready nodes with positive allocatable CPU")

            response = call_with_retries(
                lambda: self.custom_api.list_cluster_custom_object(
                    "metrics.k8s.io",
                    "v1beta1",
                    "nodes",
                    _request_timeout=api_timeout(self.api_timeout_seconds),
                ),
                operation="list node CPU metrics",
                retries=self.api_retries,
            )
            by_name = {
                item.get("metadata", {}).get("name"): item
                for item in (response.get("items") or [])
            }
            missing = sorted(set(capacities) - set(by_name))
            if missing:
                raise ValueError(f"metrics missing for Ready nodes: {missing}")
            used = 0.0
            observed = []
            for name in capacities:
                item = by_name[name]
                used += parse_cpu_quantity(item["usage"]["cpu"])
                observed.append(_parse_rfc3339(item["timestamp"]))
            return MetricsSample(
                utilization=used / sum(capacities.values()),
                # The oldest component determines aggregate freshness.
                observed_at=min(observed),
                allocatable_cores=sum(capacities.values()),
            )
        except Exception as exc:
            if isinstance(exc, FeedbackUnavailable):
                raise
            raise FeedbackUnavailable(str(exc)) from exc


class Pacer:
    def __init__(
        self,
        mode: str,
        *,
        fixed_delay: float = 0.0,
        feedback: Optional[Any] = None,
        cpu_threshold: float = 0.85,
        hysteresis: float = 0.05,
        max_wait: float = 30.0,
        metrics_max_age: float = 30.0,
        poll_interval: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        on_sample: Optional[Callable[[MetricsSample, float], None]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ):
        if mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("invalid pacing mode")
        if fixed_delay < 0 or max_wait <= 0 or metrics_max_age <= 0 or poll_interval <= 0:
            raise ValueError("pacing durations are invalid")
        if not 0 < cpu_threshold <= 1 or not 0 <= hysteresis < cpu_threshold:
            raise ValueError("adaptive thresholds are invalid")
        if mode == "adaptive" and feedback is None:
            raise ValueError("adaptive pacing requires a feedback provider")
        self.mode = mode
        self.fixed_delay = fixed_delay
        self.feedback = feedback
        self.cpu_threshold = cpu_threshold
        self.hysteresis = hysteresis
        self.max_wait = max_wait
        self.metrics_max_age = metrics_max_age
        self.poll_interval = poll_interval
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.sleep = sleep
        self.on_sample = on_sample
        self.stop_requested = stop_requested

    def _check_stop(self) -> None:
        if self.stop_requested is not None and self.stop_requested():
            raise PacingInterrupted("shutdown requested during pacing")

    def wait(self) -> None:
        self._check_stop()
        if self.mode == "none":
            return
        if self.mode == "fixed":
            self._sleep_until(self.monotonic() + self.fixed_delay)
            return
        self._wait_adaptive()

    def _sleep_until(self, deadline: float) -> None:
        while True:
            self._check_stop()
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                return
            self.sleep(min(self.poll_interval, remaining))

    def _wait_adaptive(self) -> None:
        deadline = self.monotonic() + self.max_wait
        low_watermark = self.cpu_threshold - self.hysteresis
        consecutive_fresh_low = 0
        last_counted_timestamp: Optional[float] = None
        last_error: Optional[str] = None

        while self.monotonic() < deadline:
            self._check_stop()
            try:
                sample = self.feedback.sample()  # type: ignore[union-attr]
                age = max(0.0, self.wall_time() - sample.observed_at)
                if self.on_sample is not None:
                    self.on_sample(sample, age)
                if age > self.metrics_max_age:
                    last_error = (
                        f"metrics sample is stale ({age:.3f}s > {self.metrics_max_age:.3f}s)"
                    )
                    consecutive_fresh_low = 0
                elif sample.utilization <= low_watermark:
                    if sample.observed_at == last_counted_timestamp:
                        last_error = (
                            "metrics timestamp has not advanced; waiting for a new "
                            "collection sample"
                        )
                    else:
                        last_counted_timestamp = sample.observed_at
                        consecutive_fresh_low += 1
                        if consecutive_fresh_low >= 2:
                            return
                else:
                    # The band between low_watermark and cpu_threshold prevents
                    # a single borderline sample from releasing another job.
                    consecutive_fresh_low = 0
                    last_counted_timestamp = None
                    last_error = (
                        f"CPU utilization {sample.utilization:.3f} is above release "
                        f"watermark {low_watermark:.3f}"
                    )
            except FeedbackUnavailable as exc:
                consecutive_fresh_low = 0
                last_error = f"metrics unavailable: {exc}"

            remaining = deadline - self.monotonic()
            if remaining > 0:
                self.sleep(min(self.poll_interval, remaining))

        self._check_stop()

        raise PacingError(
            "adaptive pacing deadline expired without two fresh low-utilization "
            f"samples; last observation: {last_error or 'none'}"
        )
