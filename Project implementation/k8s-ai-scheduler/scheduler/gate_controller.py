"""Production ML ordering controller based on stable Pod scheduling gates.

Pods are created with ``spec.schedulingGates: [{name:
ml.scheduler/release}]``.  This controller removes only that gate in ranked
order.  The normal kube-scheduler remains responsible for filtering, scoring,
volume topology, taints, affinity and binding.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kubernetes import client, watch  # noqa: E402

from scheduler.burst import (  # noqa: E402
    AnnotationValidationError,
    BurstCollector,
    BurstContractError,
    RunSettings,
    extract_features,
    group_pods_by_run,
    pod_run_id,
    run_settings_for_pods,
)
from scheduler.config import ConfigurationError, SchedulerConfig  # noqa: E402
from scheduler.constants import (  # noqa: E402
    RELEASE_GATE,
    RUN_ID_ANNOTATION,
    RUN_ID_LABEL,
)
from scheduler.execution import (  # noqa: E402
    ExecutionStartError,
    execution_container_for_pod,
    wait_for_execution_start,
)
from scheduler.kube import (  # noqa: E402
    ApiFailureKind,
    KubernetesOperationError,
    api_timeout,
    call_with_retries,
    load_kubernetes_configuration,
)
from scheduler.pacing import (  # noqa: E402
    ClusterMetricsFeedback,
    MetricsSample,
    Pacer,
    PacingError,
    PacingInterrupted,
)
from scheduler.rank import compute_ranks, sort_by_rank  # noqa: E402
from scheduler.records import (  # noqa: E402
    AtomicRecordStore,
    RecordStoreError,
    ScheduleRecord,
)
from scheduler.telemetry import (  # noqa: E402
    HealthServer,
    HealthState,
    JsonEventLogger,
    default_metrics,
)


class GateReleaseError(RuntimeError):
    pass


class ControllerStopping(RuntimeError):
    pass


class SchedulingGateController:
    def __init__(
        self,
        config: SchedulerConfig,
        *,
        gate_name: str = RELEASE_GATE,
        core_api: Any = None,
        custom_api: Any = None,
        load_config: bool = True,
        enable_health_server: bool = True,
        logger: Optional[JsonEventLogger] = None,
        monotonic=time.monotonic,
        wall_time=time.time,
        sleep=time.sleep,
    ):
        self.config = config.validate(manual_binding=False)
        if not gate_name:
            raise ConfigurationError("gate_name must not be empty")
        self.gate_name = gate_name
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.sleep = sleep
        self.logger = logger or JsonEventLogger(
            static_fields={"component": "scheduling-gate-controller", "namespace": config.namespace}
        )
        self.metrics = default_metrics()
        self.health = HealthState()
        self.health_server = HealthServer(
            config.health_host, config.health_port, self.health, self.metrics
        )
        self.enable_health_server = enable_health_server
        self._stop = False
        self._processed_runs = set()
        self._failed_runs = set()
        self.records: List[ScheduleRecord] = []
        self.feedback: Optional[ClusterMetricsFeedback] = None

        if core_api is None:
            if load_config:
                source = load_kubernetes_configuration()
                self.logger.info("kubernetes_auth_loaded", source=source)
            self.v1 = client.CoreV1Api()
        else:
            self.v1 = core_api
        self.custom_api = custom_api or client.CustomObjectsApi()

        # A successful narrow list proves credentials and RBAC before readiness.
        call_with_retries(
            lambda: self.v1.list_namespaced_pod(
                self.config.namespace,
                limit=1,
                _request_timeout=api_timeout(self.config.api_timeout),
            ),
            operation="validate pod list permission",
            retries=self.config.api_retries,
        )
        self.health.set_ready(True, "Kubernetes API and namespace access validated")
        self.metrics.set("ml_scheduler_ready", 1)

    def _has_gate(self, pod: Any) -> bool:
        return any(
            getattr(gate, "name", None) == self.gate_name
            for gate in (
                getattr(getattr(pod, "spec", None), "scheduling_gates", None) or []
            )
        )

    def _list_run_pods(self) -> List[Any]:
        response = call_with_retries(
            lambda: self.v1.list_namespaced_pod(
                self.config.namespace,
                _request_timeout=api_timeout(self.config.api_timeout),
            ),
            operation="list production run pods",
            retries=self.config.api_retries,
        )
        return [
            pod
            for pod in (getattr(response, "items", None) or [])
            if getattr(getattr(pod, "spec", None), "scheduler_name", None)
            == self.config.scheduler_name
            and (
                (getattr(getattr(pod, "metadata", None), "labels", None) or {}).get(
                    RUN_ID_LABEL
                )
                or (
                    getattr(getattr(pod, "metadata", None), "annotations", None)
                    or {}
                ).get(RUN_ID_ANNOTATION)
            )
        ]

    def _list_eligible_pods(self) -> List[Any]:
        """Compatibility name: includes gated and already released run members."""

        return self._list_run_pods()

    def _watch_for_change(self, timeout: float) -> None:
        if timeout < 1.0:
            self.sleep(timeout)
            return
        watcher = watch.Watch()
        try:
            stream = watcher.stream(
                self.v1.list_namespaced_pod,
                namespace=self.config.namespace,
                field_selector="status.phase=Pending",
                timeout_seconds=max(1, int(math.ceil(timeout))),
                _request_timeout=api_timeout(max(self.config.api_timeout, timeout + 1)),
            )
            next(iter(stream), None)
        finally:
            watcher.stop()

    @staticmethod
    def _pod_order_key(pod: Any):
        metadata = getattr(pod, "metadata", None)
        created = getattr(metadata, "creation_timestamp", None)
        if hasattr(created, "isoformat"):
            created = created.isoformat()
        return (
            str(created or ""),
            str(getattr(metadata, "uid", "")),
            str(getattr(metadata, "name", "")),
        )

    def _set_ready(self, ready: bool, reason: str) -> None:
        self.health.set_ready(ready, reason)
        self.metrics.set("ml_scheduler_ready", 1 if ready else 0)

    def _record_metadata(self, settings: RunSettings) -> Dict[str, object]:
        return {
            "profile": "production-scheduling-gate",
            "gate_name": self.gate_name,
            "run_id": settings.run_id,
            "expected_count": settings.expected_count,
            "scheduler_name": self.config.scheduler_name,
            "namespace": self.config.namespace,
            "pacing_mode": settings.pacing_mode,
            "fixed_delay": settings.fixed_delay,
            "reverse": settings.reverse,
            **self.config.runtime_metadata(),
        }

    def _persisted_candidate_status(
        self,
        settings: RunSettings,
        pods: Sequence[Any],
        *,
        one_shot: bool,
    ) -> Optional[str]:
        store = AtomicRecordStore(
            self._results_path(settings.run_id, one_shot=one_shot),
            metadata=self._record_metadata(settings),
        )
        try:
            if not store.load_existing():
                return None
        except RecordStoreError:
            # Let process_run surface the exact unsafe-state error and persist
            # controller degradation instead of silently skipping the run.
            return None
        observed = {
            (str(pod.metadata.name), str(pod.metadata.uid)) for pod in pods
        }
        recorded = {
            (str(record.get("job_id")), str(record.get("pod_uid")))
            for record in store.records
        }
        if observed != recorded:
            return None
        return store.status

    def _discover_settings(
        self, *, wait_forever: bool, one_shot: bool = False
    ) -> RunSettings:
        deadline = None if wait_forever else self.monotonic() + self.config.burst_timeout
        while not self._stop and (deadline is None or self.monotonic() < deadline):
            valid_pods = []
            for pod in self._list_run_pods():
                try:
                    pod_run_id(pod)
                except AnnotationValidationError as exc:
                    name = getattr(getattr(pod, "metadata", None), "name", "unknown")
                    self._set_ready(False, f"invalid run identity on Pod {name}")
                    self.logger.error(
                        "run_discovery_rejected", pod_name=name, error=str(exc)
                    )
                    continue
                valid_pods.append(pod)
            groups = group_pods_by_run(valid_pods)
            if self.config.run_id:
                candidates = [(self.config.run_id, groups.get(self.config.run_id, []))]
            else:
                candidates = [
                    (run_id, pods)
                    for run_id, pods in groups.items()
                    if run_id not in self._processed_runs
                ]
            candidates = [item for item in candidates if item[1]]
            candidates.sort(key=lambda item: self._pod_order_key(item[1][0]))
            for run_id, pods in candidates:
                if run_id in self._processed_runs:
                    continue
                try:
                    settings = run_settings_for_pods(
                        pods,
                        fallback_run_id=self.config.run_id,
                        fallback_expected_count=self.config.expected_count,
                        fallback_pacing_mode=self.config.pacing_mode,
                        fallback_fixed_delay=self.config.fixed_delay,
                        fallback_reverse=self.config.reverse,
                    )
                except (AnnotationValidationError, BurstContractError) as exc:
                    self._processed_runs.add(run_id)
                    self._failed_runs.add(run_id)
                    self._set_ready(False, f"run {run_id} metadata is invalid")
                    self.metrics.inc(
                        "ml_scheduler_failures_total",
                        labels={"stage": type(exc).__name__},
                    )
                    self.logger.error(
                        "run_discovery_rejected", run_id=run_id, error=str(exc)
                    )
                    continue
                persisted = self._persisted_candidate_status(
                    settings, pods, one_shot=one_shot
                )
                if persisted == "completed":
                    self._processed_runs.add(run_id)
                    continue
                if persisted == "failed":
                    self._processed_runs.add(run_id)
                    self._failed_runs.add(run_id)
                    self._set_ready(False, f"run {run_id} has failed persisted state")
                    continue
                return settings
            self.sleep(self.config.poll_interval)
        if self._stop:
            raise ControllerStopping("controller stopped while waiting for a run")
        raise BurstContractError(
            f"no runnable gated run appeared within {self.config.burst_timeout:.3f}s"
        )

    def _collect(self, settings: RunSettings) -> List[Any]:
        collector = BurstCollector(
            self._list_run_pods,
            quiet_period=self.config.quiet_period,
            timeout=self.config.burst_timeout,
            poll_interval=self.config.poll_interval,
            wait_for_change=self._watch_for_change,
            monotonic=self.monotonic,
            sleep=self.sleep,
        )
        pods = collector.collect(settings.run_id, settings.expected_count)
        verified = run_settings_for_pods(
            pods,
            fallback_run_id=settings.run_id,
            fallback_expected_count=settings.expected_count,
            fallback_pacing_mode=settings.pacing_mode,
            fallback_fixed_delay=settings.fixed_delay,
            fallback_reverse=settings.reverse,
        )
        if verified != settings:
            raise BurstContractError(
                f"run {settings.run_id!r} configuration changed during collection"
            )
        return pods

    def _remove_gate(self, pod_name: str) -> None:
        last_error: Optional[Exception] = None
        for _attempt in range(self.config.api_retries + 1):
            pod = call_with_retries(
                lambda: self.v1.read_namespaced_pod(
                    pod_name,
                    self.config.namespace,
                    _request_timeout=api_timeout(self.config.api_timeout),
                ),
                operation=f"read gated pod {pod_name}",
                retries=self.config.api_retries,
            )
            gates = getattr(getattr(pod, "spec", None), "scheduling_gates", None) or []
            index = next(
                (
                    index
                    for index, gate in enumerate(gates)
                    if getattr(gate, "name", None) == self.gate_name
                ),
                None,
            )
            if index is None:
                return
            resource_version = str(
                getattr(getattr(pod, "metadata", None), "resource_version", "")
            )
            patch = [
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": resource_version,
                },
                {
                    "op": "test",
                    "path": f"/spec/schedulingGates/{index}/name",
                    "value": self.gate_name,
                },
                {"op": "remove", "path": f"/spec/schedulingGates/{index}"},
            ]
            try:
                call_with_retries(
                    lambda _patch=patch: self.v1.patch_namespaced_pod(
                        pod_name,
                        self.config.namespace,
                        _patch,
                        _content_type="application/json-patch+json",
                        _request_timeout=api_timeout(self.config.api_timeout),
                    ),
                    operation=f"remove scheduling gate from {pod_name}",
                    retries=self.config.api_retries,
                )
                return
            except KubernetesOperationError as exc:
                last_error = exc
                if exc.kind not in {ApiFailureKind.CONFLICT, ApiFailureKind.INVALID}:
                    raise GateReleaseError(str(exc)) from exc
                self.sleep(min(0.2 * (2**_attempt), 2.0))
        raise GateReleaseError(
            f"could not remove gate from {pod_name} after concurrent updates: {last_error}"
        )

    def _wait_for_execution_start(self, pod_name: str, *, container: str) -> float:
        try:
            return wait_for_execution_start(
                self.v1,
                pod_name,
                self.config.namespace,
                timeout=self.config.execution_timeout,
                api_timeout_seconds=self.config.api_timeout,
                api_retries=self.config.api_retries,
                container=container,
                poll_interval=self.config.poll_interval,
                monotonic=self.monotonic,
                sleep=self.sleep,
                stop_requested=lambda: self._stop,
            )
        except InterruptedError as exc:
            raise ControllerStopping(str(exc)) from exc

    def _on_metrics_sample(self, sample: MetricsSample, age: float) -> None:
        self.metrics.set("ml_scheduler_cpu_utilization_ratio", sample.utilization)
        self.metrics.set("ml_scheduler_metrics_age_seconds", age)

    def _pacer(
        self,
        settings: RunSettings,
        *,
        store: Optional[AtomicRecordStore] = None,
    ) -> Pacer:
        if settings.pacing_mode == "adaptive" and self.feedback is None:
            self.feedback = ClusterMetricsFeedback(
                self.v1,
                self.custom_api,
                api_timeout_seconds=self.config.api_timeout,
                api_retries=self.config.api_retries,
            )
        last_recorded_timestamp: List[Optional[float]] = [None]

        def record_sample(sample: MetricsSample, age: float) -> None:
            self._on_metrics_sample(sample, age)
            if store is not None and sample.observed_at != last_recorded_timestamp[0]:
                store.append_event({
                    "event": "adaptive_metrics_sample",
                    "utilization": sample.utilization,
                    "observed_at": sample.observed_at,
                    "age_seconds": age,
                })
                last_recorded_timestamp[0] = sample.observed_at

        return Pacer(
            settings.pacing_mode,
            fixed_delay=settings.fixed_delay,
            feedback=self.feedback if settings.pacing_mode == "adaptive" else None,
            cpu_threshold=self.config.cpu_threshold,
            hysteresis=self.config.adaptive_hysteresis,
            max_wait=self.config.max_wait,
            metrics_max_age=self.config.metrics_max_age,
            poll_interval=self.config.poll_interval,
            monotonic=self.monotonic,
            wall_time=self.wall_time,
            sleep=self.sleep,
            on_sample=record_sample,
            stop_requested=lambda: self._stop,
        )

    def _results_path(self, run_id: str, *, one_shot: bool) -> str:
        safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in run_id)
        if "{run_id}" in self.config.results_path:
            return self.config.results_path.format(run_id=safe)
        if one_shot:
            return self.config.results_path
        root, extension = os.path.splitext(self.config.results_path)
        return f"{root}-{safe}{extension or '.json'}"

    def process_run(self, settings: RunSettings, *, one_shot: bool = False) -> List[ScheduleRecord]:
        self.metrics.inc("ml_scheduler_bursts_total")
        pods = self._collect(settings)
        self.metrics.set("ml_scheduler_burst_jobs", len(pods), {"run_id": settings.run_id})
        pods_by_name = {pod.metadata.name: pod for pod in pods}
        jobs = [extract_features(pod) for pod in pods]
        ranks = compute_ranks(jobs)
        tie_keys = {pod.metadata.name: self._pod_order_key(pod) for pod in pods}
        ordered = sort_by_rank(
            jobs,
            reverse_order=settings.reverse,
            tie_breaker=lambda job: tie_keys[job.job_id],
        )
        expected_records = [
            ScheduleRecord(
                job.job_id,
                index,
                ranks[job.job_id],
                pod_uid=str(pods_by_name[job.job_id].metadata.uid),
            )
            for index, job in enumerate(ordered, start=1)
        ]
        expected_by_job = {record.job_id: record for record in expected_records}
        store = AtomicRecordStore(
            self._results_path(settings.run_id, one_shot=one_shot),
            metadata=self._record_metadata(settings),
        )
        resumed = store.load_existing()
        if resumed:
            if store.status == "completed":
                raise RecordStoreError("completed scheduler state cannot be replayed")
            if store.status == "failed":
                raise RecordStoreError("failed scheduler state requires operator action")
            existing_by_job: Dict[str, ScheduleRecord] = {}
            for raw in store.records:
                try:
                    existing = ScheduleRecord(**raw)
                except (TypeError, ValueError) as exc:
                    raise RecordStoreError(
                        f"invalid persisted record for {raw.get('job_id')!r}"
                    ) from exc
                expected = expected_by_job.get(existing.job_id)
                if expected is None or (
                    existing.order,
                    existing.rank,
                    existing.pod_uid,
                ) != (expected.order, expected.rank, expected.pod_uid):
                    raise RecordStoreError(
                        f"persisted record drift for {existing.job_id!r}"
                    )
                if existing.status not in {
                    "ranked",
                    "releasing",
                    "released",
                    "execution_started",
                }:
                    raise RecordStoreError(
                        f"persisted record status is not resumable: {existing.status!r}"
                    )
                existing_by_job[existing.job_id] = existing
            records = [
                existing_by_job.get(expected.job_id, expected)
                for expected in expected_records
            ]
            store.set_status("running")
            for record in records:
                store.upsert(record)
            store.append_event({
                "event": "controller_resumed",
                "run_id": settings.run_id,
                "timestamp": self.wall_time(),
                "completed_jobs": sum(
                    record.status == "execution_started" for record in records
                ),
            })
            self.logger.info(
                "run_resumed",
                run_id=settings.run_id,
                completed_jobs=sum(
                    record.status == "execution_started" for record in records
                ),
            )
        else:
            if not all(self._has_gate(pod) for pod in pods):
                raise GateReleaseError(
                    "a new production run must start with the release gate on every Pod"
                )
            records = expected_records
            store.initialize()
            store.set_status("running")
            for record in records:
                store.upsert(record)
            self.logger.info(
                "burst_ranked",
                run_id=settings.run_id,
                order=[job.job_id for job in ordered],
                ranks=ranks,
            )
        pacer = self._pacer(settings, store=store)
        try:
            for index, (job, record) in enumerate(zip(ordered, records, strict=True)):
                if self._stop:
                    raise ControllerStopping("shutdown requested before next release")
                pod = pods_by_name[job.job_id]
                container = execution_container_for_pod(pod)
                if record.status in {"released", "execution_started"} and self._has_gate(pod):
                    raise RecordStoreError(
                        f"persisted {record.status} record {job.job_id!r} is gated"
                    )
                if record.status == "execution_started":
                    continue
                if resumed and record.status == "ranked" and not self._has_gate(pod):
                    raise RecordStoreError(
                        f"unrecorded external gate removal for {job.job_id!r}"
                    )
                if index > 0 and self._has_gate(pod):
                    previous = ordered[index - 1].job_id
                    pacing_done = any(
                        event.get("event") == "pacing_wait_completed"
                        and event.get("after_job_id") == previous
                        and event.get("before_job_id") == job.job_id
                        for event in store.events
                    )
                    if not pacing_done:
                        store.append_event({
                            "event": "pacing_wait_started",
                            "after_job_id": previous,
                            "before_job_id": job.job_id,
                            "mode": settings.pacing_mode,
                            "fixed_delay_seconds": settings.fixed_delay,
                            "timestamp": self.wall_time(),
                        })
                        try:
                            pacer.wait()
                        except PacingInterrupted as exc:
                            store.append_event({
                                "event": "pacing_wait_interrupted",
                                "after_job_id": previous,
                                "before_job_id": job.job_id,
                                "mode": settings.pacing_mode,
                                "timestamp": self.wall_time(),
                            })
                            raise ControllerStopping(str(exc)) from exc
                        except Exception as exc:
                            store.append_event({
                                "event": "pacing_wait_failed",
                                "after_job_id": previous,
                                "before_job_id": job.job_id,
                                "mode": settings.pacing_mode,
                                "timestamp": self.wall_time(),
                                "error": str(exc),
                            })
                            raise
                        store.append_event({
                            "event": "pacing_wait_completed",
                            "after_job_id": previous,
                            "before_job_id": job.job_id,
                            "mode": settings.pacing_mode,
                            "timestamp": self.wall_time(),
                        })
                if self._stop:
                    raise ControllerStopping("shutdown requested before gate removal")
                if self._has_gate(pod):
                    if record.release_time is None:
                        record.release_time = self.wall_time()
                    record.status = "releasing"
                    store.upsert(record)
                    self._remove_gate(job.job_id)
                    record.status = "released"
                    store.upsert(record)
                    self.logger.info(
                        "scheduling_gate_removed",
                        run_id=settings.run_id,
                        job_id=job.job_id,
                        rank=record.rank,
                        order=record.order,
                    )
                elif record.status == "releasing":
                    if record.release_time is None:
                        record.release_time = self.wall_time()
                    record.status = "released"
                    store.upsert(record)
                    store.append_event({
                        "event": "release_reconciled_after_restart",
                        "job_id": job.job_id,
                        "timestamp": self.wall_time(),
                    })
                record.exec_start_time = self._wait_for_execution_start(
                    job.job_id, container=container
                )
                record.status = "execution_started"
                store.upsert(record)
                self.metrics.inc(
                    "ml_scheduler_releases_total", labels={"profile": "scheduling-gate"}
                )
            store.set_status("completed")
            self.records.extend(records)
            if not self._failed_runs:
                self._set_ready(True, "Kubernetes API healthy; all observed runs reconciled")
            self.logger.info(
                "run_completed",
                run_id=settings.run_id,
                result_path=store.path,
                jobs=len(records),
            )
            return records
        except ControllerStopping:
            store.append_event({
                "event": "controller_stopped",
                "run_id": settings.run_id,
                "timestamp": self.wall_time(),
            })
            raise
        except Exception as exc:
            current = next(
                (record for record in records if record.status in {"releasing", "released"}),
                None,
            )
            if current is not None:
                current.status = "failed"
                current.error = str(exc)
                store.upsert(current)
            store.set_status("failed", str(exc))
            self._failed_runs.add(settings.run_id)
            self._set_ready(False, f"run {settings.run_id} failed: {type(exc).__name__}")
            self.metrics.inc(
                "ml_scheduler_failures_total", labels={"stage": type(exc).__name__}
            )
            self.logger.error(
                "run_failed",
                run_id=settings.run_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    def _start_health(self) -> None:
        if self.enable_health_server:
            self.health_server.start()

    def request_stop(self) -> None:
        self._stop = True
        self.health.set_ready(False, "shutdown requested")
        self.metrics.set("ml_scheduler_ready", 0)

    def stop(self) -> None:
        self.request_stop()
        if self.enable_health_server:
            self.health_server.stop()

    def run_once(self) -> List[ScheduleRecord]:
        self._start_health()
        settings = self._discover_settings(wait_forever=False, one_shot=True)
        try:
            return self.process_run(settings, one_shot=True)
        except ControllerStopping:
            raise
        except Exception as exc:
            self._preserve_unhandled_failure(settings, exc, one_shot=True)
            raise
        finally:
            if self.enable_health_server:
                self.health_server.stop()

    def _preserve_unhandled_failure(
        self, settings: RunSettings, exc: Exception, *, one_shot: bool = False
    ) -> None:
        path = self._results_path(settings.run_id, one_shot=one_shot)
        if not os.path.exists(path):
            store = AtomicRecordStore(
                path, metadata=self._record_metadata(settings)
            )
            store.initialize()
            store.set_status("failed", str(exc))
            self.metrics.inc(
                "ml_scheduler_failures_total",
                labels={"stage": type(exc).__name__},
            )
        self._failed_runs.add(settings.run_id)
        self._set_ready(False, f"run {settings.run_id} failed: {type(exc).__name__}")
        self.logger.error(
            "run_rejected",
            run_id=settings.run_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def run_forever(self) -> None:
        self._start_health()
        self.logger.info("controller_ready", gate_name=self.gate_name)
        try:
            while not self._stop:
                settings = self._discover_settings(wait_forever=True)
                try:
                    self.process_run(settings, one_shot=False)
                except ControllerStopping:
                    break
                except Exception as exc:
                    self._preserve_unhandled_failure(settings, exc)
                finally:
                    self._processed_runs.add(settings.run_id)
        finally:
            self.stop()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-name", default="default-scheduler")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--gate-name", default=RELEASE_GATE)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--pacing-mode", choices=["none", "fixed", "adaptive"], default="none")
    parser.add_argument("--fixed-delay", type=float, default=0.0)
    parser.add_argument("--cpu-threshold", type=float, default=0.85)
    parser.add_argument("--adaptive-hysteresis", type=float, default=0.05)
    parser.add_argument("--max-wait", type=float, default=30.0)
    parser.add_argument("--metrics-max-age", type=float, default=30.0)
    parser.add_argument("--quiet-period", "--burst-window", dest="quiet_period", type=float, default=1.5)
    parser.add_argument("--burst-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--api-timeout", type=float, default=10.0)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--results", default="../results/gate_schedule_run.json")
    parser.add_argument("--health-host", default="0.0.0.0")
    parser.add_argument("--health-port", type=int, default=8080)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = SchedulerConfig(
            scheduler_name=args.scheduler_name,
            namespace=args.namespace,
            run_id=args.run_id,
            expected_count=args.expected_count,
            quiet_period=args.quiet_period,
            burst_timeout=args.burst_timeout,
            poll_interval=args.poll_interval,
            pacing_mode=args.pacing_mode,
            fixed_delay=args.fixed_delay,
            cpu_threshold=args.cpu_threshold,
            adaptive_hysteresis=args.adaptive_hysteresis,
            max_wait=args.max_wait,
            metrics_max_age=args.metrics_max_age,
            execution_timeout=args.execution_timeout,
            api_timeout=args.api_timeout,
            api_retries=args.api_retries,
            reverse=args.reverse,
            results_path=args.results,
            health_host=args.health_host,
            health_port=args.health_port,
        )
        controller = SchedulingGateController(config, gate_name=args.gate_name)
    except ConfigurationError as exc:
        parser.error(str(exc))
    signal.signal(signal.SIGTERM, lambda *_args: controller.request_stop())
    signal.signal(signal.SIGINT, lambda *_args: controller.request_stop())
    try:
        if args.once:
            controller.run_once()
        else:
            controller.run_forever()
        return 0
    except ControllerStopping:
        controller.logger.info("controller_stopped", reason="shutdown requested")
        return 0
    except (
        BurstContractError,
        ExecutionStartError,
        GateReleaseError,
        KubernetesOperationError,
        PacingError,
    ) as exc:
        controller.logger.error("controller_terminated", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
