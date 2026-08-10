"""Article-reproduction scheduler using explicit single-node Pod binding.

This profile intentionally mirrors the article's manual-binding experiment and
is constrained to a dedicated, validated single-node target. Production
workloads should use :mod:`scheduler.gate_controller`, which leaves feasibility,
scoring and binding to the normal kube-scheduler.
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
    BurstCollector,
    BurstContractError,
    RunSettings,
    extract_features,
    group_pods_by_run,
    run_settings_for_pods,
)
from scheduler.config import ConfigurationError, SchedulerConfig  # noqa: E402
from scheduler.execution import (
    ExecutionStartError,
    wait_for_execution_start,
)  # noqa: E402
from scheduler.kube import (  # noqa: E402
    ApiFailureKind,
    KubernetesOperationError,
    api_timeout,
    call_with_retries,
    load_kubernetes_configuration,
    validate_target_node,
)
from scheduler.pacing import (  # noqa: E402
    MetricsSample,
    Pacer,
    PacingError,
    RealClusterFeedback,
)
from scheduler.rank import (  # noqa: E402
    JobFeatures,
    compute_workload_ranks,
    sort_workloads_by_rank,
)
from scheduler.records import AtomicRecordStore, ScheduleRecord  # noqa: E402
from scheduler.telemetry import (  # noqa: E402
    HealthServer,
    HealthState,
    JsonEventLogger,
    default_metrics,
)


class ManualBindingSafetyError(RuntimeError):
    pass


class MLAwareScheduler:
    """Reconcile and manually bind complete, explicitly identified bursts.

    The first nine arguments preserve the original constructor.  New safety
    arguments are keyword-only.  An explicit ``target_node`` is now mandatory;
    silently selecting the first API-list result is unsafe and non-deterministic.
    """

    def __init__(
        self,
        scheduler_name: str,
        namespace: str,
        pacing_mode: str,
        fixed_delay: float,
        cpu_threshold: float,
        max_wait: float,
        burst_window: float,
        reverse: bool,
        results_path: str,
        *,
        target_node: Optional[str] = None,
        run_id: Optional[str] = None,
        expected_count: Optional[int] = None,
        quiet_period: Optional[float] = None,
        burst_timeout: float = 120.0,
        poll_interval: float = 0.5,
        execution_timeout: float = 120.0,
        metrics_max_age: float = 30.0,
        adaptive_hysteresis: float = 0.05,
        api_timeout_seconds: float = 10.0,
        api_retries: int = 4,
        health_host: str = "0.0.0.0",
        health_port: int = 8080,
        core_api: Any = None,
        custom_api: Any = None,
        load_config: bool = True,
        enable_health_server: bool = True,
        logger: Optional[JsonEventLogger] = None,
        monotonic=time.monotonic,
        wall_time=time.time,
        sleep=time.sleep,
    ):
        self.config = SchedulerConfig(
            scheduler_name=scheduler_name,
            namespace=namespace,
            run_id=run_id,
            expected_count=expected_count,
            quiet_period=burst_window if quiet_period is None else quiet_period,
            burst_timeout=burst_timeout,
            poll_interval=poll_interval,
            pacing_mode=pacing_mode,
            fixed_delay=fixed_delay,
            cpu_threshold=cpu_threshold,
            adaptive_hysteresis=adaptive_hysteresis,
            max_wait=max_wait,
            metrics_max_age=metrics_max_age,
            execution_timeout=execution_timeout,
            api_timeout=api_timeout_seconds,
            api_retries=api_retries,
            target_node=target_node,
            reverse=reverse,
            results_path=results_path,
            health_host=health_host,
            health_port=health_port,
        ).validate(manual_binding=True)
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.sleep = sleep
        self.logger = logger or JsonEventLogger(
            static_fields={"component": "manual-bind-scheduler", "namespace": namespace}
        )
        self.metrics = default_metrics()
        self.health = HealthState()
        self.health_server = HealthServer(health_host, health_port, self.health, self.metrics)
        self.enable_health_server = enable_health_server
        self._stop = False
        self._processed_runs = set()
        self.records: List[ScheduleRecord] = []

        if core_api is None:
            if load_config:
                auth_source = load_kubernetes_configuration()
                self.logger.info("kubernetes_auth_loaded", source=auth_source)
            self.v1 = client.CoreV1Api()
        else:
            self.v1 = core_api
        self.custom_api = custom_api or client.CustomObjectsApi()

        self.node = validate_target_node(
            self.v1,
            self.config.target_node or "",
            timeout=self.config.api_timeout,
            retries=self.config.api_retries,
        )
        self.node_name = self.config.target_node
        self.feedback: Optional[RealClusterFeedback] = None
        self.health.set_ready(True, "target node and Kubernetes API validated")
        self.metrics.set("ml_scheduler_ready", 1)

    # Compatibility helpers retained for callers of the original prototype.
    def _extract_features(self, pod: Any) -> JobFeatures:
        return extract_features(pod)

    def _first_node_name(self) -> str:
        return str(self.node_name)

    def _pending_pods_for_me(self) -> List[Any]:
        return self._list_eligible_pods()

    def _wait_for_execution_start(self, pod_name: str, timeout: Optional[float] = None) -> float:
        return wait_for_execution_start(
            self.v1,
            pod_name,
            self.config.namespace,
            timeout=timeout or self.config.execution_timeout,
            api_timeout_seconds=self.config.api_timeout,
            api_retries=self.config.api_retries,
            poll_interval=self.config.poll_interval,
            monotonic=self.monotonic,
            sleep=self.sleep,
        )

    def _list_eligible_pods(self) -> List[Any]:
        response = call_with_retries(
            lambda: self.v1.list_namespaced_pod(
                self.config.namespace,
                field_selector="status.phase=Pending",
                _request_timeout=api_timeout(self.config.api_timeout),
            ),
            operation="list pending scheduler pods",
            retries=self.config.api_retries,
        )
        return [
            pod
            for pod in (getattr(response, "items", None) or [])
            if getattr(getattr(pod, "spec", None), "scheduler_name", None)
            == self.config.scheduler_name
            and getattr(getattr(pod, "spec", None), "node_name", None) is None
            and getattr(getattr(pod, "status", None), "phase", None) == "Pending"
        ]

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

    def _discover_settings(self, *, wait_forever: bool) -> RunSettings:
        deadline = None if wait_forever else self.monotonic() + self.config.burst_timeout
        while not self._stop and (deadline is None or self.monotonic() < deadline):
            pods = self._list_eligible_pods()
            groups = group_pods_by_run(pods)
            if self.config.run_id:
                selected = groups.get(self.config.run_id, [])
                if selected:
                    return run_settings_for_pods(
                        selected,
                        fallback_run_id=self.config.run_id,
                        fallback_expected_count=self.config.expected_count,
                        fallback_pacing_mode=self.config.pacing_mode,
                        fallback_fixed_delay=self.config.fixed_delay,
                        fallback_reverse=self.config.reverse,
                    )
            else:
                candidates = [
                    (run_id, group)
                    for run_id, group in groups.items()
                    if run_id not in self._processed_runs
                ]
                if candidates:
                    candidates.sort(key=lambda item: self._pod_order_key(item[1][0]))
                    _, selected = candidates[0]
                    return run_settings_for_pods(
                        selected,
                        fallback_run_id=None,
                        fallback_expected_count=None,
                        fallback_pacing_mode=self.config.pacing_mode,
                        fallback_fixed_delay=self.config.fixed_delay,
                        fallback_reverse=self.config.reverse,
                    )
            self.sleep(self.config.poll_interval)
        if self._stop:
            raise BurstContractError("scheduler stopped while waiting for a run")
        raise BurstContractError(
            f"no eligible run appeared within {self.config.burst_timeout:.3f}s"
        )

    def _collect(self, settings: RunSettings) -> List[Any]:
        collector = BurstCollector(
            self._list_eligible_pods,
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

    def _validate_pod_for_manual_binding(self, pod: Any) -> None:
        spec = getattr(pod, "spec", None)
        name = getattr(getattr(pod, "metadata", None), "name", "unknown")
        if getattr(spec, "affinity", None) is not None:
            raise ManualBindingSafetyError(
                f"pod {name!r} uses affinity; use the production gate controller"
            )
        if getattr(spec, "scheduling_gates", None):
            raise ManualBindingSafetyError(
                f"pod {name!r} uses scheduling gates; use the production gate controller"
            )
        if getattr(spec, "topology_spread_constraints", None):
            raise ManualBindingSafetyError(f"pod {name!r} uses topology spread constraints")
        for volume in getattr(spec, "volumes", None) or []:
            if getattr(volume, "persistent_volume_claim", None) is not None:
                raise ManualBindingSafetyError(
                    f"pod {name!r} uses a PVC; manual binding can bypass volume topology"
                )
        for container_spec in getattr(spec, "containers", None) or []:
            for port in getattr(container_spec, "ports", None) or []:
                if getattr(port, "host_port", None):
                    raise ManualBindingSafetyError(
                        f"pod {name!r} uses hostPort; manual binding cannot check conflicts"
                    )
        node_selector = getattr(spec, "node_selector", None) or {}
        node_labels = getattr(getattr(self.node, "metadata", None), "labels", None) or {}
        mismatches = [key for key, value in node_selector.items() if node_labels.get(key) != value]
        if mismatches:
            raise ManualBindingSafetyError(
                f"pod {name!r} nodeSelector does not match target node keys {mismatches}"
            )
        tolerations = getattr(spec, "tolerations", None) or []
        for taint in getattr(getattr(self.node, "spec", None), "taints", None) or []:
            if getattr(taint, "effect", None) not in {"NoSchedule", "NoExecute"}:
                continue
            if not any(self._tolerates(toleration, taint) for toleration in tolerations):
                raise ManualBindingSafetyError(
                    f"pod {name!r} does not tolerate target-node taint "
                    f"{getattr(taint, 'key', '')}={getattr(taint, 'value', '')}:"
                    f"{getattr(taint, 'effect', '')}"
                )

    @staticmethod
    def _tolerates(toleration: Any, taint: Any) -> bool:
        effect = getattr(toleration, "effect", None)
        if effect and effect != getattr(taint, "effect", None):
            return False
        operator = getattr(toleration, "operator", None) or "Equal"
        key = getattr(toleration, "key", None) or ""
        taint_key = getattr(taint, "key", None) or ""
        if operator == "Exists":
            return not key or key == taint_key
        if operator != "Equal" or key != taint_key:
            return False
        return (getattr(toleration, "value", None) or "") == (getattr(taint, "value", None) or "")

    def _bind_pod(self, pod_name: str) -> None:
        target = client.V1ObjectReference(kind="Node", api_version="v1", name=self.node_name)
        metadata = client.V1ObjectMeta(name=pod_name, namespace=self.config.namespace)
        body = client.V1Binding(target=target, metadata=metadata)
        try:
            call_with_retries(
                lambda: self.v1.create_namespaced_binding(
                    namespace=self.config.namespace,
                    body=body,
                    _request_timeout=api_timeout(self.config.api_timeout),
                ),
                operation=f"bind pod {pod_name} to {self.node_name}",
                retries=self.config.api_retries,
            )
        except KubernetesOperationError as exc:
            if exc.kind is not ApiFailureKind.CONFLICT:
                raise
            current = call_with_retries(
                lambda: self.v1.read_namespaced_pod(
                    pod_name,
                    self.config.namespace,
                    _request_timeout=api_timeout(self.config.api_timeout),
                ),
                operation=f"verify conflicting bind for {pod_name}",
                retries=self.config.api_retries,
            )
            assigned = getattr(getattr(current, "spec", None), "node_name", None)
            if assigned != self.node_name:
                raise

    def _feedback_for_adaptive(self) -> RealClusterFeedback:
        if self.feedback is None:
            self.feedback = RealClusterFeedback(
                self.v1,
                self.custom_api,
                str(self.node_name),
                api_timeout_seconds=self.config.api_timeout,
                api_retries=self.config.api_retries,
                wall_time=self.wall_time,
            )
        return self.feedback

    def _on_metrics_sample(self, sample: MetricsSample, age: float) -> None:
        self.metrics.set("ml_scheduler_cpu_utilization_ratio", sample.utilization)
        self.metrics.set("ml_scheduler_metrics_age_seconds", age)

    def _pacer(
        self,
        settings: RunSettings,
        *,
        store: Optional[AtomicRecordStore] = None,
    ) -> Pacer:
        feedback = self._feedback_for_adaptive() if settings.pacing_mode == "adaptive" else None
        last_recorded_timestamp: List[Optional[float]] = [None]

        def record_sample(sample: MetricsSample, age: float) -> None:
            self._on_metrics_sample(sample, age)
            if store is not None and sample.observed_at != last_recorded_timestamp[0]:
                store.append_event(
                    {
                        "event": "adaptive_metrics_sample",
                        "utilization": sample.utilization,
                        "observed_at": sample.observed_at,
                        "age_seconds": age,
                    }
                )
                last_recorded_timestamp[0] = sample.observed_at

        return Pacer(
            settings.pacing_mode,
            fixed_delay=settings.fixed_delay,
            feedback=feedback,
            cpu_threshold=self.config.cpu_threshold,
            hysteresis=self.config.adaptive_hysteresis,
            max_wait=self.config.max_wait,
            metrics_max_age=self.config.metrics_max_age,
            poll_interval=self.config.poll_interval,
            monotonic=self.monotonic,
            wall_time=self.wall_time,
            sleep=self.sleep,
            on_sample=record_sample,
        )

    def _pace(self) -> None:
        settings = RunSettings(
            self.config.run_id or "compatibility",
            self.config.expected_count or 1,
            self.config.pacing_mode,
            self.config.fixed_delay,
            self.config.reverse,
        )
        self._pacer(settings).wait()

    def _results_path(self, run_id: str, *, one_shot: bool) -> str:
        if "{run_id}" in self.config.results_path:
            safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in run_id)
            return self.config.results_path.format(run_id=safe)
        if one_shot:
            return self.config.results_path
        root, extension = os.path.splitext(self.config.results_path)
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in run_id)
        return f"{root}-{safe}{extension or '.json'}"

    def _record_metadata(self, settings: RunSettings) -> Dict[str, object]:
        return {
            "profile": "article-manual-bind",
            "run_id": settings.run_id,
            "expected_count": settings.expected_count,
            "scheduler_name": self.config.scheduler_name,
            "namespace": self.config.namespace,
            "target_node": self.node_name,
            "pacing_mode": settings.pacing_mode,
            "fixed_delay": settings.fixed_delay,
            "reverse": settings.reverse,
            "rank_policy": settings.rank_policy,
            **self.config.runtime_metadata(),
        }

    def process_run(self, settings: RunSettings, *, one_shot: bool = False) -> List[ScheduleRecord]:
        self.metrics.inc("ml_scheduler_bursts_total")
        self.logger.info(
            "burst_collection_started",
            run_id=settings.run_id,
            expected_count=settings.expected_count,
        )
        pods = self._collect(settings)
        self.metrics.set("ml_scheduler_burst_jobs", len(pods), {"run_id": settings.run_id})
        pods_by_name = {pod.metadata.name: pod for pod in pods}
        jobs = [extract_features(pod) for pod in pods]
        ranks = compute_workload_ranks(jobs, training_policy=settings.rank_policy)
        tie_keys = {pod.metadata.name: self._pod_order_key(pod) for pod in pods}
        ordered = sort_workloads_by_rank(
            jobs,
            reverse_order=settings.reverse,
            tie_breaker=lambda job: tie_keys[job.job_id],
            training_policy=settings.rank_policy,
        )
        store = AtomicRecordStore(
            self._results_path(settings.run_id, one_shot=one_shot),
            metadata=self._record_metadata(settings),
        )
        store.initialize()
        store.set_status("running")
        records = [
            ScheduleRecord(
                job.job_id,
                index,
                ranks[job.job_id],
                pod_uid=str(pods_by_name[job.job_id].metadata.uid),
            )
            for index, job in enumerate(ordered, start=1)
        ]
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
                pod = pods_by_name[job.job_id]
                self._validate_pod_for_manual_binding(pod)
                record.bind_time = self.wall_time()
                record.status = "binding"
                store.upsert(record)
                self._bind_pod(job.job_id)
                record.status = "bound"
                store.upsert(record)
                self.logger.info(
                    "pod_bound",
                    run_id=settings.run_id,
                    job_id=job.job_id,
                    target_node=self.node_name,
                    rank=record.rank,
                    order=record.order,
                )
                record.exec_start_time = self._wait_for_execution_start(job.job_id)
                record.status = "execution_started"
                store.upsert(record)
                self.metrics.inc("ml_scheduler_releases_total", labels={"profile": "manual-bind"})
                if index < len(records) - 1:
                    store.append_event(
                        {
                            "event": "pacing_wait_started",
                            "after_job_id": job.job_id,
                            "before_job_id": ordered[index + 1].job_id,
                            "mode": settings.pacing_mode,
                            "fixed_delay_seconds": settings.fixed_delay,
                            "timestamp": self.wall_time(),
                        }
                    )
                    try:
                        pacer.wait()
                    except Exception as exc:
                        store.append_event(
                            {
                                "event": "pacing_wait_failed",
                                "after_job_id": job.job_id,
                                "mode": settings.pacing_mode,
                                "timestamp": self.wall_time(),
                                "error": str(exc),
                            }
                        )
                        raise
                    store.append_event(
                        {
                            "event": "pacing_wait_completed",
                            "after_job_id": job.job_id,
                            "before_job_id": ordered[index + 1].job_id,
                            "mode": settings.pacing_mode,
                            "timestamp": self.wall_time(),
                        }
                    )
            store.set_status("completed")
            self.records.extend(records)
            self.logger.info(
                "run_completed",
                run_id=settings.run_id,
                result_path=store.path,
                jobs=len(records),
            )
            return records
        except Exception as exc:
            current = next(
                (record for record in records if record.status in {"binding", "bound"}),
                None,
            )
            if current is not None:
                current.status = "failed"
                current.error = str(exc)
                store.upsert(current)
            store.set_status("failed", str(exc))
            self.metrics.inc(
                "ml_scheduler_failures_total",
                labels={"stage": type(exc).__name__},
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

    def stop(self) -> None:
        self._stop = True
        self.health.set_ready(False, "shutdown requested")
        self.metrics.set("ml_scheduler_ready", 0)
        if self.enable_health_server:
            self.health_server.stop()

    def run_once(self) -> List[ScheduleRecord]:
        self._start_health()
        settings = self._discover_settings(wait_forever=False)
        try:
            return self.process_run(settings, one_shot=True)
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
            store = AtomicRecordStore(path, metadata=self._record_metadata(settings))
            store.initialize()
            store.set_status("failed", str(exc))
            self.metrics.inc(
                "ml_scheduler_failures_total",
                labels={"stage": type(exc).__name__},
            )
        self.logger.error(
            "run_rejected",
            run_id=settings.run_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def run_forever(self) -> None:
        self._start_health()
        self.logger.info("scheduler_ready", target_node=self.node_name)
        try:
            while not self._stop:
                settings = self._discover_settings(wait_forever=True)
                try:
                    self.process_run(settings, one_shot=False)
                except Exception as exc:
                    # A malformed run is isolated; a unique run-id prevents an
                    # immediate retry loop while leaving the controller healthy.
                    self._preserve_unhandled_failure(settings, exc)
                finally:
                    self._processed_runs.add(settings.run_id)
        finally:
            self.stop()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-name", default="ml-aware-scheduler")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--target-node", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--pacing-mode", choices=["none", "fixed", "adaptive"], default="none")
    parser.add_argument("--fixed-delay", type=float, default=0.0)
    parser.add_argument("--cpu-threshold", type=float, default=0.85)
    parser.add_argument("--adaptive-hysteresis", type=float, default=0.05)
    parser.add_argument("--max-wait", type=float, default=30.0)
    parser.add_argument("--metrics-max-age", type=float, default=30.0)
    parser.add_argument(
        "--burst-window",
        "--quiet-period",
        dest="quiet_period",
        type=float,
        default=1.5,
        help="membership must remain unchanged for this duration",
    )
    parser.add_argument("--burst-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--api-timeout", type=float, default=10.0)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--results", default="../results/schedule_run.json")
    parser.add_argument("--health-host", default="0.0.0.0")
    parser.add_argument("--health-port", type=int, default=8080)
    parser.add_argument(
        "--once",
        action="store_true",
        help="process one run and exit; default is a multi-run controller",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        scheduler = MLAwareScheduler(
            scheduler_name=args.scheduler_name,
            namespace=args.namespace,
            pacing_mode=args.pacing_mode,
            fixed_delay=args.fixed_delay,
            cpu_threshold=args.cpu_threshold,
            max_wait=args.max_wait,
            burst_window=args.quiet_period,
            reverse=args.reverse,
            results_path=args.results,
            target_node=args.target_node,
            run_id=args.run_id,
            expected_count=args.expected_count,
            burst_timeout=args.burst_timeout,
            poll_interval=args.poll_interval,
            execution_timeout=args.execution_timeout,
            metrics_max_age=args.metrics_max_age,
            adaptive_hysteresis=args.adaptive_hysteresis,
            api_timeout_seconds=args.api_timeout,
            api_retries=args.api_retries,
            health_host=args.health_host,
            health_port=args.health_port,
        )
    except ConfigurationError as exc:
        parser.error(str(exc))
    signal.signal(signal.SIGTERM, lambda *_args: scheduler.stop())
    signal.signal(signal.SIGINT, lambda *_args: scheduler.stop())
    try:
        if args.once:
            scheduler.run_once()
        else:
            scheduler.run_forever()
        return 0
    except (
        BurstContractError,
        ExecutionStartError,
        KubernetesOperationError,
        PacingError,
    ) as exc:
        scheduler.logger.error("scheduler_terminated", error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
