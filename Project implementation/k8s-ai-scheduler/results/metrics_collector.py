"""Strict collector for versioned Kubernetes experiment results.

Unlike the original collector, this module never drops failed Pods or missing
markers.  It emits a complete failure record and, in strict mode, raises
``IncompleteRunError`` after attaching the partial document.  The CLI always
writes that evidence before returning a non-zero status.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:  # permit pure parser/schema unit tests without a Kubernetes installation
    from kubernetes import client, config
except ImportError:  # pragma: no cover - exercised only in minimal environments
    client = None  # type: ignore[assignment]
    config = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from scheduler.rank import JobFeatures, compute_ranks  # noqa: E402
from workload.generate_workload import deterministic_job_seed  # noqa: E402
from k8s.work_model import (  # noqa: E402
    BLAS_THREADS_ANNOTATION,
    BLAS_THREADS_ENV,
    REPRODUCTION_BLAS_THREADS,
    WORK_MODEL_ANNOTATION,
    WORK_MODEL_ENV,
    WORK_MODEL_VERSION,
    estimate_work,
)
from experiments.schema import (  # noqa: E402
    IncompleteRunError,
    make_result_document,
    summarize_jobs,
    validate_result_document,
)


FEATURE_ANNOTATIONS = {
    "T": "ml.scheduler/estimated-training-time",
    "R": "ml.scheduler/loss-reduction-rate",
    "M": "ml.scheduler/matrix-size",
    "G": "ml.scheduler/gradient-update-size",
    "C": "ml.scheduler/checkpoint-interval",
    "P": "ml.scheduler/model-partitions",
}
LABELS = {
    "run_id": "ml.scheduler/run-id",
    "scenario": "ml.scheduler/scenario",
    "config": "ml.scheduler/config",
    "repetition": "ml.scheduler/repetition",
}
RUN_ANNOTATIONS = {
    "run_id": "ml.scheduler/run-id",
    "pacing_mode": "ml.scheduler/pacing-mode",
    "fixed_delay_seconds": "ml.scheduler/fixed-delay-seconds",
    "reverse": "ml.scheduler/reverse",
    "expected_jobs": "ml.scheduler/expected-jobs",
    "seed": "ml.scheduler/seed",
    "load_profile": "ml.scheduler/load-profile",
    "work_model_version": WORK_MODEL_ANNOTATION,
}
_LEGACY_LINE = re.compile(r"^\[(?P<timestamp>[0-9]+(?:\.[0-9]+)?)\]\s+(?P<event>[A-Z_]+)")


def _to_epoch(value: Any) -> float:
    if value is None:
        raise ValueError("timestamp is missing")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        result = value.timestamp()
    if not math.isfinite(result):
        raise ValueError("timestamp is not finite")
    return result


def parse_log_events(logs: str) -> Dict[str, Dict[str, Any]]:
    """Parse structured trainer events and the original ``[epoch] EVENT`` form."""
    events: Dict[str, Dict[str, Any]] = {}
    for raw_line in logs.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record: Optional[Dict[str, Any]] = None
        try:
            candidate = json.loads(line)
            if isinstance(candidate, dict) and isinstance(candidate.get("event"), str):
                record = candidate
        except json.JSONDecodeError:
            pass
        if record is None:
            match = _LEGACY_LINE.match(line)
            if match:
                record = {
                    "event": match.group("event"),
                    "timestamp": float(match.group("timestamp")),
                    "legacy": True,
                }
        if record is None:
            continue
        event = record["event"]
        try:
            record["timestamp"] = _to_epoch(record.get("timestamp"))
        except (TypeError, ValueError):
            continue
        # Last occurrence is authoritative (e.g. restarted-container logs).
        events[event] = record
    return events


def _extract_features(pod: Any) -> Tuple[Optional[JobFeatures], Optional[str]]:
    annotations = pod.metadata.annotations or {}
    values: Dict[str, float] = {}
    for feature, key in FEATURE_ANNOTATIONS.items():
        raw = annotations.get(key)
        if raw is None:
            return None, f"missing annotation {key}"
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"annotation {key} is not numeric: {raw!r}"
        if not math.isfinite(value) or value <= 0:
            return None, f"annotation {key} must be finite and > 0: {raw!r}"
        values[feature] = value
    return JobFeatures(job_id=pod.metadata.name, **values), None


def _container_error(pod: Any) -> Optional[str]:
    statuses = getattr(pod.status, "container_statuses", None) or []
    for status in statuses:
        state = getattr(status, "state", None)
        terminated = getattr(state, "terminated", None) if state else None
        waiting = getattr(state, "waiting", None) if state else None
        if terminated and terminated.exit_code != 0:
            return f"container {status.name} exited {terminated.exit_code}: {terminated.reason or ''} {terminated.message or ''}".strip()
        if waiting and waiting.reason in {"ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError"}:
            return f"container {status.name} waiting: {waiting.reason}: {waiting.message or ''}".strip()
    return None


def _pod_context(pod: Any) -> Dict[str, Any]:
    labels = pod.metadata.labels or {}
    return {
        "run_id": labels.get(LABELS["run_id"]),
        "scenario": labels.get(LABELS["scenario"]),
        "config": labels.get(LABELS["config"]),
        "repetition": labels.get(LABELS["repetition"]),
    }


def _pod_contract_errors(pod: Any, expected: Mapping[str, Any]) -> List[str]:
    """Validate the observed Pod against the pre-registered run contract."""
    errors: List[str] = []
    metadata = getattr(pod, "metadata", None)
    spec = getattr(pod, "spec", None)
    annotations = getattr(metadata, "annotations", None) or {}
    name = getattr(metadata, "name", "")

    scheduler_name = getattr(spec, "scheduler_name", None)
    if scheduler_name != expected.get("scheduler_name"):
        errors.append(
            f"schedulerName={scheduler_name!r} expected={expected.get('scheduler_name')!r}"
        )

    string_contract = {
        RUN_ANNOTATIONS["run_id"]: expected.get("run_id"),
        RUN_ANNOTATIONS["pacing_mode"]: expected.get("pacing_mode"),
        RUN_ANNOTATIONS["reverse"]: str(bool(expected.get("reverse"))).lower(),
        RUN_ANNOTATIONS["expected_jobs"]: str(expected.get("expected_jobs")),
        RUN_ANNOTATIONS["seed"]: str(expected.get("seed")),
        RUN_ANNOTATIONS["load_profile"]: expected.get("load_profile"),
        RUN_ANNOTATIONS["work_model_version"]: expected.get(
            "work_model_version", WORK_MODEL_VERSION
        ),
    }
    for annotation, wanted in string_contract.items():
        if wanted is not None and annotations.get(annotation) != wanted:
            errors.append(
                f"annotation {annotation}={annotations.get(annotation)!r} expected={wanted!r}"
            )
    try:
        observed_delay = float(annotations.get(RUN_ANNOTATIONS["fixed_delay_seconds"], "nan"))
        expected_delay = float(expected.get("fixed_delay_seconds"))
        if not math.isfinite(observed_delay) or not math.isclose(
            observed_delay, expected_delay, rel_tol=0.0, abs_tol=1e-9
        ):
            errors.append(
                f"fixed delay={observed_delay!r} expected={expected_delay!r}"
            )
    except (TypeError, ValueError):
        errors.append("fixed-delay annotation is missing or invalid")

    gates = [
        getattr(gate, "name", None)
        for gate in (getattr(spec, "scheduling_gates", None) or [])
    ]
    expected_gate = expected.get("scheduling_gate")
    wanted_gates = [] if expected_gate is None else [expected_gate]
    if gates != wanted_gates:
        errors.append(f"scheduling gates={gates!r} expected={wanted_gates!r}")

    pull_secrets = [
        getattr(secret, "name", None)
        for secret in (getattr(spec, "image_pull_secrets", None) or [])
    ]
    wanted_pull_secrets = list(expected.get("image_pull_secrets") or [])
    if pull_secrets != wanted_pull_secrets:
        errors.append(
            f"imagePullSecrets={pull_secrets!r} expected={wanted_pull_secrets!r}"
        )

    containers = {
        getattr(container, "name", ""): container
        for container in (getattr(spec, "containers", None) or [])
    }
    train = containers.get("train")
    if train is None:
        errors.append("train container is missing")
        return errors
    if getattr(train, "image", None) != expected.get("image"):
        errors.append(
            f"train image={getattr(train, 'image', None)!r} expected={expected.get('image')!r}"
        )
    container_statuses = {
        getattr(status, "name", ""): status
        for status in (
            getattr(getattr(pod, "status", None), "container_statuses", None) or []
        )
    }
    train_status = container_statuses.get("train")
    image_id = getattr(train_status, "image_id", None) if train_status else None
    if not isinstance(image_id, str) or not re.search(r"sha256:[a-f0-9]{64}", image_id):
        errors.append(f"train container imageID is not digest-qualified: {image_id!r}")
    env = {getattr(item, "name", ""): item for item in (getattr(train, "env", None) or [])}
    job_id_env = env.get("JOB_ID")
    field_ref = getattr(getattr(job_id_env, "value_from", None), "field_ref", None)
    if getattr(field_ref, "field_path", None) != "metadata.name":
        errors.append("JOB_ID must use fieldRef metadata.name")
    job_seed = env.get("JOB_SEED")
    expected_job_seed = str(deterministic_job_seed(int(expected["seed"]), name))
    if getattr(job_seed, "value", None) != expected_job_seed:
        errors.append(
            f"JOB_SEED={getattr(job_seed, 'value', None)!r} expected={expected_job_seed!r}"
        )
    model_version = env.get(WORK_MODEL_ENV)
    if getattr(model_version, "value", None) != WORK_MODEL_VERSION:
        errors.append(
            f"{WORK_MODEL_ENV}={getattr(model_version, 'value', None)!r} "
            f"expected={WORK_MODEL_VERSION!r}"
        )
    blas_threads = env.get(BLAS_THREADS_ENV)
    if getattr(blas_threads, "value", None) != str(REPRODUCTION_BLAS_THREADS):
        errors.append(
            f"{BLAS_THREADS_ENV}={getattr(blas_threads, 'value', None)!r} "
            f"expected={REPRODUCTION_BLAS_THREADS!r}"
        )
    if annotations.get(BLAS_THREADS_ANNOTATION) != str(REPRODUCTION_BLAS_THREADS):
        errors.append(
            f"annotation {BLAS_THREADS_ANNOTATION}="
            f"{annotations.get(BLAS_THREADS_ANNOTATION)!r} "
            f"expected={REPRODUCTION_BLAS_THREADS!r}"
        )
    for feature, annotation in FEATURE_ANNOTATIONS.items():
        item = env.get(f"JOB_{feature}")
        if getattr(item, "value", None) != annotations.get(annotation):
            errors.append(
                f"JOB_{feature}={getattr(item, 'value', None)!r} does not match {annotation}"
            )
    return errors


def _trainer_evidence_errors(
    pod_name: str,
    features: Optional[JobFeatures],
    annotations: Mapping[str, Any],
    events: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """Fail closed when a successful marker cannot prove the shared work model."""

    errors: List[str] = []
    annotation_version = annotations.get(WORK_MODEL_ANNOTATION)
    if annotation_version != WORK_MODEL_VERSION:
        errors.append(
            f"annotation {WORK_MODEL_ANNOTATION}={annotation_version!r} "
            f"expected={WORK_MODEL_VERSION!r}"
        )
    if annotations.get(BLAS_THREADS_ANNOTATION) != str(REPRODUCTION_BLAS_THREADS):
        errors.append(
            f"annotation {BLAS_THREADS_ANNOTATION}="
            f"{annotations.get(BLAS_THREADS_ANNOTATION)!r} "
            f"expected={REPRODUCTION_BLAS_THREADS!r}"
        )
    if features is None:
        return errors

    initialization = events.get("INITIALIZATION_COMPLETED")
    completed = events.get("EXECUTION_COMPLETED")
    if initialization is None:
        errors.append("INITIALIZATION_COMPLETED marker missing")
    else:
        initialized_model = initialization.get("work_model")
        if not isinstance(initialized_model, Mapping):
            errors.append("INITIALIZATION_COMPLETED work_model evidence missing")
        elif initialized_model.get("model_version") != WORK_MODEL_VERSION:
            errors.append("INITIALIZATION_COMPLETED work model version mismatch")
        blas_runtime = initialization.get("blas_runtime")
        blas_libraries = (
            blas_runtime.get("libraries")
            if isinstance(blas_runtime, Mapping)
            else None
        )
        if (
            not isinstance(blas_runtime, Mapping)
            or blas_runtime.get("expected_threads") != REPRODUCTION_BLAS_THREADS
            or not isinstance(blas_libraries, list)
            or not blas_libraries
            or any(
                not isinstance(pool, Mapping)
                or pool.get("num_threads") != REPRODUCTION_BLAS_THREADS
                for pool in (blas_libraries or [])
            )
        ):
            errors.append("INITIALIZATION_COMPLETED BLAS thread evidence is invalid")
    if completed is None:
        return errors

    expected = estimate_work(
        R=features.R,
        M=int(features.M),
        G=features.G,
        C=int(features.C),
        P=int(features.P),
    )
    exact_contract = {
        "job_id": pod_name,
        "work_model_version": WORK_MODEL_VERSION,
        "steps": expected.planned_steps,
        "step_budget": expected.step_budget,
        "convergence_steps": expected.convergence_steps,
        "termination_reason": expected.termination_reason,
        "gradient_bytes": expected.gradient_bytes,
        "checkpoint_bytes": expected.checkpoint_bytes,
        "checkpoint_count": expected.checkpoint_count,
        "blas_threads": REPRODUCTION_BLAS_THREADS,
    }
    for field, wanted in exact_contract.items():
        if completed.get(field) != wanted:
            errors.append(
                f"EXECUTION_COMPLETED {field}={completed.get(field)!r} expected={wanted!r}"
            )
    library_count = completed.get("blas_library_count")
    if (
        isinstance(library_count, bool)
        or not isinstance(library_count, int)
        or library_count <= 0
    ):
        errors.append("EXECUTION_COMPLETED blas_library_count must be > 0")
    for field in ("final_loss", "checkpoint_seconds", "duration_seconds"):
        value = completed.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            errors.append(f"EXECUTION_COMPLETED {field} must be finite and >= 0")
    return errors


def _load_kubernetes_configuration(mode: str, context: Optional[str] = None) -> None:
    if config is None:
        raise RuntimeError("the kubernetes Python package is not installed")
    if mode == "incluster":
        if context:
            raise ValueError("a kubeconfig context cannot be used with incluster mode")
        config.load_incluster_config()
    elif mode == "kubeconfig":
        config.load_kube_config(context=context)
    elif mode == "auto":
        if context:
            config.load_kube_config(context=context)
        else:
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
    else:
        raise ValueError(f"unknown Kubernetes configuration mode {mode!r}")


def _pod_image_inventory(pods: Iterable[Any]) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    for pod in pods:
        requested = {
            getattr(container, "name", ""): getattr(container, "image", None)
            for container in (getattr(getattr(pod, "spec", None), "containers", None) or [])
        }
        statuses = {
            getattr(status, "name", ""): status
            for status in (
                getattr(getattr(pod, "status", None), "container_statuses", None) or []
            )
        }
        containers = []
        for name in sorted(set(requested) | set(statuses)):
            status = statuses.get(name)
            containers.append({
                "name": name,
                "requested_image": requested.get(name),
                "reported_image": getattr(status, "image", None) if status else None,
                "image_id": getattr(status, "image_id", None) if status else None,
                "restart_count": getattr(status, "restart_count", None) if status else None,
            })
        metadata = getattr(pod, "metadata", None)
        inventory.append({
            "name": getattr(metadata, "name", None),
            "uid": getattr(metadata, "uid", None),
            "node_name": getattr(getattr(pod, "spec", None), "node_name", None),
            "containers": containers,
        })
    return inventory


def _cluster_environment(core_api: Any, pods: Iterable[Any]) -> Dict[str, Any]:
    pod_list = list(pods)
    environment: Dict[str, Any] = {
        "kubernetes": {"workload_pods": _pod_image_inventory(pod_list)}
    }
    try:
        nodes = core_api.list_node().items
        environment["kubernetes"]["nodes"] = [
            {
                "name": node.metadata.name,
                "kubelet_version": getattr(
                    getattr(node.status, "node_info", None), "kubelet_version", None
                ),
                "container_runtime_version": getattr(
                    getattr(node.status, "node_info", None),
                    "container_runtime_version",
                    None,
                ),
                "allocatable": dict(node.status.allocatable or {}),
                "capacity": dict(node.status.capacity or {}),
            }
            for node in nodes
        ]
    except Exception as exc:
        environment["kubernetes"]["metadata_error"] = str(exc)
    return environment


def collect(
    namespace: str,
    label_selector: str = "app=ml-sim-job",
    *,
    expected_count: Optional[int] = None,
    run_id: str = "manual-run",
    scenario: str = "manual",
    config_name: str = "manual",
    repetition: int = 0,
    seed: int = 0,
    strict: bool = True,
    kube_config_mode: str = "auto",
    kube_context: Optional[str] = None,
    core_api: Any = None,
    environment: Optional[Mapping[str, Any]] = None,
    expected_pod_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect one run and return the shared schema document.

    ``expected_count`` is mandatory in strict mode.  A caller may inject a
    CoreV1Api-compatible object for deterministic integration tests.
    """
    if expected_count is not None and expected_count <= 0:
        raise ValueError("expected_count must be > 0")
    if repetition < 0:
        raise ValueError("repetition must be >= 0")
    if core_api is None:
        _load_kubernetes_configuration(kube_config_mode, context=kube_context)
        core_api = client.CoreV1Api()

    pods = core_api.list_namespaced_pod(namespace, label_selector=label_selector).items
    pods = sorted(pods, key=lambda pod: pod.metadata.name)
    failures: List[Dict[str, Any]] = []
    if expected_count is None:
        failures.append({
            "code": "EXPECTED_COUNT_REQUIRED",
            "message": "strict collection requires --expected-jobs",
        })
        expected_for_document = len(pods) or 1
    else:
        expected_for_document = expected_count
        if len(pods) != expected_count:
            failures.append({
                "code": "POD_COUNT_MISMATCH",
                "message": f"expected {expected_count} pods, observed {len(pods)}",
            })

    extracted: Dict[str, JobFeatures] = {}
    feature_errors: Dict[str, str] = {}
    for pod in pods:
        features, error = _extract_features(pod)
        if error:
            feature_errors[pod.metadata.name] = error
        else:
            extracted[pod.metadata.name] = features  # type: ignore[assignment]
    ranks = compute_ranks(list(extracted.values())) if extracted else {}

    rows: List[Dict[str, Any]] = []
    for pod in pods:
        name = pod.metadata.name
        context = _pod_context(pod)
        context_mismatches = [
            f"{field} label={context[field]!r} expected={expected!r}"
            for field, expected in (
                ("run_id", run_id),
                ("scenario", scenario),
                ("config", config_name),
                ("repetition", str(repetition)),
            )
            if context[field] != expected
        ]
        feature = extracted.get(name)
        feature_values = {
            key: float(getattr(feature, key)) if feature else 0.0
            for key in FEATURE_ANNOTATIONS
        }
        row: Dict[str, Any] = {
            "job_id": name,
            "category": name.rsplit("-", 2)[0].replace("-", "_"),
            "features": feature_values,
            "rank": float(ranks.get(name, 0.0)),
            "submission_time": None,
            "execution_start_time": None,
            "completion_time": None,
            "jct_s": None,
            "status": "failed",
            "node_name": getattr(pod.spec, "node_name", None),
            "error": None,
            "trainer_evidence": None,
        }
        errors: List[str] = []
        if name in feature_errors:
            errors.append(feature_errors[name])
        errors.extend(context_mismatches)
        if expected_pod_contract is not None:
            errors.extend(_pod_contract_errors(pod, expected_pod_contract))
        try:
            row["submission_time"] = _to_epoch(pod.metadata.creation_timestamp)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        pod_error = _container_error(pod)
        if pod_error:
            errors.append(pod_error)

        logs = ""
        try:
            logs = core_api.read_namespaced_pod_log(name, namespace, container="train")
        except Exception as exc:
            errors.append(f"unable to read trainer logs: {exc}")
        events = parse_log_events(logs)
        failed_event = events.get("EXECUTION_FAILED")
        if failed_event:
            errors.append(f"trainer reported EXECUTION_FAILED: {failed_event.get('error', 'unknown error')}")
        started = events.get("EXECUTION_STARTED")
        completed = events.get("EXECUTION_COMPLETED")
        errors.extend(
            _trainer_evidence_errors(
                name,
                feature,
                getattr(pod.metadata, "annotations", None) or {},
                events,
            )
        )
        if not started:
            errors.append("EXECUTION_STARTED marker missing")
        if not completed:
            errors.append("EXECUTION_COMPLETED marker missing")
        if getattr(pod.status, "phase", None) == "Failed":
            errors.append(f"Pod phase is Failed: {getattr(pod.status, 'reason', '')}")

        if not errors:
            row["execution_start_time"] = started["timestamp"]
            row["completion_time"] = completed["timestamp"]
            if not row["submission_time"] <= row["execution_start_time"] <= row["completion_time"]:
                errors.append("timestamps violate submission <= start <= completion")
            else:
                row["jct_s"] = row["completion_time"] - row["submission_time"]
                row["status"] = "completed"
                row["trainer_evidence"] = dict(completed)
        if errors:
            row["error"] = "; ".join(dict.fromkeys(errors))
            failures.append({"code": "JOB_FAILED", "job_id": name, "message": row["error"]})
        rows.append(row)

    env = _cluster_environment(core_api, pods)
    env["kubernetes"]["namespace"] = namespace
    for key, value in (environment or {}).items():
        if isinstance(value, Mapping) and isinstance(env.get(key), Mapping):
            env[key] = {**dict(env[key]), **dict(value)}
        else:
            env[key] = value
    kubernetes_environment = env.get("kubernetes")
    metadata_error = (
        kubernetes_environment.get("metadata_error")
        if isinstance(kubernetes_environment, Mapping)
        else "environment.kubernetes is not an object"
    )
    if metadata_error:
        failures.append({
            "code": "CLUSTER_METADATA_UNAVAILABLE",
            "message": f"unable to capture node metadata: {metadata_error}",
        })
    document = make_result_document(
        run_id=run_id,
        scenario=scenario,
        config=config_name,
        repetition=repetition,
        seed=seed,
        expected_jobs=expected_for_document,
        jobs=rows,
        environment=env,
        failures=failures,
        source="kubernetes",
    )
    if strict:
        errors = validate_result_document(document, strict=False)
        if failures or errors or expected_count is None or document["run"]["status"] != "completed":
            reason = "; ".join(failure["message"] for failure in failures) or "; ".join(errors)
            raise IncompleteRunError(reason or "run is incomplete", document)
        validate_result_document(document, strict=True)
    return document


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compatibility wrapper for callers using the original flat row shape."""
    normalized = []
    for row in rows:
        submission = row.get("submission_time", row.get("submission_t"))
        start = row.get("execution_start_time", row.get("exec_start"))
        end = row.get("completion_time", row.get("exec_end"))
        normalized.append({
            **row,
            "submission_time": submission,
            "execution_start_time": start,
            "completion_time": end,
            "jct_s": row.get("jct_s", row.get("jct", end - submission)),
            "status": row.get("status", "completed"),
        })
    result = summarize_jobs(normalized)
    result["n_jobs"] = len(normalized)
    result["raw_rows"] = rows
    return result


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--label-selector", default="app=ml-sim-job")
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--config", dest="config_name", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kube-config", choices=("auto", "incluster", "kubeconfig"), default="auto")
    parser.add_argument("--context", help="explicit kubeconfig context for collector safety")
    parser.add_argument("--out", default="../results/metrics.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        document = collect(
            args.namespace,
            args.label_selector,
            expected_count=args.expected_jobs,
            run_id=args.run_id,
            scenario=args.scenario,
            config_name=args.config_name,
            repetition=args.repetition,
            seed=args.seed,
            strict=True,
            kube_config_mode=args.kube_config,
            kube_context=args.context,
        )
        exit_code = 0
    except IncompleteRunError as exc:
        document = exc.document
        exit_code = 2
    _write_json(Path(args.out), document)
    print(json.dumps({"run": document["run"], "summary": document["summary"], "failures": document["failures"]}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
