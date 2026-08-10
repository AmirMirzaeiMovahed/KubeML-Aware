#!/usr/bin/env python3
"""Run a bounded paired ML-training pilot on a shared single-node cluster.

This is production-extension evidence, not the clean article reproduction.
It compares identical deterministic NumPy training bursts in an isolated
baseline namespace and through the production scheduling-gate controller.
CPU requests create real kube-scheduler contention, while health checks abort
and clean up the exact pilot Pods if colocated VPN Pods stop being Ready.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k8s.work_model import (  # noqa: E402
    BLAS_THREADS_ANNOTATION,
    BLAS_THREADS_ENV,
    REPRODUCTION_BLAS_THREADS,
    WORK_MODEL_ANNOTATION,
    WORK_MODEL_ENV,
    WORK_MODEL_VERSION,
)
from scheduler.constants import (  # noqa: E402
    ANNOTATION_MAP,
    EXECUTION_CONTAINER_ANNOTATION,
    EXPECTED_JOBS_ANNOTATION,
    FAST_PATH_ANNOTATION,
    PACING_MODE_ANNOTATION,
    RANK_POLICY_ANNOTATION,
    RELEASE_GATE,
    REVERSE_ANNOTATION,
    RUN_ID_ANNOTATION,
    RUN_ID_LABEL,
    TAIL_BALANCE_ANNOTATION,
    WORKLOAD_KIND_ANNOTATION,
)
from scheduler.rank import compute_ranks  # noqa: E402
from workload.generate_workload import (  # noqa: E402
    deterministic_job_seed,
    generate_burst,
    generate_category_burst,
)

CUSTOM_NAMESPACE = "ai-scheduler"
BASELINE_NAMESPACE = "kubeml-baseline"
SCHEDULER_DEPLOYMENT = "kubeml-scheduler-ml-ai-scheduler"
PILOT_LABEL = "pilot.kubeml/run-id"
ARM_LABEL = "pilot.kubeml/arm"
JOB_LABEL = "pilot.kubeml/job-index"

ARM_SETTINGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "code": "b",
        "rank_policy": "six_feature",
        "tail_balance": False,
        "fast_path": False,
        "reverse": False,
    },
    "kubeml": {
        "code": "k",
        "rank_policy": "six_feature",
        "tail_balance": True,
        "fast_path": True,
        "reverse": False,
    },
    "duration-only": {
        "code": "d",
        "rank_policy": "duration_only",
        "tail_balance": False,
        "fast_path": True,
        "reverse": False,
    },
    "six-feature-no-tail": {
        "code": "n",
        "rank_policy": "six_feature",
        "tail_balance": False,
        "fast_path": True,
        "reverse": False,
    },
    "six-feature-no-fastpath": {
        "code": "f",
        "rank_policy": "six_feature",
        "tail_balance": False,
        "fast_path": False,
        "reverse": False,
    },
    "reversed": {
        "code": "r",
        "rank_policy": "six_feature",
        "tail_balance": False,
        "fast_path": False,
        "reverse": True,
    },
}


def kubectl(
    *args: str,
    input_document: object | None = None,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        ["kubectl", *args],
        input=(json.dumps(input_document) if input_document is not None else None),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"kubectl failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def utc_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ensure_baseline_namespace() -> None:
    manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": BASELINE_NAMESPACE,
            "labels": {"app.kubernetes.io/part-of": "kubeml-training-pilot"},
        },
    }
    kubectl("apply", "-f", "-", input_document=manifest)


def service_snapshot() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for namespace in ("parapolu", "vpn"):
        payload = json.loads(kubectl("-n", namespace, "get", "pods", "-o", "json"))
        rows = []
        for pod in payload.get("items", []):
            statuses = pod.get("status", {}).get("containerStatuses", [])
            rows.append(
                {
                    "name": pod["metadata"]["name"],
                    "phase": pod.get("status", {}).get("phase"),
                    "ready": bool(statuses) and all(item.get("ready") for item in statuses),
                    "restarts": sum(item.get("restartCount", 0) for item in statuses),
                }
            )
        result[namespace] = rows
    return result


def assert_services_ready(snapshot: dict[str, list[dict[str, Any]]]) -> None:
    for namespace, pods in snapshot.items():
        if not pods:
            raise RuntimeError(f"protected namespace {namespace!r} has no Pods")
        unhealthy = [row["name"] for row in pods if row["phase"] != "Running" or not row["ready"]]
        if unhealthy:
            raise RuntimeError(f"protected namespace {namespace!r} has unhealthy Pods: {unhealthy}")


def node_cpu_percent() -> float:
    lines = kubectl("top", "node", "--no-headers").strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected one node metrics row, got {lines!r}")
    fields = lines[0].split()
    return float(fields[2].removesuffix("%"))


def wait_for_cool_node(max_percent: float, timeout: float = 180.0) -> float:
    deadline = time.monotonic() + timeout
    consecutive = 0
    while time.monotonic() < deadline:
        usage = node_cpu_percent()
        snapshot = service_snapshot()
        assert_services_ready(snapshot)
        if usage <= max_percent:
            consecutive += 1
            if consecutive >= 2:
                return usage
        else:
            consecutive = 0
        time.sleep(5)
    raise TimeoutError(f"node CPU did not cool below {max_percent:.1f}%")


def pod_manifest(
    *,
    arm: str,
    repetition: int,
    run_id: str,
    job: Any,
    index: int,
    image: str,
    cpu: str,
    expected_jobs: int,
) -> dict[str, Any]:
    arm_settings = ARM_SETTINGS[arm]
    namespace = CUSTOM_NAMESPACE if arm == "kubeml" else BASELINE_NAMESPACE
    name = f"tfp-{arm[0]}-r{repetition}-{index:02d}"
    labels = {
        PILOT_LABEL: run_id,
        ARM_LABEL: arm,
        JOB_LABEL: str(index),
        "app.kubernetes.io/name": "kubeml-training-pilot",
        RUN_ID_LABEL: run_id,
    }
    annotations = {
        WORKLOAD_KIND_ANNOTATION: "training",
        RUN_ID_ANNOTATION: run_id,
        EXPECTED_JOBS_ANNOTATION: str(expected_jobs),
        PACING_MODE_ANNOTATION: "none",
        REVERSE_ANNOTATION: str(arm_settings["reverse"]).lower(),
        RANK_POLICY_ANNOTATION: str(arm_settings["rank_policy"]),
        TAIL_BALANCE_ANNOTATION: str(arm_settings["tail_balance"]).lower(),
        FAST_PATH_ANNOTATION: str(arm_settings["fast_path"]).lower(),
        EXECUTION_CONTAINER_ANNOTATION: "train",
        WORK_MODEL_ANNOTATION: WORK_MODEL_VERSION,
        BLAS_THREADS_ANNOTATION: str(REPRODUCTION_BLAS_THREADS),
        "ml.scheduler/source-job-id": job.job_id,
        "ml.scheduler/seed": str(deterministic_job_seed(8100 + repetition, job.job_id)),
        **{ANNOTATION_MAP[key]: str(getattr(job, key)) for key in ANNOTATION_MAP},
    }
    spec: dict[str, Any] = {
        "schedulerName": "default-scheduler",
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 10,
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "volumes": [{"name": "tmp", "emptyDir": {"sizeLimit": "64Mi"}}],
        "containers": [
            {
                "name": "train",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
                "resources": {
                    "requests": {"cpu": cpu, "memory": "128Mi"},
                    "limits": {"cpu": cpu, "memory": "1Gi"},
                },
                "env": [
                    {"name": "JOB_ID", "value": name},
                    {
                        "name": "JOB_SEED",
                        "value": str(deterministic_job_seed(8100 + repetition, job.job_id)),
                    },
                    {"name": WORK_MODEL_ENV, "value": WORK_MODEL_VERSION},
                    {"name": BLAS_THREADS_ENV, "value": str(REPRODUCTION_BLAS_THREADS)},
                    *[
                        {"name": f"JOB_{key}", "value": str(getattr(job, key))}
                        for key in ANNOTATION_MAP
                    ],
                ],
            }
        ],
    }
    # Both arms use the same launch barrier so sequential Kubernetes API
    # creation time cannot give the unranked baseline an early-start bonus.
    # The baseline harness removes only this barrier in manifest/FIFO order;
    # default-scheduler still performs all feasibility, scoring and binding.
    spec["schedulingGates"] = [{"name": RELEASE_GATE}]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": spec,
    }


def wait_for_completion(
    namespace: str,
    run_id: str,
    expected: int,
    *,
    timeout: float,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_sample = 0.0
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = json.loads(
            kubectl("-n", namespace, "get", "pods", "-l", f"{PILOT_LABEL}={run_id}", "-o", "json")
        )
        last_payload = payload
        items = payload.get("items", [])
        failed = [
            pod["metadata"]["name"]
            for pod in items
            if pod.get("status", {}).get("phase") == "Failed"
        ]
        if failed:
            raise RuntimeError(f"training Pods failed: {failed}")
        if len(items) == expected and all(
            pod.get("status", {}).get("phase") == "Succeeded" for pod in items
        ):
            return payload
        if time.monotonic() - last_sample >= 2.0:
            services = service_snapshot()
            assert_services_ready(services)
            samples.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "node_cpu_percent": node_cpu_percent(),
                    "protected_services": services,
                }
            )
            last_sample = time.monotonic()
        time.sleep(0.5)
    phases = {
        pod["metadata"]["name"]: pod.get("status", {}).get("phase")
        for pod in last_payload.get("items", [])
    }
    raise TimeoutError(f"run {run_id!r} timed out: {phases}")


def release_baseline_barrier(
    namespace: str,
    run_id: str,
    pod_names: list[str],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Synchronize a default-scheduler burst, then release it in FIFO order."""

    deadline = time.monotonic() + timeout
    observed: set[str] = set()
    while time.monotonic() < deadline:
        payload = json.loads(
            kubectl(
                "-n",
                namespace,
                "get",
                "pods",
                "-l",
                f"{PILOT_LABEL}={run_id}",
                "-o",
                "json",
            )
        )
        observed = {item["metadata"]["name"] for item in payload.get("items", [])}
        if observed == set(pod_names):
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(
            f"baseline launch barrier expected {pod_names!r}, observed {sorted(observed)!r}"
        )
    time.sleep(0.25)
    started = time.time()
    for name in pod_names:
        kubectl(
            "-n",
            namespace,
            "patch",
            "pod",
            name,
            "--type=json",
            "-p",
            json.dumps([{"op": "remove", "path": "/spec/schedulingGates/0"}]),
        )
    completed = time.time()
    return {
        "mode": "synchronized-fifo-launch-barrier",
        "scheduler": "default-scheduler",
        "release_order": pod_names,
        "release_started_at": started,
        "release_completed_at": completed,
        "release_duration_seconds": completed - started,
    }


def collect_run(
    arm: str,
    repetition: int,
    run_id: str,
    namespace: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for pod in payload["items"]:
        metadata = pod["metadata"]
        logs = kubectl("-n", namespace, "logs", metadata["name"], "-c", "train")
        events = {}
        for line in logs.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and isinstance(event.get("event"), str):
                events[event["event"]] = event
        if "EXECUTION_STARTED" not in events or "EXECUTION_COMPLETED" not in events:
            raise RuntimeError(f"missing execution markers for {metadata['name']}")
        creation = utc_timestamp(metadata["creationTimestamp"])
        execution = float(events["EXECUTION_STARTED"]["timestamp"])
        completion = float(events["EXECUTION_COMPLETED"]["timestamp"])
        rows.append(
            {
                "job_id": metadata["name"],
                "source_job_id": metadata["annotations"]["ml.scheduler/source-job-id"],
                "job_index": int(metadata["labels"][JOB_LABEL]),
                "creation_time": creation,
                "execution_start_time": execution,
                "completion_time": completion,
                "jct_seconds": completion - creation,
                "execution_seconds": completion - execution,
            }
        )
    rows.sort(key=lambda row: row["execution_start_time"])
    jcts = [row["jct_seconds"] for row in rows]
    starts = [row["execution_start_time"] for row in rows]
    metrics = {
        "avg_jct_seconds": sum(jcts) / len(jcts),
        "p95_jct_seconds": percentile(jcts, 0.95),
        "min_jct_seconds": min(jcts),
        "max_jct_seconds": max(jcts),
        "makespan_seconds": max(row["completion_time"] for row in rows)
        - min(row["creation_time"] for row in rows),
        "avg_ilt_seconds": sum(b - a for a, b in pairwise(starts)) / (len(starts) - 1),
    }
    return {
        "arm": arm,
        "repetition": repetition,
        "run_id": run_id,
        "namespace": namespace,
        "metrics": metrics,
        "jobs": rows,
    }


def cleanup(namespace: str, run_id: str) -> None:
    kubectl(
        "-n",
        namespace,
        "delete",
        "pod",
        "-l",
        f"{PILOT_LABEL}={run_id}",
        "--wait=true",
        "--timeout=60s",
        check=False,
    )


def execute_arm(
    *,
    arm: str,
    repetition: int,
    image: str,
    cpu: str,
    output: Path,
    timeout: float,
    cooldown_cpu_percent: float,
    jobs_per_run: int,
    workload_profile: str,
) -> dict[str, Any]:
    namespace = CUSTOM_NAMESPACE if arm == "kubeml" else BASELINE_NAMESPACE
    run_id = f"tfp-{ARM_SETTINGS[arm]['code']}-r{repetition}-{int(time.time())}"
    if workload_profile == "heavy":
        jobs = generate_category_burst(
            jobs_per_run,
            category="heavy",
            seed=8100 + repetition,
        )
    else:
        jobs = generate_burst(jobs_per_run, seed=8100 + repetition, load="normal")
    ranks = compute_ranks(jobs)
    manifests = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            pod_manifest(
                arm=arm,
                repetition=repetition,
                run_id=run_id,
                job=job,
                index=index,
                image=image,
                cpu=cpu,
                expected_jobs=len(jobs),
            )
            for index, job in enumerate(jobs)
        ],
    }
    (output / f"{run_id}-manifest.json").write_text(
        json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
    )
    starting_cpu = wait_for_cool_node(cooldown_cpu_percent)
    samples: list[dict[str, Any]] = []
    print(
        f"START arm={arm} repetition={repetition} run_id={run_id} cpu={starting_cpu:.1f}%",
        flush=True,
    )
    kubectl("apply", "-f", "-", input_document=manifests)
    try:
        admission = None
        if arm == "baseline":
            admission = release_baseline_barrier(
                namespace,
                run_id,
                [item["metadata"]["name"] for item in manifests["items"]],
            )
        pods = wait_for_completion(
            namespace,
            run_id,
            len(jobs),
            timeout=timeout,
            samples=samples,
        )
        result = collect_run(arm, repetition, run_id, namespace, pods)
        if admission is not None:
            result["admission"] = admission
        result["node_samples"] = samples
        result["features"] = [
            {
                "source_job_id": job.job_id,
                "rank": ranks[job.job_id],
                **{key: getattr(job, key) for key in ANNOTATION_MAP},
            }
            for job in jobs
        ]
        if arm != "baseline":
            gate_text = kubectl(
                "-n",
                CUSTOM_NAMESPACE,
                "exec",
                f"deploy/{SCHEDULER_DEPLOYMENT}",
                "--",
                "cat",
                f"/results/gate-{run_id}.json",
            )
            gate = json.loads(gate_text)
            gate_path = output / f"gate-{run_id}.json"
            gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
            result["gate_record"] = gate_path.name
        print(
            f"DONE arm={arm} repetition={repetition} "
            f"avg_jct={result['metrics']['avg_jct_seconds']:.3f}s "
            f"makespan={result['metrics']['makespan_seconds']:.3f}s",
            flush=True,
        )
        return result
    finally:
        cleanup(namespace, run_id)


def aggregate(runs: list[dict[str, Any]], repetitions: int, arms: list[str]) -> dict[str, Any]:
    metric_names = sorted(runs[0]["metrics"])
    means = {}
    for arm in arms:
        subset = [run for run in runs if run["arm"] == arm]
        means[arm] = {
            metric: sum(run["metrics"][metric] for run in subset) / len(subset)
            for metric in metric_names
        }
    paired: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        baseline = next(
            run for run in runs if run["arm"] == "baseline" and run["repetition"] == repetition
        )
        row: dict[str, Any] = {"repetition": repetition}
        for arm in arms:
            if arm == "baseline":
                continue
            candidate = next(
                run for run in runs if run["arm"] == arm and run["repetition"] == repetition
            )
            for metric in ("avg_jct_seconds", "p95_jct_seconds", "makespan_seconds"):
                row[f"{arm}.{metric}_improvement_pct"] = (
                    100
                    * (baseline["metrics"][metric] - candidate["metrics"][metric])
                    / baseline["metrics"][metric]
                )
        paired.append(row)
    report: dict[str, Any] = {
        "means": means,
        "paired": paired,
    }
    report["mean_improvement_vs_baseline_pct"] = {
        arm: {
            metric: sum(row[f"{arm}.{metric}_improvement_pct"] for row in paired) / repetitions
            for metric in ("avg_jct_seconds", "p95_jct_seconds", "makespan_seconds")
        }
        for arm in arms
        if arm != "baseline"
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARM_SETTINGS),
        default=["baseline", "kubeml"],
        help=(
            "registered paired arms; use all six arms for the reviewer-hardening ablation matrix"
        ),
    )
    parser.add_argument(
        "--reviewer-matrix",
        action="store_true",
        help="run the predeclared 30-block, six-arm reviewer-hardening matrix",
    )
    parser.add_argument("--image", default="registry.local/kubeml/ml-sim-job:0.3.0")
    parser.add_argument("--cpu", default="1")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--workload-profile", choices=("heavy", "mixed"), default="heavy")
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--cooldown-cpu-percent", type=float, default=35)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/root/kubeml-fastpath-030/benchmarks")
    )
    args = parser.parse_args(argv)
    if args.reviewer_matrix:
        args.repetitions = 30
        args.arms = list(ARM_SETTINGS)
    if args.repetitions <= 0:
        parser.error("--repetitions must be > 0")
    if args.jobs <= 1:
        parser.error("--jobs must be > 1")
    arms = list(dict.fromkeys(args.arms))
    if "baseline" not in arms or len(arms) < 2:
        parser.error("--arms must include baseline and at least one comparison arm")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    ensure_baseline_namespace()
    initial_services = service_snapshot()
    assert_services_ready(initial_services)
    runs: list[dict[str, Any]] = []
    try:
        for repetition in range(args.repetitions):
            order = arms.copy()
            random.Random(20260810 + repetition).shuffle(order)
            for arm in order:
                runs.append(
                    execute_arm(
                        arm=arm,
                        repetition=repetition,
                        image=args.image,
                        cpu=args.cpu,
                        output=output,
                        timeout=args.timeout_seconds,
                        cooldown_cpu_percent=args.cooldown_cpu_percent,
                        jobs_per_run=args.jobs,
                        workload_profile=args.workload_profile,
                    )
                )
                (output / "runs.partial.json").write_text(
                    json.dumps(runs, indent=2) + "\n", encoding="utf-8"
                )
                time.sleep(5)
        final_services = service_snapshot()
        assert_services_ready(final_services)
        report = {
            "schema_version": "1.0",
            "kind": "paired-real-training-fast-path-pilot",
            "eligible_for_article_claim": False,
            "reason": (
                "Shared France K3s node with colocated VPN services; validates the "
                "production extension but is not the clean registered article matrix."
            ),
            "workload": {
                "type": "training-only",
                "implementation": "versioned NumPy matrix/gradient/checkpoint trainer",
                "jobs_per_run": args.jobs,
                "load": args.workload_profile,
                "seed_base": 8100,
                "cpu_request_and_limit": args.cpu,
                "image": args.image,
                "baseline": (
                    "Kubernetes default-scheduler with a synchronized FIFO "
                    "launch barrier; no custom ranking or binding"
                ),
            },
            "arms": {arm: ARM_SETTINGS[arm] for arm in arms},
            "arm_order": "deterministic per-repetition shuffle; seed 20260810 + repetition",
            "repetitions": args.repetitions,
            "initial_protected_services": initial_services,
            "final_protected_services": final_services,
            "runs": runs,
            "aggregate": aggregate(runs, args.repetitions, arms),
        }
        result_path = output / "paired-training-results.json"
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"RESULT={result_path}", flush=True)
        print(json.dumps(report["aggregate"], indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve pilot failure evidence
        (output / "failure.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        print(f"FAILED output={output}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
