"""Materialize and execute the pre-registered 70/90-run cluster matrix.

Without ``--execute`` this command is side-effect-free except for writing the
plan JSON (and, if requested, deterministic manifests via ``--materialize``).
Execution assumes the long-running custom scheduler is already installed.  Per
run pacing/reverse settings are carried by Pod annotations, so no Deployment
patching or restart hook is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "workload"))
sys.path.append(str(ROOT))
from generate_workload import (  # noqa: E402
    LOAD_PROFILES,
    OUTPUT_SENTINEL,
    build_workload_document,
    generate_burst,
    prepare_output_directory,
    save_json,
    to_pod_yaml,
    write_pod_manifests,
)

from experiments.controls import (  # noqa: E402
    DEFAULT_COOLDOWN_CLEAN_POLLS,
    DEFAULT_COOLDOWN_SECONDS,
    execution_controls_contract,
    prewarm_pod_manifest,
    validate_cooldown_evidence,
    validate_execution_controls_evidence,
    validate_minikube_attestation,
    validate_prewarm_observation,
)
from experiments.environment import (  # noqa: E402
    ArticleEnvironmentError,
    validate_article_environment,
)
from experiments.schema import (  # noqa: E402
    IncompleteRunError,
    validate_result_document,
)
from experiments.submission import (  # noqa: E402
    BurstSubmissionError,
    submit_burst,
)
from k8s.work_model import (  # noqa: E402
    REPRODUCTION_BLAS_THREADS,
    WORK_MODEL_VERSION,
)
from results.metrics_collector import collect  # noqa: E402

PLAN_SCHEMA_VERSION = "1.2"
DEFAULT_PLAN = Path(__file__).with_name("scenarios.yaml")
SCHEDULE_SCHEMA_VERSION = 3
DEFAULT_SCHEDULER_DEPLOYMENT = "ml-ai-scheduler-ml-ai-scheduler"
DEFAULT_SCHEDULER_CONTAINER = "scheduler"
DEFAULT_SCHEDULER_RESULTS_TEMPLATE = "/results/schedule-{run_id}.json"
_IMMUTABLE_IMAGE_RE = re.compile(r"^.+@sha256:[a-f0-9]{64}$")


@dataclass(frozen=True)
class RunSpec:
    sequence: int
    group: str
    run_id: str
    scenario: str
    jobs: int
    load: str
    repetition: int
    seed: int
    config: str
    scheduler_name: str
    manifest_set: str
    pacing_mode: str
    fixed_delay_seconds: float
    reverse: bool


class ClusterRunError(RuntimeError):
    pass


def _read_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"plan must be a YAML object: {path}")
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported plan schema_version: {payload.get('schema_version')!r}")
    return payload


def expand_plan(path: Path = DEFAULT_PLAN, *, include_adaptive: bool = False) -> List[RunSpec]:
    document = _read_yaml(path)
    if document.get("execution_controls") != execution_controls_contract():
        raise ValueError("plan execution_controls has drifted from the runner")
    declared_load_profiles = document.get("load_profiles")
    if not isinstance(declared_load_profiles, Mapping):
        raise ValueError("load_profiles is missing")
    for name, implementation in LOAD_PROFILES.items():
        declared = declared_load_profiles.get(name)
        declared_contract = (
            {key: value for key, value in declared.items() if key != "description"}
            if isinstance(declared, Mapping)
            else None
        )
        implementation_contract = {
            key: value for key, value in implementation.items() if key != "description"
        }
        if declared_contract != implementation_contract:
            raise ValueError(f"plan load profile {name!r} has drifted from workload generation")
    profiles = document.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("profiles is missing")
    base = profiles.get("article_exact")
    extension = profiles.get("adaptive_extension", {})
    if not isinstance(base, Mapping) or not isinstance(base.get("groups"), list):
        raise ValueError("profiles.article_exact.groups is missing")

    extension_by_group: Dict[str, List[Mapping[str, Any]]] = {}
    if include_adaptive:
        for item in extension.get("groups", []):
            extension_by_group[item["target_group"]] = list(item["configurations"])

    specs: List[RunSpec] = []
    sequence = 1
    seed_policy = document.get("seed_policy") or {}
    for group in base["groups"]:
        group_name = str(group["name"])
        expected_seed_base = seed_policy.get(f"{group_name}_base")
        if expected_seed_base is None or int(group["seed_base"]) != int(expected_seed_base):
            raise ValueError(f"seed policy and {group_name!r} group seed_base disagree")
        configs = list(group["configurations"]) + extension_by_group.get(group_name, [])
        repetitions = int(group["repetitions"])
        if group_name == "pacing":
            scenarios = [{
                "name": group["scenario"],
                "jobs": group["jobs"],
                "load": group["load"],
                "seed_offset": 0,
            }]
        else:
            scenarios = group["scenarios"]
        for scenario in scenarios:
            for repetition in range(repetitions):
                seed = int(group["seed_base"]) + int(scenario.get("seed_offset", 0)) + repetition
                for config_item in configs:
                    config_name = str(config_item["name"])
                    scheduler_name = str(config_item["scheduler_name"])
                    manifest_set = str(config_item["manifest_set"])
                    expected_manifest_set = (
                        "default" if scheduler_name == "default-scheduler" else "custom"
                    )
                    if manifest_set != expected_manifest_set:
                        raise ValueError(
                            f"configuration {config_name!r} manifest_set={manifest_set!r} "
                            f"does not match scheduler_name={scheduler_name!r}"
                        )
                    run_id = f"k8s-{group_name}-{scenario['name']}-{config_name}-r{repetition}"
                    if len(run_id) > 63:
                        raise ValueError(f"generated run_id exceeds Kubernetes label limit: {run_id}")
                    specs.append(RunSpec(
                        sequence=sequence,
                        group=group_name,
                        run_id=run_id,
                        scenario=str(scenario["name"]),
                        jobs=int(scenario["jobs"]),
                        load=str(scenario["load"]),
                        repetition=repetition,
                        seed=seed,
                        config=config_name,
                        scheduler_name=scheduler_name,
                        manifest_set=manifest_set,
                        pacing_mode=str(config_item["pacing_mode"]),
                        fixed_delay_seconds=float(config_item["fixed_delay_seconds"]),
                        reverse=bool(config_item["reverse"]),
                    ))
                    sequence += 1

    expected = 90 if include_adaptive else 70
    if len(specs) != expected:
        raise ValueError(f"plan expansion invariant failed: expected {expected} runs, got {len(specs)}")
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError("plan contains duplicate run IDs")

    order = document.get("execution_order", {})
    if order.get("strategy") != "deterministic-randomized-blocks":
        raise ValueError("execution_order.strategy must be deterministic-randomized-blocks")
    rng = random.Random(int(order["seed"]))
    block_map: Dict[tuple[str, str, int], List[RunSpec]] = {}
    for spec in specs:
        block_map.setdefault((spec.group, spec.scenario, spec.repetition), []).append(spec)
    blocks = list(block_map.values())
    rng.shuffle(blocks)
    ordered_specs: List[RunSpec] = []
    for block in blocks:
        rng.shuffle(block)
        ordered_specs.extend(block)
    return [replace(spec, sequence=index) for index, spec in enumerate(ordered_specs, start=1)]


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_document(
    specs: Sequence[RunSpec], *, include_adaptive: bool, source_plan: Path = DEFAULT_PLAN
) -> Dict[str, Any]:
    document = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "ml-scheduler-expanded-experiment-plan",
        "include_adaptive": include_adaptive,
        "run_count": len(specs),
        "source_plan_sha256": _canonical_sha256(dict(_read_yaml(source_plan))),
        "runs": [asdict(spec) for spec in specs],
    }
    document["plan_sha256"] = _canonical_sha256(document)
    return document


def _write_or_validate_plan_lock(
    path: Path, document: Mapping[str, Any], *, require_existing: bool
) -> bool:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise ClusterRunError(
                f"plan lock differs from the current registered plan: {path}. "
                "Use a new --plan-out path and review it before execution."
            )
        return False
    if require_existing:
        raise ClusterRunError(
            f"reviewed plan lock is missing: {path}; run without --execute first"
        )
    _atomic_json(path, document)
    return True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _workload_document_for_spec(
    spec: RunSpec,
    jobs: Sequence[Any],
    *,
    image: str,
    namespace: str,
    image_pull_secrets: Sequence[str],
) -> Dict[str, Any]:
    workload = build_workload_document(
        list(jobs),
        seed=spec.seed,
        load=spec.load,
        run_id=spec.run_id,
        scenario=spec.scenario,
        repetition=spec.repetition,
    )
    workload["run"].update({
        "config": spec.config,
        "scheduler_name": spec.scheduler_name,
        "manifest_set": spec.manifest_set,
        "pacing_mode": spec.pacing_mode,
        "fixed_delay_seconds": spec.fixed_delay_seconds,
        "reverse": spec.reverse,
        "namespace": namespace,
    })
    workload["container"] = {
        "image": image,
        "image_pull_secrets": list(image_pull_secrets),
    }
    return workload


def _expected_manifest(
    spec: RunSpec,
    job: Any,
    *,
    image: str,
    namespace: str,
    scheduling_gate: Optional[str],
    image_pull_secrets: Sequence[str],
) -> Mapping[str, Any]:
    return yaml.safe_load(to_pod_yaml(
        job,
        spec.scheduler_name,
        image,
        namespace,
        run_id=spec.run_id,
        scenario=spec.scenario,
        config=spec.config,
        repetition=spec.repetition,
        expected_jobs=spec.jobs,
        seed=spec.seed,
        pacing_mode=spec.pacing_mode,
        fixed_delay_seconds=spec.fixed_delay_seconds,
        reverse=spec.reverse,
        scheduling_gate=scheduling_gate if spec.scheduler_name == "default-scheduler" else None,
        load_profile=spec.load,
        image_pull_secrets=image_pull_secrets,
    ))


def validate_materialized_run(
    spec: RunSpec,
    *,
    output: Path,
    image: str,
    namespace: str,
    scheduling_gate: Optional[str],
    image_pull_secrets: Sequence[str],
) -> Path:
    if not (output / OUTPUT_SENTINEL).is_file():
        raise ClusterRunError(f"materialized run sentinel is missing: {output}")
    jobs = generate_burst(spec.jobs, seed=spec.seed, load=spec.load)
    expected_workload = _workload_document_for_spec(
        spec,
        jobs,
        image=image,
        namespace=namespace,
        image_pull_secrets=image_pull_secrets,
    )
    jobs_path = output / "jobs.json"
    if not jobs_path.is_file():
        raise ClusterRunError(f"materialized jobs.json is missing: {jobs_path}")
    observed_workload = json.loads(jobs_path.read_text(encoding="utf-8"))
    if _canonical_sha256(observed_workload) != _canonical_sha256(expected_workload):
        raise ClusterRunError(
            f"materialized workload contract differs from RunSpec: {output}"
        )

    manifest_directory = output / "pods"
    observed_paths = sorted(manifest_directory.glob("*.yaml"))
    expected_names = sorted(f"{job.job_id}.yaml" for job in jobs)
    if [path.name for path in observed_paths] != expected_names:
        raise ClusterRunError(
            f"materialized manifest set differs for {spec.run_id}: "
            f"expected {len(expected_names)}, observed {len(observed_paths)}"
        )
    jobs_by_name = {job.job_id: job for job in jobs}
    for path in observed_paths:
        expected_manifest = _expected_manifest(
            spec,
            jobs_by_name[path.stem],
            image=image,
            namespace=namespace,
            scheduling_gate=scheduling_gate,
            image_pull_secrets=image_pull_secrets,
        )
        observed_manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if observed_manifest != expected_manifest:
            raise ClusterRunError(f"materialized manifest has drifted: {path}")
    return manifest_directory


def artifact_hashes(run_root: Path) -> Dict[str, str]:
    paths = [run_root / "jobs.json", *sorted((run_root / "pods").glob("*.yaml"))]
    return {
        path.relative_to(run_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def materialize_run(
    spec: RunSpec,
    *,
    work_root: Path,
    image: str,
    namespace: str,
    overwrite: bool,
    scheduling_gate: Optional[str] = None,
    image_pull_secrets: Sequence[str] = (),
) -> Path:
    output = work_root / spec.run_id
    normalized_pull_secrets = tuple(dict.fromkeys(image_pull_secrets))
    if output.exists() and any(output.iterdir()) and not overwrite:
        return validate_materialized_run(
            spec,
            output=output,
            image=image,
            namespace=namespace,
            scheduling_gate=scheduling_gate,
            image_pull_secrets=normalized_pull_secrets,
        )
    prepare_output_directory(output, overwrite=overwrite)
    jobs = generate_burst(spec.jobs, seed=spec.seed, load=spec.load)
    workload = _workload_document_for_spec(
        spec,
        jobs,
        image=image,
        namespace=namespace,
        image_pull_secrets=normalized_pull_secrets,
    )
    save_json(jobs, output / "jobs.json", document=workload)

    manifest_directory = output / "pods"
    write_pod_manifests(
        jobs,
        manifest_directory,
        spec.scheduler_name,
        image,
        namespace=namespace,
        run_id=spec.run_id,
        scenario=spec.scenario,
        config=spec.config,
        repetition=spec.repetition,
        seed=spec.seed,
        pacing_mode=spec.pacing_mode,
        fixed_delay_seconds=spec.fixed_delay_seconds,
        reverse=spec.reverse,
        scheduling_gate=scheduling_gate if spec.scheduler_name == "default-scheduler" else None,
        load_profile=spec.load,
        image_pull_secrets=normalized_pull_secrets,
    )
    manifests = sorted(manifest_directory.glob("*.yaml"))
    if len(manifests) != spec.jobs:
        raise RuntimeError(f"materialized {len(manifests)} manifests; expected {spec.jobs}")
    return manifest_directory


def materialize_plan(
    specs: Sequence[RunSpec],
    *,
    work_root: Path,
    image: str,
    namespace: str,
    overwrite: bool,
    scheduling_gate: Optional[str] = None,
    image_pull_secrets: Sequence[str] = (),
) -> None:
    for spec in specs:
        materialize_run(
            spec,
            work_root=work_root,
            image=image,
            namespace=namespace,
            overwrite=overwrite,
            scheduling_gate=scheduling_gate,
            image_pull_secrets=image_pull_secrets,
        )


class Kubectl:
    def __init__(self, executable: str, context: Optional[str] = None):
        self.executable = executable
        self.context = context
        self.prefix = [executable]
        if context:
            self.prefix.extend(["--context", context])

    def run(
        self,
        *arguments: str,
        timeout: float = 60.0,
        check: bool = True,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [*self.prefix, *arguments],
            text=True,
            capture_output=True,
            input=input_text,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            raise ClusterRunError(
                f"command failed ({completed.returncode}): {' '.join([*self.prefix, *arguments])}\n"
                f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
            )
        return completed

    def pod_json(self, namespace: str, selector: str) -> Dict[str, Any]:
        result = self.run(
            "get", "pods", "-n", namespace, "-l", selector, "-o", "json", timeout=30,
        )
        return json.loads(result.stdout)

    def json(self, *arguments: str, timeout: float = 60.0) -> Dict[str, Any]:
        result = self.run(*arguments, "-o", "json", timeout=timeout)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ClusterRunError(
                f"kubectl returned invalid JSON for {' '.join(arguments)}"
            ) from exc
        if not isinstance(payload, dict):
            raise ClusterRunError(f"kubectl JSON for {' '.join(arguments)} is not an object")
        return payload


def _run_local_command(arguments: Sequence[str], *, timeout: float = 60.0) -> str:
    completed = subprocess.run(
        list(arguments),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise ClusterRunError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    return completed.stdout


def capture_minikube_environment(
    *, executable: str, profile: str
) -> Dict[str, Any]:
    version = _run_local_command([executable, "version"], timeout=30)
    try:
        profiles = json.loads(
            _run_local_command(
                [executable, "profile", "list", "--output", "json"], timeout=30
            )
        )
        status = json.loads(
            _run_local_command(
                [executable, "status", "--profile", profile, "--output", "json"],
                timeout=30,
            )
        )
    except json.JSONDecodeError as exc:
        raise ClusterRunError("Minikube returned invalid JSON attestation") from exc
    if not isinstance(profiles, Mapping) or not isinstance(status, Mapping):
        raise ClusterRunError("Minikube attestation must contain JSON objects")
    try:
        return validate_minikube_attestation(
            profiles, status, profile=profile, version=version
        )
    except ValueError as exc:
        raise ClusterRunError(str(exc)) from exc


def prewarm_trainer_image(
    kubectl: Kubectl,
    *,
    namespace: str,
    target_node: str,
    image: str,
    image_pull_secrets: Sequence[str],
) -> Dict[str, Any]:
    manifest = prewarm_pod_manifest(
        namespace=namespace,
        target_node=target_node,
        image=image,
        image_pull_secrets=image_pull_secrets,
    )
    name = str((manifest.get("metadata") or {})["name"])
    kubectl.run(
        "delete",
        "pod",
        name,
        "-n",
        namespace,
        "--ignore-not-found=true",
        "--wait=true",
        "--timeout=60s",
        timeout=90,
    )
    try:
        kubectl.run(
            "create",
            "-f",
            "-",
            input_text=json.dumps(manifest, separators=(",", ":")),
            timeout=60,
        )
        kubectl.run(
            "wait",
            "-n",
            namespace,
            f"pod/{name}",
            "--for=jsonpath={.status.phase}=Succeeded",
            "--timeout=300s",
            timeout=330,
        )
        pod = kubectl.json("get", "pod", name, "-n", namespace, timeout=30)
        logs = kubectl.run(
            "logs", name, "-n", namespace, "-c", "attest", timeout=30
        ).stdout
        return validate_prewarm_observation(
            pod, logs, expected_image=image, target_node=target_node
        )
    except ValueError as exc:
        raise ClusterRunError(f"trainer prewarm attestation failed: {exc}") from exc
    finally:
        kubectl.run(
            "delete",
            "pod",
            name,
            "-n",
            namespace,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=60s",
            timeout=90,
            check=False,
        )


def wait_for_cooldown(
    kubectl: Kubectl,
    *,
    namespace: str,
    target_node: str,
    scheduler_selector: str,
    expected_scheduler_uid: str,
    cooldown_seconds: float,
    minimum_clean_polls: int,
    poll_seconds: float = 2.0,
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    deadline = started + cooldown_seconds
    clean_polls = 0
    last_conditions: Dict[str, str] = {}
    while True:
        workloads = kubectl.pod_json(
            namespace, "app.kubernetes.io/name=ml-sim-job"
        ).get("items") or []
        if workloads:
            raise ClusterRunError("workload Pods appeared during cooldown")
        scheduler_pods = kubectl.pod_json(namespace, scheduler_selector).get("items") or []
        if len(scheduler_pods) != 1:
            raise ClusterRunError("scheduler Pod count changed during cooldown")
        scheduler = scheduler_pods[0]
        scheduler_metadata = scheduler.get("metadata") or {}
        scheduler_status = scheduler.get("status") or {}
        statuses = scheduler_status.get("containerStatuses") or []
        if (
            scheduler_metadata.get("uid") != expected_scheduler_uid
            or scheduler_status.get("phase") != "Running"
            or not statuses
            or any(
                not status.get("ready") or status.get("restartCount") != 0
                for status in statuses
            )
        ):
            raise ClusterRunError("scheduler continuity failed during cooldown")
        node = kubectl.json("get", "node", target_node, timeout=30)
        last_conditions = {
            str(condition.get("type")): str(condition.get("status"))
            for condition in ((node.get("status") or {}).get("conditions") or [])
        }
        pressure_clear = all(
            last_conditions.get(name) == "False"
            for name in ("MemoryPressure", "DiskPressure", "PIDPressure")
        )
        if last_conditions.get("Ready") != "True" or not pressure_clear:
            raise ClusterRunError("node became unready or pressured during cooldown")
        clean_polls += 1
        now = time.monotonic()
        if now >= deadline and clean_polls >= minimum_clean_polls:
            break
        remaining = max(0.0, deadline - now)
        time.sleep(min(poll_seconds, remaining) if remaining else poll_seconds)
    evidence = {
        "schema_version": "1.0",
        "policy": "fixed-window-clean-node-and-scheduler-continuity",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configured_seconds": cooldown_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "clean_polls": clean_polls,
        "workload_pods_observed": 0,
        "scheduler_uid": expected_scheduler_uid,
        "scheduler_continuity": True,
        "node_pressure_clear": True,
        "final_node_conditions": last_conditions,
    }
    errors = validate_cooldown_evidence(
        evidence,
        expected_seconds=cooldown_seconds,
        expected_clean_polls=minimum_clean_polls,
    )
    if errors:
        raise ClusterRunError("invalid cooldown evidence: " + "; ".join(errors))
    return evidence


def capture_and_verify_helm_release(
    *,
    executable: str,
    release: str,
    namespace: str,
    kube_context: str,
    target_node: str,
    require_metrics: bool,
) -> Dict[str, Any]:
    common = ["--namespace", namespace, "--kube-context", kube_context]
    version = _run_local_command([executable, "version", "--short"], timeout=30).strip()
    values = json.loads(_run_local_command(
        [executable, "get", "values", release, *common, "--all", "-o", "json"],
        timeout=30,
    ))
    status = json.loads(_run_local_command(
        [executable, "status", release, *common, "-o", "json"], timeout=30
    ))
    if values.get("mode") != "reproduction":
        raise ClusterRunError("Helm release is not in reproduction mode")
    scheduler = values.get("scheduler") or {}
    experiment = values.get("experiment") or {}
    if scheduler.get("targetNode") != target_node:
        raise ClusterRunError("Helm targetNode differs from --target-node")
    if scheduler.get("expectedCount") != 0 or experiment.get("runId"):
        raise ClusterRunError("Helm release is not using a dynamic matrix contract")
    if require_metrics and not (values.get("rbac") or {}).get("nodeMetricsRead"):
        raise ClusterRunError("adaptive plan requires rbac.nodeMetricsRead=true")
    release_status = (status.get("info") or {}).get("status")
    if str(release_status).lower() != "deployed":
        raise ClusterRunError(f"Helm release status is not deployed: {release_status!r}")
    return {
        "version": version,
        "release": release,
        "status": status,
        "computed_values": values,
    }


def _argument_value(arguments: Sequence[str], name: str) -> Optional[str]:
    try:
        index = list(arguments).index(name)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


SCHEDULER_RUNTIME_ARGUMENTS = {
    "--quiet-period": "quietPeriodSeconds",
    "--burst-timeout": "burstTimeoutSeconds",
    "--poll-interval": "pollIntervalSeconds",
    "--execution-timeout": "executionTimeoutSeconds",
    "--api-timeout": "apiTimeoutSeconds",
    "--cpu-threshold": "cpuThreshold",
    "--adaptive-hysteresis": "adaptiveHysteresis",
    "--max-wait": "maxWaitSeconds",
    "--metrics-max-age": "metricsMaxAgeSeconds",
}

SCHEDULER_RUNTIME_METADATA = {
    "quiet_period_seconds": "quietPeriodSeconds",
    "burst_timeout_seconds": "burstTimeoutSeconds",
    "poll_interval_seconds": "pollIntervalSeconds",
    "execution_timeout_seconds": "executionTimeoutSeconds",
    "api_timeout_seconds": "apiTimeoutSeconds",
    "cpu_threshold": "cpuThreshold",
    "adaptive_hysteresis": "adaptiveHysteresis",
    "max_wait_seconds": "maxWaitSeconds",
    "metrics_max_age_seconds": "metricsMaxAgeSeconds",
}


def validate_scheduler_argument_contract(
    arguments: Sequence[str],
    scheduler_values: Mapping[str, Any],
    *,
    target_node: str,
    results_template: str,
) -> None:
    """Fail when the live Deployment differs from computed Helm values."""

    expected_strings = {
        "--scheduler-name": scheduler_values.get("name"),
        "--target-node": scheduler_values.get("targetNode"),
        "--results": scheduler_values.get("resultsPath"),
    }
    if expected_strings["--target-node"] != target_node:
        raise ClusterRunError("computed Helm target node differs from --target-node")
    if expected_strings["--results"] != results_template:
        raise ClusterRunError("computed Helm results path differs from runner")
    for argument, expected in expected_strings.items():
        actual = _argument_value(arguments, argument)
        if not isinstance(expected, str) or not expected or actual != expected:
            raise ClusterRunError(
                f"live scheduler argument {argument}={actual!r} differs from "
                f"computed Helm value {expected!r}"
            )
    for argument, value_key in SCHEDULER_RUNTIME_ARGUMENTS.items():
        actual = _argument_value(arguments, argument)
        expected = scheduler_values.get(value_key)
        try:
            matches = math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ClusterRunError(
                f"live scheduler argument {argument}={actual!r} differs from "
                f"computed Helm {value_key}={expected!r}"
            )
    actual_retries = _argument_value(arguments, "--api-retries")
    expected_retries = scheduler_values.get("apiRetries")
    try:
        retries_match = int(actual_retries) == int(expected_retries)
    except (TypeError, ValueError):
        retries_match = False
    if not retries_match:
        raise ClusterRunError(
            f"live scheduler argument --api-retries={actual_retries!r} differs "
            f"from computed Helm apiRetries={expected_retries!r}"
        )


def verify_scheduler_deployment(
    kubectl: Kubectl,
    *,
    namespace: str,
    deployment_name: str,
    container_name: str,
    target_node: str,
    results_template: str,
    require_metrics: bool,
    scheduler_values: Mapping[str, Any],
) -> Dict[str, Any]:
    kubectl.run(
        "rollout", "status", f"deployment/{deployment_name}", "-n", namespace,
        "--timeout=180s", timeout=210,
    )
    deployment = kubectl.json(
        "get", "deployment", deployment_name, "-n", namespace, timeout=30
    )
    spec = deployment.get("spec") or {}
    status = deployment.get("status") or {}
    metadata = deployment.get("metadata") or {}
    if spec.get("replicas") != 1 or status.get("readyReplicas") != 1:
        raise ClusterRunError("scheduler Deployment must have exactly one Ready replica")
    if status.get("observedGeneration") != metadata.get("generation"):
        raise ClusterRunError("scheduler Deployment has not observed its latest generation")
    containers = (spec.get("template") or {}).get("spec", {}).get("containers") or []
    matches = [item for item in containers if item.get("name") == container_name]
    if len(matches) != 1:
        raise ClusterRunError(
            f"scheduler container {container_name!r} is missing or duplicated"
        )
    container = matches[0]
    arguments = [str(value) for value in (container.get("args") or [])]
    if "scheduler.custom_scheduler" not in arguments:
        raise ClusterRunError("installed controller is not the reproduction custom scheduler")
    if scheduler_values.get("name") != "ml-aware-scheduler":
        raise ClusterRunError("computed Helm scheduler name is not ml-aware-scheduler")
    validate_scheduler_argument_contract(
        arguments,
        scheduler_values,
        target_node=target_node,
        results_template=results_template,
    )
    if _argument_value(arguments, "--scheduler-name") != "ml-aware-scheduler":
        raise ClusterRunError("installed reproduction scheduler name is not ml-aware-scheduler")
    if _argument_value(arguments, "--target-node") != target_node:
        raise ClusterRunError("installed scheduler target node differs from --target-node")
    if _argument_value(arguments, "--results") != results_template:
        raise ClusterRunError("installed scheduler results template differs from runner")
    if "--run-id" in arguments or "--expected-count" in arguments:
        raise ClusterRunError(
            "installed scheduler has a fixed run contract; use values-reproduction-matrix.yaml"
        )
    image = str(container.get("image") or "")
    if not _IMMUTABLE_IMAGE_RE.fullmatch(image):
        raise ClusterRunError("scheduler Deployment image must be pinned by sha256 digest")

    pod_spec = (spec.get("template") or {}).get("spec") or {}
    service_account = pod_spec.get("serviceAccountName")
    if not service_account:
        raise ClusterRunError("scheduler Deployment has no ServiceAccount")
    identity = f"system:serviceaccount:{namespace}:{service_account}"
    authorization_checks = [
        ("get", "pods"),
        ("get", "pods/log"),
        ("create", "pods/binding"),
    ]
    for verb, resource in authorization_checks:
        answer = kubectl.run(
            "auth", "can-i", verb, resource, "-n", namespace,
            f"--as={identity}", timeout=30,
        ).stdout.strip().lower()
        if answer != "yes":
            raise ClusterRunError(
                f"scheduler ServiceAccount cannot {verb} {resource}: {answer!r}"
            )
    node_answer = kubectl.run(
        "auth", "can-i", "get", "nodes", f"--as={identity}", timeout=30
    ).stdout.strip().lower()
    if node_answer != "yes":
        raise ClusterRunError("scheduler ServiceAccount cannot get nodes")
    if require_metrics:
        metrics_answer = kubectl.run(
            "auth", "can-i", "get", "nodes.metrics.k8s.io",
            f"--as={identity}", timeout=30,
        ).stdout.strip().lower()
        if metrics_answer != "yes":
            raise ClusterRunError(
                "adaptive plan requires ServiceAccount get access to nodes.metrics.k8s.io"
            )
    exec_answer = kubectl.run(
        "auth", "can-i", "create", "pods/exec", "-n", namespace, timeout=30
    ).stdout.strip().lower()
    if exec_answer != "yes":
        raise ClusterRunError(
            "runner identity cannot create pods/exec to archive scheduler evidence"
        )

    node = kubectl.json("get", "node", target_node, timeout=30)
    node_spec = node.get("spec") or {}
    ready = any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in ((node.get("status") or {}).get("conditions") or [])
    )
    if not ready or node_spec.get("unschedulable"):
        raise ClusterRunError("target node is not Ready and schedulable")

    health_check = (
        "import urllib.request; "
        "response=urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=5); "
        "assert response.status == 200; print(response.read().decode())"
    )
    kubectl.run(
        "exec", "-n", namespace, f"deployment/{deployment_name}",
        "-c", container_name, "--", "python", "-c", health_check, timeout=30,
    )
    return deployment


def capture_cluster_environment(
    kubectl: Kubectl,
    *,
    namespace: str,
    target_node: str,
    deployment: Mapping[str, Any],
    scheduler_selector: str,
) -> Dict[str, Any]:
    version = kubectl.json("version", timeout=30)
    node = kubectl.json("get", "node", target_node, timeout=30)
    nodes = kubectl.json("get", "nodes", timeout=30)
    target_node_pods = kubectl.json(
        "get", "pods", "-A", "--field-selector", f"spec.nodeName={target_node}",
        timeout=30,
    )
    scheduler_pods = kubectl.pod_json(namespace, scheduler_selector)
    pod_inventory = []
    for pod in scheduler_pods.get("items") or []:
        statuses = {
            item.get("name"): item
            for item in ((pod.get("status") or {}).get("containerStatuses") or [])
        }
        pod_inventory.append({
            "name": (pod.get("metadata") or {}).get("name"),
            "uid": (pod.get("metadata") or {}).get("uid"),
            "node_name": (pod.get("spec") or {}).get("nodeName"),
            "phase": (pod.get("status") or {}).get("phase"),
            "containers": [
                {
                    "name": container.get("name"),
                    "requested_image": container.get("image"),
                    "image_id": (statuses.get(container.get("name")) or {}).get("imageID"),
                    "ready": (statuses.get(container.get("name")) or {}).get("ready"),
                    "restart_count": (statuses.get(container.get("name")) or {}).get("restartCount"),
                }
                for container in ((pod.get("spec") or {}).get("containers") or [])
            ],
        })
    node_status = node.get("status") or {}
    target_inventory = [
        {
            "namespace": (pod.get("metadata") or {}).get("namespace"),
            "name": (pod.get("metadata") or {}).get("name"),
            "uid": (pod.get("metadata") or {}).get("uid"),
            "phase": (pod.get("status") or {}).get("phase"),
            "labels": (pod.get("metadata") or {}).get("labels") or {},
        }
        for pod in (target_node_pods.get("items") or [])
    ]
    return {
        "kubectl_context": kubectl.context,
        "namespace": namespace,
        "kubernetes_version": version,
        "cluster_nodes": [
            {
                "name": (item.get("metadata") or {}).get("name"),
                "uid": (item.get("metadata") or {}).get("uid"),
            }
            for item in (nodes.get("items") or [])
        ],
        "target_node_pods": target_inventory,
        "target_node": {
            "name": (node.get("metadata") or {}).get("name"),
            "uid": (node.get("metadata") or {}).get("uid"),
            "labels": (node.get("metadata") or {}).get("labels") or {},
            "capacity": node_status.get("capacity") or {},
            "allocatable": node_status.get("allocatable") or {},
            "node_info": node_status.get("nodeInfo") or {},
            "conditions": node_status.get("conditions") or [],
        },
        "scheduler_deployment": {
            "name": (deployment.get("metadata") or {}).get("name"),
            "uid": (deployment.get("metadata") or {}).get("uid"),
            "generation": (deployment.get("metadata") or {}).get("generation"),
            "labels": (deployment.get("metadata") or {}).get("labels") or {},
            "pods": pod_inventory,
        },
    }


def verify_adaptive_metrics_cadence(
    kubectl: Kubectl,
    *,
    target_node: str,
    max_wait_seconds: float,
    metrics_max_age_seconds: float,
) -> Dict[str, Any]:
    if max_wait_seconds < 10:
        raise ClusterRunError("adaptive max-wait must be at least 10 seconds")
    path = f"/apis/metrics.k8s.io/v1beta1/nodes/{target_node}"

    def sample() -> Dict[str, Any]:
        result = kubectl.run("get", f"--raw={path}", timeout=30)
        try:
            payload = json.loads(result.stdout)
            timestamp = str(payload["timestamp"])
            normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
            observed = datetime.fromisoformat(normalized)
            if observed.tzinfo is None:
                raise ValueError("metrics timestamp has no timezone")
            age = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
            if age < -5 or age > metrics_max_age_seconds:
                raise ClusterRunError(
                    f"metrics sample age {age:.3f}s exceeds configured freshness"
                )
            return {"timestamp": timestamp, "age_seconds": max(0.0, age)}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ClusterRunError("metrics API returned an invalid node sample") from exc

    first = sample()
    started = time.monotonic()
    deadline = started + max_wait_seconds * 0.8
    while time.monotonic() < deadline:
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))
        current = sample()
        if current["timestamp"] != first["timestamp"]:
            elapsed = time.monotonic() - started
            return {
                "first_timestamp": first["timestamp"],
                "second_timestamp": current["timestamp"],
                "observed_advance_seconds": elapsed,
                "max_wait_seconds": max_wait_seconds,
                "metrics_max_age_seconds": metrics_max_age_seconds,
            }
    raise ClusterRunError(
        "metrics timestamp did not advance within 80% of adaptive max-wait"
    )


def read_scheduler_record(
    kubectl: Kubectl,
    *,
    namespace: str,
    deployment_name: str,
    container_name: str,
    results_template: str,
    run_id: str,
) -> Dict[str, Any]:
    path = results_template.format(run_id=run_id)
    result = kubectl.run(
        "exec", "-n", namespace, f"deployment/{deployment_name}",
        "-c", container_name, "--", "cat", path, timeout=30,
    )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ClusterRunError(f"scheduler record is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ClusterRunError(f"scheduler record is not an object: {path}")
    return document


def validate_scheduler_record(
    spec: RunSpec,
    schedule: Mapping[str, Any],
    result_document: Mapping[str, Any],
    *,
    target_node: str,
) -> None:
    errors: List[str] = []
    if schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        errors.append("unsupported scheduler record schema")
    if schedule.get("status") != "completed" or schedule.get("error") is not None:
        errors.append("scheduler record is not completed")
    metadata = schedule.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append("scheduler metadata is missing")
        metadata = {}
    expected_metadata = {
        "profile": "article-manual-bind",
        "run_id": spec.run_id,
        "expected_count": spec.jobs,
        "scheduler_name": spec.scheduler_name,
        "target_node": target_node,
        "pacing_mode": spec.pacing_mode,
        "reverse": spec.reverse,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            errors.append(
                f"scheduler metadata {field}={metadata.get(field)!r} expected={expected!r}"
            )
    try:
        if not math.isclose(
            float(metadata.get("fixed_delay")),
            spec.fixed_delay_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append("scheduler fixed delay differs from plan")
    except (TypeError, ValueError):
        errors.append("scheduler fixed delay is invalid")

    runtime_values = (
        (((
            (result_document.get("environment") or {}).get("cluster_snapshot")
            or {}
        ).get("helm") or {}).get("computed_values") or {}).get("scheduler")
        or {}
    )
    if not isinstance(runtime_values, Mapping):
        runtime_values = {}
    if metadata.get("runtime_contract_version") != "1.0":
        errors.append("scheduler runtime contract version is missing or unsupported")
    for metadata_field, helm_field in SCHEDULER_RUNTIME_METADATA.items():
        observed = metadata.get(metadata_field)
        expected = runtime_values.get(helm_field)
        try:
            matches = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(
                f"scheduler runtime {metadata_field}={observed!r} differs "
                f"from Helm {helm_field}={expected!r}"
            )
    try:
        retries_match = int(metadata.get("api_retries")) == int(
            runtime_values.get("apiRetries")
        )
    except (TypeError, ValueError):
        retries_match = False
    if not retries_match:
        errors.append("scheduler runtime api_retries differs from Helm")

    records = schedule.get("records")
    if not isinstance(records, list) or len(records) != spec.jobs:
        errors.append(f"scheduler record count must be {spec.jobs}")
        records = []
    jobs = {
        job.get("job_id"): job
        for job in (result_document.get("jobs") or [])
        if isinstance(job, Mapping)
    }
    submission_jobs = (
        (((result_document.get("environment") or {}).get("submission") or {}).get("jobs"))
        or []
    )
    submitted_uids = {
        item.get("job_id"): item.get("uid")
        for item in submission_jobs
        if isinstance(item, Mapping)
    }
    record_ids = [record.get("job_id") for record in records if isinstance(record, Mapping)]
    if len(record_ids) != len(set(record_ids)) or set(record_ids) != set(jobs):
        errors.append("scheduler record job set differs from collected result")
    ordered = sorted(
        (record for record in records if isinstance(record, Mapping)),
        key=lambda record: record.get("order", 0),
    )
    if [record.get("order") for record in ordered] != list(range(1, spec.jobs + 1)):
        errors.append("scheduler order is not the complete 1..N sequence")
    ranks: List[float] = []
    for record in ordered:
        job_id = record.get("job_id")
        job = jobs.get(job_id)
        pod_uid = record.get("pod_uid")
        if not isinstance(pod_uid, str) or not pod_uid:
            errors.append(f"scheduler record {job_id!r} has no pod_uid")
        elif submitted_uids.get(job_id) not in (None, pod_uid):
            errors.append(f"scheduler Pod UID differs for {job_id!r}")
        if record.get("status") != "execution_started":
            errors.append(f"scheduler record for {job_id!r} is not execution_started")
        for field in ("rank", "bind_time", "exec_start_time"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"scheduler record {job_id!r} has invalid {field}")
        if job is not None and isinstance(record.get("rank"), (int, float)):
            if not math.isclose(float(record["rank"]), float(job["rank"]), rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"scheduler rank differs for {job_id!r}")
        if job is not None and isinstance(record.get("exec_start_time"), (int, float)):
            if not math.isclose(
                float(record["exec_start_time"]),
                float(job["execution_start_time"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                errors.append(f"scheduler execution marker differs for {job_id!r}")
        if isinstance(record.get("rank"), (int, float)):
            ranks.append(float(record["rank"]))
    if len(ranks) == spec.jobs:
        pairs = zip(ranks, ranks[1:], strict=False)
        order_is_valid = all(
            left <= right + 1e-12 if spec.reverse else left + 1e-12 >= right
            for left, right in pairs
        )
        if not order_is_valid:
            errors.append("scheduler rank direction differs from plan")
    events = schedule.get("events")
    if not isinstance(events, list):
        errors.append("scheduler pacing event list is missing")
        events = []
    starts = {
        event.get("after_job_id"): event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "pacing_wait_started"
    }
    completions = {
        event.get("after_job_id"): event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "pacing_wait_completed"
    }
    if len(starts) != max(0, spec.jobs - 1) or set(starts) != set(completions):
        errors.append("scheduler pacing waits are incomplete")
    for job_id, started in starts.items():
        completed = completions.get(job_id) or {}
        if started.get("mode") != spec.pacing_mode or completed.get("mode") != spec.pacing_mode:
            errors.append(f"pacing event mode differs for {job_id!r}")
        if spec.pacing_mode == "fixed":
            try:
                elapsed = float(completed["timestamp"]) - float(started["timestamp"])
                if elapsed + 0.05 < spec.fixed_delay_seconds:
                    errors.append(
                        f"fixed pacing wait for {job_id!r} was {elapsed:.3f}s, "
                        f"below {spec.fixed_delay_seconds:.3f}s"
                    )
            except (KeyError, TypeError, ValueError):
                errors.append(f"fixed pacing timestamps are invalid for {job_id!r}")
    if spec.pacing_mode == "adaptive":
        if not any(
            isinstance(event, Mapping) and event.get("event") == "adaptive_metrics_sample"
            for event in events
        ):
            errors.append("adaptive scheduler record has no metrics-sample evidence")
    if errors:
        raise ClusterRunError("invalid scheduler evidence: " + "; ".join(errors))


def _ensure_clean_cluster(kubectl: Kubectl, namespace: str) -> None:
    payload = kubectl.pod_json(namespace, "app.kubernetes.io/name=ml-sim-job")
    if payload.get("items"):
        names = [item["metadata"]["name"] for item in payload["items"]]
        raise ClusterRunError(
            f"experiment namespace is not clean; found workload Pods: {names}. "
            "Remove or archive them before executing the matrix."
        )


def _wait_for_terminal_pods(
    kubectl: Kubectl,
    spec: RunSpec,
    *,
    namespace: str,
    timeout_seconds: float,
    poll_seconds: float,
    failure_probe: Optional[Callable[[], None]] = None,
    failure_probe_interval: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    selector = f"ml.scheduler/run-id={spec.run_id}"
    last_phases: Dict[str, str] = {}
    next_failure_probe = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if failure_probe is not None and now >= next_failure_probe:
            failure_probe()
            next_failure_probe = now + failure_probe_interval
        items = kubectl.pod_json(namespace, selector).get("items", [])
        last_phases = {item["metadata"]["name"]: item.get("status", {}).get("phase", "Unknown") for item in items}
        failed = [name for name, phase in last_phases.items() if phase == "Failed"]
        if failed:
            raise ClusterRunError(f"run {spec.run_id} has failed Pods: {failed}")
        if len(items) == spec.jobs and all(phase == "Succeeded" for phase in last_phases.values()):
            return
        time.sleep(poll_seconds)
    raise ClusterRunError(
        f"run {spec.run_id} timed out after {timeout_seconds}s; "
        f"observed {len(last_phases)}/{spec.jobs} Pods, phases={last_phases}"
    )


def probe_scheduler_run(
    kubectl: Kubectl,
    *,
    namespace: str,
    deployment_name: str,
    container_name: str,
    scheduler_selector: str,
    expected_pod_uid: str,
    results_template: str,
    spec: RunSpec,
) -> None:
    scheduler_pods = kubectl.pod_json(namespace, scheduler_selector).get("items") or []
    if len(scheduler_pods) != 1:
        raise ClusterRunError("scheduler Pod count changed during experiment run")
    pod = scheduler_pods[0]
    metadata = pod.get("metadata") or {}
    status = pod.get("status") or {}
    if metadata.get("uid") != expected_pod_uid:
        raise ClusterRunError("scheduler Pod was replaced during experiment run")
    container_statuses = {
        item.get("name"): item for item in (status.get("containerStatuses") or [])
    }
    scheduler_status = container_statuses.get(container_name) or {}
    if (
        status.get("phase") != "Running"
        or not scheduler_status.get("ready")
        or scheduler_status.get("restartCount") != 0
    ):
        raise ClusterRunError("scheduler Pod became unhealthy or restarted during run")
    if spec.scheduler_name == "default-scheduler":
        return
    path = results_template.format(run_id=spec.run_id)
    record = kubectl.run(
        "exec", "-n", namespace, f"deployment/{deployment_name}",
        "-c", container_name, "--", "cat", path, timeout=30, check=False,
    )
    if record.returncode != 0:
        return
    try:
        document = json.loads(record.stdout)
    except json.JSONDecodeError as exc:
        raise ClusterRunError("scheduler emitted an invalid intermediate record") from exc
    if document.get("status") == "failed":
        raise ClusterRunError(
            f"scheduler rejected run {spec.run_id}: {document.get('error') or 'unknown error'}"
        )


def _cleanup_run(kubectl: Kubectl, spec: RunSpec, *, namespace: str, timeout_seconds: float) -> None:
    selector = f"ml.scheduler/run-id={spec.run_id}"
    kubectl.run(
        "delete", "pods", "-n", namespace, "-l", selector,
        "--wait=true", f"--timeout={int(timeout_seconds)}s", "--ignore-not-found=true",
        timeout=timeout_seconds + 30,
    )


def _pod_contract_for_spec(
    spec: RunSpec,
    *,
    image: str,
    image_pull_secrets: Sequence[str],
) -> Dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "scheduler_name": spec.scheduler_name,
        "pacing_mode": spec.pacing_mode,
        "fixed_delay_seconds": spec.fixed_delay_seconds,
        "reverse": spec.reverse,
        "expected_jobs": spec.jobs,
        "seed": spec.seed,
        "load_profile": spec.load,
        "work_model_version": WORK_MODEL_VERSION,
        "scheduling_gate": None,
        "image": image,
        "image_pull_secrets": list(image_pull_secrets),
    }


def _attach_execution_evidence(
    document: Dict[str, Any],
    *,
    plan_sha256: str,
    cluster_environment: Mapping[str, Any],
    manifest_directory: Path,
    trainer_image: str,
    image_pull_secrets: Sequence[str],
    scheduler_deployment: str,
    target_node: str,
    kube_context: str,
    submission_evidence: Mapping[str, Any],
) -> None:
    environment = document.setdefault("environment", {})
    environment["orchestration"] = {
        "plan_sha256": plan_sha256,
        "kubectl_context": kube_context,
        "trainer_image": trainer_image,
        "image_pull_secrets": list(image_pull_secrets),
        "scheduler_deployment": scheduler_deployment,
        "target_node": target_node,
        "artifact_sha256": artifact_hashes(manifest_directory.parent),
        "cluster_snapshot_sha256": _canonical_sha256(cluster_environment),
        "submission_sha256": _canonical_sha256(submission_evidence),
    }
    environment["cluster_snapshot"] = dict(cluster_environment)
    environment["submission"] = dict(submission_evidence)


def validate_result_for_spec(
    document: Mapping[str, Any],
    spec: RunSpec,
    *,
    plan_sha256: str,
    target_node: str,
    require_article_environment: bool = False,
) -> None:
    validate_result_document(document, strict=True)
    run = document.get("run") or {}
    expected_run = {
        "run_id": spec.run_id,
        "scenario": spec.scenario,
        "config": spec.config,
        "repetition": spec.repetition,
        "seed": spec.seed,
        "expected_jobs": spec.jobs,
    }
    mismatches = {
        key: (run.get(key), expected)
        for key, expected in expected_run.items()
        if run.get(key) != expected
    }
    if document.get("source") != "kubernetes":
        mismatches["source"] = (document.get("source"), "kubernetes")
    orchestration = (document.get("environment") or {}).get("orchestration") or {}
    if orchestration.get("plan_sha256") != plan_sha256:
        mismatches["plan_sha256"] = (
            orchestration.get("plan_sha256"), plan_sha256
        )
    if orchestration.get("target_node") != target_node:
        mismatches["target_node"] = (
            orchestration.get("target_node"), target_node
        )
    trainer_image = orchestration.get("trainer_image")
    if not isinstance(trainer_image, str) or not _IMMUTABLE_IMAGE_RE.fullmatch(trainer_image):
        mismatches["trainer_image"] = (trainer_image, "digest-qualified image")
    for field in ("kubectl_context", "scheduler_deployment"):
        if not isinstance(orchestration.get(field), str) or not orchestration.get(field):
            mismatches[field] = (orchestration.get(field), "non-empty string")
    artifact_digests = orchestration.get("artifact_sha256")
    if not isinstance(artifact_digests, Mapping) or len(artifact_digests) != spec.jobs + 1 or any(
        not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
        for value in (artifact_digests.values() if isinstance(artifact_digests, Mapping) else [])
    ):
        mismatches["artifact_sha256"] = (artifact_digests, f"{spec.jobs + 1} sha256 values")
    snapshot_digest = orchestration.get("cluster_snapshot_sha256")
    if not isinstance(snapshot_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", snapshot_digest):
        mismatches["cluster_snapshot_sha256"] = (snapshot_digest, "sha256")
    environment = document.get("environment") or {}
    snapshot = environment.get("cluster_snapshot")
    snapshot_target = snapshot.get("target_node") if isinstance(snapshot, Mapping) else None
    if not isinstance(snapshot_target, Mapping) or snapshot_target.get("name") != target_node:
        mismatches["cluster_snapshot"] = (snapshot, f"target node {target_node!r}")
    elif snapshot_digest != _canonical_sha256(snapshot):
        mismatches["cluster_snapshot_sha256"] = (
            snapshot_digest, _canonical_sha256(snapshot)
        )
    if isinstance(snapshot, Mapping):
        if not isinstance(snapshot.get("kubernetes_version"), Mapping):
            mismatches["kubernetes_version"] = (
                snapshot.get("kubernetes_version"), "version object"
            )
        helm_snapshot = snapshot.get("helm")
        if not isinstance(helm_snapshot, Mapping) or not helm_snapshot.get("version"):
            mismatches["helm"] = (helm_snapshot, "captured Helm release metadata")
        policy = snapshot.get("reproduction_policy")
        if require_article_environment and (
            not isinstance(policy, Mapping)
            or policy.get("profile") != "article-exact"
            or policy.get("article_claim_eligible") is not True
            or policy.get("errors") != []
        ):
            mismatches["reproduction_policy"] = (
                policy, "eligible article-exact policy"
            )
        if require_article_environment:
            minikube = snapshot.get("minikube")
            if (
                not isinstance(minikube, Mapping)
                or minikube.get("driver") != "docker"
                or str(minikube.get("profile_status") or "").lower()
                != "running"
            ):
                mismatches["minikube"] = (
                    minikube,
                    "running Docker-driver profile attestation",
                )
            control_errors = validate_execution_controls_evidence(
                snapshot.get("execution_controls") or {},
                expected_image=str(trainer_image),
                target_node=target_node,
            )
            if control_errors:
                mismatches["execution_controls"] = (
                    control_errors,
                    "valid prewarm, BLAS and cooldown evidence",
                )

    submission = environment.get("submission")
    if not isinstance(submission, Mapping):
        mismatches["submission"] = (submission, "object")
    else:
        expected_job_ids = {
            str(job.get("job_id")) for job in (document.get("jobs") or [])
        }
        submission_rows = submission.get("jobs") or []
        submitted_job_ids = {
            str(row.get("job_id"))
            for row in submission_rows
            if isinstance(row, Mapping) and row.get("status") == "created"
        }
        if (
            submission.get("mode") != "concurrent-client-barrier"
            or submission.get("seed") != spec.seed
            or submission.get("expected_jobs") != spec.jobs
            or len(submission_rows) != spec.jobs
            or submitted_job_ids != expected_job_ids
        ):
            mismatches["submission.contract"] = (
                submission, "complete concurrent submission matching RunSpec"
            )
        spread = submission.get("server_creation_spread_seconds")
        spread_limit = submission.get("max_creation_spread_seconds")
        if (
            isinstance(spread, bool)
            or not isinstance(spread, (int, float))
            or isinstance(spread_limit, bool)
            or not isinstance(spread_limit, (int, float))
            or not 0 <= float(spread) <= float(spread_limit)
        ):
            mismatches["submission.server_creation_spread_seconds"] = (
                spread, f"finite value within {spread_limit!r}"
            )
        submitted_at = {
            str(row.get("job_id")): row.get("server_creation_timestamp")
            for row in submission_rows
            if isinstance(row, Mapping)
        }
        for job in document.get("jobs") or []:
            job_id = str(job.get("job_id"))
            observed = job.get("submission_time")
            expected = submitted_at.get(job_id)
            if (
                isinstance(expected, bool)
                or not isinstance(expected, (int, float))
                or isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(
                    float(observed), float(expected), rel_tol=0.0, abs_tol=1e-6
                )
            ):
                mismatches[f"submission.timestamp.{job_id}"] = (observed, expected)
        submission_digest = orchestration.get("submission_sha256")
        expected_digest = _canonical_sha256(submission)
        if submission_digest != expected_digest:
            mismatches["submission_sha256"] = (
                submission_digest, expected_digest
            )
    kubernetes_environment = environment.get("kubernetes")
    workload_pods = (
        kubernetes_environment.get("workload_pods")
        if isinstance(kubernetes_environment, Mapping)
        else None
    )
    if not isinstance(workload_pods, list) or len(workload_pods) != spec.jobs:
        mismatches["workload_pods"] = (
            len(workload_pods) if isinstance(workload_pods, list) else None,
            spec.jobs,
        )
    wrong_nodes = [
        job.get("job_id")
        for job in (document.get("jobs") or [])
        if isinstance(job, Mapping) and job.get("node_name") != target_node
    ]
    if wrong_nodes:
        mismatches["job_node_names"] = (wrong_nodes, target_node)
    if require_article_environment:
        invalid_trainer_evidence = []
        for job in document.get("jobs") or []:
            evidence = job.get("trainer_evidence") if isinstance(job, Mapping) else None
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("work_model_version") != WORK_MODEL_VERSION
                or evidence.get("blas_threads") != REPRODUCTION_BLAS_THREADS
                or not isinstance(evidence.get("blas_library_count"), int)
                or evidence.get("blas_library_count", 0) <= 0
            ):
                invalid_trainer_evidence.append(
                    job.get("job_id") if isinstance(job, Mapping) else None
                )
        if invalid_trainer_evidence:
            mismatches["trainer_evidence"] = (
                invalid_trainer_evidence,
                "versioned work model and single-thread BLAS evidence for every job",
            )
    if mismatches:
        raise ClusterRunError(f"result does not match RunSpec: {mismatches}")
    if spec.scheduler_name != "default-scheduler":
        schedule = document.get("scheduler_record")
        if not isinstance(schedule, Mapping):
            raise ClusterRunError("custom run result is missing scheduler_record evidence")
        validate_scheduler_record(
            spec, schedule, document, target_node=target_node
        )


def execute_run(
    spec: RunSpec,
    *,
    kubectl: Kubectl,
    namespace: str,
    manifest_directory: Path,
    results_root: Path,
    timeout_seconds: float,
    poll_seconds: float,
    cleanup_on_failure: bool,
    kube_context: str,
    trainer_image: str,
    image_pull_secrets: Sequence[str],
    scheduler_deployment: str,
    scheduler_container: str,
    scheduler_results_template: str,
    target_node: str,
    plan_sha256: str,
    cluster_environment: Mapping[str, Any],
    submit_manifest_directory: Callable[[Path, RunSpec], Mapping[str, Any]],
    require_article_environment: bool,
) -> Dict[str, Any]:
    selector = f"ml.scheduler/run-id={spec.run_id}"
    scheduler_pods = (
        (cluster_environment.get("scheduler_deployment") or {}).get("pods") or []
    )
    if len(scheduler_pods) != 1 or not scheduler_pods[0].get("uid"):
        raise ClusterRunError("cluster snapshot does not contain one scheduler Pod UID")
    expected_scheduler_pod_uid = str(scheduler_pods[0]["uid"])
    existing = kubectl.pod_json(namespace, selector).get("items", [])
    if existing:
        raise ClusterRunError(f"run selector already exists before apply: {selector}")
    submission_evidence: Optional[Mapping[str, Any]] = None
    try:
        submission_evidence = submit_manifest_directory(manifest_directory, spec)
        _wait_for_terminal_pods(
            kubectl,
            spec,
            namespace=namespace,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            failure_probe=lambda: probe_scheduler_run(
                kubectl,
                namespace=namespace,
                deployment_name=scheduler_deployment,
                container_name=scheduler_container,
                scheduler_selector="app.kubernetes.io/component=scheduler",
                expected_pod_uid=expected_scheduler_pod_uid,
                results_template=scheduler_results_template,
                spec=spec,
            ),
        )
        try:
            expected_contract = _pod_contract_for_spec(
                spec,
                image=trainer_image,
                image_pull_secrets=image_pull_secrets,
            )
            document = collect(
                namespace,
                selector,
                expected_count=spec.jobs,
                run_id=spec.run_id,
                scenario=spec.scenario,
                config_name=spec.config,
                repetition=spec.repetition,
                seed=spec.seed,
                strict=True,
                kube_config_mode="kubeconfig",
                kube_context=kube_context,
                expected_pod_contract=expected_contract,
            )
        except IncompleteRunError as exc:
            document = exc.document
            _attach_execution_evidence(
                document,
                plan_sha256=plan_sha256,
                cluster_environment=cluster_environment,
                manifest_directory=manifest_directory,
                trainer_image=trainer_image,
                image_pull_secrets=image_pull_secrets,
                scheduler_deployment=scheduler_deployment,
                target_node=target_node,
                kube_context=kube_context,
                submission_evidence=submission_evidence,
            )
            _atomic_json(results_root / "failed" / f"{spec.run_id}.json", document)
            raise
        _attach_execution_evidence(
            document,
            plan_sha256=plan_sha256,
            cluster_environment=cluster_environment,
            manifest_directory=manifest_directory,
            trainer_image=trainer_image,
            image_pull_secrets=image_pull_secrets,
            scheduler_deployment=scheduler_deployment,
            target_node=target_node,
            kube_context=kube_context,
            submission_evidence=submission_evidence,
        )
        if spec.scheduler_name != "default-scheduler":
            schedule = read_scheduler_record(
                kubectl,
                namespace=namespace,
                deployment_name=scheduler_deployment,
                container_name=scheduler_container,
                results_template=scheduler_results_template,
                run_id=spec.run_id,
            )
            validate_scheduler_record(
                spec, schedule, document, target_node=target_node
            )
            document["scheduler_record"] = schedule
            _atomic_json(results_root / "schedules" / f"{spec.run_id}.json", schedule)
        validate_result_for_spec(
            document,
            spec,
            plan_sha256=plan_sha256,
            target_node=target_node,
            require_article_environment=require_article_environment,
        )
        _atomic_json(results_root / "runs" / f"{spec.run_id}.json", document)
        _cleanup_run(kubectl, spec, namespace=namespace, timeout_seconds=120)
        return document
    except Exception as exc:
        if submission_evidence is None and isinstance(exc, BurstSubmissionError):
            submission_evidence = exc.evidence
        failed_path = results_root / "failed" / f"{spec.run_id}.json"
        if not failed_path.exists():
            try:
                partial = collect(
                    namespace,
                    selector,
                    expected_count=spec.jobs,
                    run_id=spec.run_id,
                    scenario=spec.scenario,
                    config_name=spec.config,
                    repetition=spec.repetition,
                    seed=spec.seed,
                    strict=False,
                    kube_config_mode="kubeconfig",
                    kube_context=kube_context,
                    expected_pod_contract=_pod_contract_for_spec(
                        spec,
                        image=trainer_image,
                        image_pull_secrets=image_pull_secrets,
                    ),
                )
                if submission_evidence is not None:
                    _attach_execution_evidence(
                        partial,
                        plan_sha256=plan_sha256,
                        cluster_environment=cluster_environment,
                        manifest_directory=manifest_directory,
                        trainer_image=trainer_image,
                        image_pull_secrets=image_pull_secrets,
                        scheduler_deployment=scheduler_deployment,
                        target_node=target_node,
                        kube_context=kube_context,
                        submission_evidence=submission_evidence,
                    )
                partial["run"]["status"] = "failed"
                partial["failures"].append({
                    "code": "ORCHESTRATION_FAILED",
                    "message": str(exc),
                })
                _atomic_json(failed_path, partial)
            except Exception:
                # Preserve the original orchestration exception; Pods remain for diagnosis.
                pass
        if cleanup_on_failure:
            _cleanup_run(kubectl, spec, namespace=namespace, timeout_seconds=120)
        raise


def _select_specs(specs: Sequence[RunSpec], start_at: Optional[str], limit: Optional[int]) -> List[RunSpec]:
    selected = list(specs)
    if start_at:
        positions = [index for index, spec in enumerate(selected) if spec.run_id == start_at]
        if not positions:
            raise ValueError(f"--start-at run ID not found: {start_at}")
        selected = selected[positions[0]:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        selected = selected[:limit]
    return selected


def build_burst_submitter(
    *,
    kube_context: str,
    namespace: str,
    worker_ceiling: int,
    max_creation_spread_seconds: float,
) -> tuple[Callable[[Path, RunSpec], Mapping[str, Any]], Any]:
    """Build one pooled Kubernetes API client for all registered bursts."""

    from kubernetes import client
    from kubernetes import config as kubernetes_config

    configuration = client.Configuration()
    kubernetes_config.load_kube_config(
        context=kube_context, client_configuration=configuration
    )
    configuration.connection_pool_maxsize = max(
        int(configuration.connection_pool_maxsize or 0), worker_ceiling
    )
    api_client = client.ApiClient(configuration=configuration)
    core_api = client.CoreV1Api(api_client)

    def create_pod(pod_namespace: str, body: Mapping[str, Any]) -> Any:
        return core_api.create_namespaced_pod(
            pod_namespace,
            body,
            _request_timeout=(10, 120),
        )

    def submit(directory: Path, spec: RunSpec) -> Mapping[str, Any]:
        return submit_burst(
            directory,
            expected_namespace=namespace,
            expected_count=spec.jobs,
            seed=spec.seed,
            create_pod=create_pod,
            max_workers=worker_ceiling,
            max_creation_spread_seconds=max_creation_spread_seconds,
        )

    return submit, api_client


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--include-adaptive", action="store_true")
    parser.add_argument("--plan-out", type=Path, default=ROOT / "results" / "cluster_plan.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "workload" / "cluster_runs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "cluster")
    parser.add_argument("--image", default="ml-sim-job:0.1.0")
    parser.add_argument("--namespace", default="ai-scheduler")
    parser.add_argument("--scheduling-gate", default=None)
    parser.add_argument(
        "--image-pull-secret", action="append", default=[],
        help="registry Secret attached to workload Pods; repeat as needed",
    )
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--overwrite-workloads", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--helm", default="helm")
    parser.add_argument("--minikube", default="minikube")
    parser.add_argument("--minikube-profile", default=None)
    parser.add_argument("--helm-release", default="ml-ai-scheduler")
    parser.add_argument("--context", default=None)
    parser.add_argument("--target-node", default=None)
    parser.add_argument(
        "--scheduler-deployment", default=DEFAULT_SCHEDULER_DEPLOYMENT
    )
    parser.add_argument("--scheduler-container", default=DEFAULT_SCHEDULER_CONTAINER)
    parser.add_argument(
        "--scheduler-results-template", default=DEFAULT_SCHEDULER_RESULTS_TEMPLATE
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS
    )
    parser.add_argument(
        "--cooldown-clean-polls", type=int, default=DEFAULT_COOLDOWN_CLEAN_POLLS
    )
    parser.add_argument(
        "--environment-profile",
        choices=("article-exact", "record-only"),
        default="article-exact",
        help="fail closed on the published environment or explicitly run an ineligible smoke test",
    )
    parser.add_argument(
        "--submission-workers",
        type=int,
        default=64,
        help="Kubernetes API connection/worker ceiling; must cover the largest burst",
    )
    parser.add_argument(
        "--max-submission-spread-seconds",
        type=float,
        default=5.0,
        help="reject a run when API-server Pod creation spans longer than this",
    )
    parser.add_argument("--cleanup-on-failure", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        parser.error("timeouts and polling interval must be > 0")
    if not math.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be finite and >= 0")
    if args.cooldown_clean_polls <= 0:
        parser.error("--cooldown-clean-polls must be > 0")
    if args.submission_workers <= 0:
        parser.error("--submission-workers must be > 0")
    if (
        not math.isfinite(args.max_submission_spread_seconds)
        or args.max_submission_spread_seconds <= 0
    ):
        parser.error("--max-submission-spread-seconds must be finite and > 0")
    if "{run_id}" not in args.scheduler_results_template:
        parser.error("--scheduler-results-template must contain {run_id}")
    if args.execute and not args.context:
        parser.error("--context is required with --execute")
    if args.execute and not args.target_node:
        parser.error("--target-node is required with --execute")
    if args.execute and not _IMMUTABLE_IMAGE_RE.fullmatch(args.image):
        parser.error("--image must be pinned as repository@sha256:<64 lowercase hex>")
    if args.execute and args.scheduling_gate:
        parser.error(
            "--scheduling-gate is incompatible with the registered article matrix; "
            "it would contaminate default baselines"
        )
    if args.execute and args.environment_profile == "article-exact":
        if args.cooldown_seconds != DEFAULT_COOLDOWN_SECONDS:
            parser.error(
                f"article-exact requires --cooldown-seconds={DEFAULT_COOLDOWN_SECONDS}"
            )
        if args.cooldown_clean_polls != DEFAULT_COOLDOWN_CLEAN_POLLS:
            parser.error(
                "article-exact requires --cooldown-clean-polls="
                f"{DEFAULT_COOLDOWN_CLEAN_POLLS}"
            )
    specs = expand_plan(args.plan, include_adaptive=args.include_adaptive)
    expanded_plan = plan_document(
        specs, include_adaptive=args.include_adaptive, source_plan=args.plan
    )
    plan_created = _write_or_validate_plan_lock(
        args.plan_out, expanded_plan, require_existing=args.execute
    )
    selected = _select_specs(specs, args.start_at, args.limit)
    if args.execute and args.submission_workers < max(spec.jobs for spec in selected):
        parser.error("--submission-workers must be at least the largest selected burst")
    print(f"Validated {len(specs)}-run plan; selected {len(selected)} runs")

    if args.materialize and not args.execute:
        materialize_plan(
            selected,
            work_root=args.work_dir,
            image=args.image,
            namespace=args.namespace,
            overwrite=args.overwrite_workloads,
            scheduling_gate=args.scheduling_gate,
            image_pull_secrets=args.image_pull_secret,
        )
        print(
            f"Prepared and contract-validated {len(selected)} runs under {args.work_dir}"
        )
        return 0
    if not args.execute:
        action = "written" if plan_created else "validated"
        print(
            f"Dry plan {action} at {args.plan_out}; "
            "pass --materialize or --execute when ready"
        )
        return 0

    kubectl = Kubectl(args.kubectl, args.context)
    kubectl.run("version", "--client=true", "-o", "json", timeout=30)
    kubectl.run("get", "namespace", args.namespace, timeout=30)
    for secret_name in args.image_pull_secret:
        secret = kubectl.json(
            "get", "secret", secret_name, "-n", args.namespace, timeout=30
        )
        if secret.get("type") != "kubernetes.io/dockerconfigjson":
            raise ClusterRunError(
                f"image pull Secret {secret_name!r} has unexpected type {secret.get('type')!r}"
            )
    helm_environment = capture_and_verify_helm_release(
        executable=args.helm,
        release=args.helm_release,
        namespace=args.namespace,
        kube_context=args.context,
        target_node=args.target_node,
        require_metrics=args.include_adaptive,
    )
    scheduler_values = (
        (helm_environment.get("computed_values") or {}).get("scheduler") or {}
    )
    deployment = verify_scheduler_deployment(
        kubectl,
        namespace=args.namespace,
        deployment_name=args.scheduler_deployment,
        container_name=args.scheduler_container,
        target_node=args.target_node,
        results_template=args.scheduler_results_template,
        require_metrics=args.include_adaptive,
        scheduler_values=scheduler_values,
    )
    _ensure_clean_cluster(kubectl, args.namespace)
    minikube_profile = args.minikube_profile or args.context or "minikube"
    if args.environment_profile == "article-exact":
        minikube_environment = capture_minikube_environment(
            executable=args.minikube,
            profile=minikube_profile,
        )
    else:
        minikube_environment = {
            "profile": minikube_profile,
            "driver": None,
            "profile_status": "not-required-for-record-only",
        }
    prewarm_evidence = prewarm_trainer_image(
        kubectl,
        namespace=args.namespace,
        target_node=args.target_node,
        image=args.image,
        image_pull_secrets=args.image_pull_secret,
    )
    _ensure_clean_cluster(kubectl, args.namespace)
    cluster_environment = capture_cluster_environment(
        kubectl,
        namespace=args.namespace,
        target_node=args.target_node,
        deployment=deployment,
        scheduler_selector="app.kubernetes.io/component=scheduler",
    )
    cluster_environment["helm"] = helm_environment
    cluster_environment["minikube"] = minikube_environment
    cluster_environment["execution_controls"] = {
        "contract": execution_controls_contract(),
        "prewarm": prewarm_evidence,
    }
    try:
        cluster_environment["reproduction_policy"] = validate_article_environment(
            cluster_environment,
            experiment_namespace=args.namespace,
            profile=args.environment_profile,
        )
    except ArticleEnvironmentError as exc:
        cluster_environment["reproduction_policy"] = exc.evidence
        cluster_environment["plan_sha256"] = expanded_plan["plan_sha256"]
        cluster_environment["trainer_image"] = args.image
        _atomic_json(
            args.results_dir / "environment" / "cluster-rejected.json",
            cluster_environment,
        )
        raise
    scheduler_pods = (
        (cluster_environment.get("scheduler_deployment") or {}).get("pods") or []
    )
    if len(scheduler_pods) != 1:
        raise ClusterRunError("expected exactly one scheduler Pod in cluster snapshot")
    scheduler_containers = {
        item.get("name"): item for item in (scheduler_pods[0].get("containers") or [])
    }
    scheduler_status = scheduler_containers.get(args.scheduler_container) or {}
    if (
        scheduler_pods[0].get("phase") != "Running"
        or not scheduler_status.get("ready")
        or scheduler_status.get("restart_count") != 0
        or not re.search(
            r"sha256:[a-f0-9]{64}", str(scheduler_status.get("image_id") or "")
        )
    ):
        raise ClusterRunError(
            "scheduler Pod must be Running, Ready, restart-free, and expose imageID"
        )
    expected_scheduler_uid = str(scheduler_pods[0]["uid"])
    if args.include_adaptive:
        containers = (
            ((deployment.get("spec") or {}).get("template") or {})
            .get("spec", {})
            .get("containers", [])
        )
        scheduler_container = next(
            item for item in containers if item.get("name") == args.scheduler_container
        )
        scheduler_args = [str(value) for value in scheduler_container.get("args") or []]
        max_wait = float(_argument_value(scheduler_args, "--max-wait") or 0)
        max_age = float(_argument_value(scheduler_args, "--metrics-max-age") or 0)
        cluster_environment["adaptive_metrics_preflight"] = verify_adaptive_metrics_cadence(
            kubectl,
            target_node=args.target_node,
            max_wait_seconds=max_wait,
            metrics_max_age_seconds=max_age,
        )
    cluster_environment["plan_sha256"] = expanded_plan["plan_sha256"]
    cluster_environment["trainer_image"] = args.image
    _atomic_json(args.results_dir / "environment" / "cluster.json", cluster_environment)
    submit_manifest_directory, api_client = build_burst_submitter(
        kube_context=args.context,
        namespace=args.namespace,
        worker_ceiling=args.submission_workers,
        max_creation_spread_seconds=args.max_submission_spread_seconds,
    )
    completed = 0
    for spec in selected:
        result_path = args.results_dir / "runs" / f"{spec.run_id}.json"
        if args.resume and result_path.is_file():
            existing_result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_result_for_spec(
                existing_result,
                spec,
                plan_sha256=expanded_plan["plan_sha256"],
                target_node=args.target_node,
                require_article_environment=(
                    args.environment_profile == "article-exact"
                ),
            )
            print(f"[{spec.sequence}/{len(specs)}] resume: already complete {spec.run_id}")
            completed += 1
            continue
        manifest_directory = materialize_run(
            spec,
            work_root=args.work_dir,
            image=args.image,
            namespace=args.namespace,
            overwrite=args.overwrite_workloads,
            scheduling_gate=args.scheduling_gate,
            image_pull_secrets=args.image_pull_secret,
        )
        _ensure_clean_cluster(kubectl, args.namespace)
        cooldown_evidence = wait_for_cooldown(
            kubectl,
            namespace=args.namespace,
            target_node=args.target_node,
            scheduler_selector="app.kubernetes.io/component=scheduler",
            expected_scheduler_uid=expected_scheduler_uid,
            cooldown_seconds=args.cooldown_seconds,
            minimum_clean_polls=args.cooldown_clean_polls,
        )
        run_environment = capture_cluster_environment(
            kubectl,
            namespace=args.namespace,
            target_node=args.target_node,
            deployment=deployment,
            scheduler_selector="app.kubernetes.io/component=scheduler",
        )
        run_environment["helm"] = helm_environment
        run_environment["minikube"] = minikube_environment
        run_environment["execution_controls"] = {
            "contract": execution_controls_contract(),
            "prewarm": prewarm_evidence,
            "pre_run_cooldown": cooldown_evidence,
        }
        run_environment["reproduction_policy"] = validate_article_environment(
            run_environment,
            experiment_namespace=args.namespace,
            profile=args.environment_profile,
        )
        run_environment["plan_sha256"] = expanded_plan["plan_sha256"]
        run_environment["trainer_image"] = args.image
        print(f"[{spec.sequence}/{len(specs)}] running {spec.run_id}")
        execute_run(
            spec,
            kubectl=kubectl,
            namespace=args.namespace,
            manifest_directory=manifest_directory,
            results_root=args.results_dir,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            cleanup_on_failure=args.cleanup_on_failure,
            kube_context=args.context,
            trainer_image=args.image,
            image_pull_secrets=args.image_pull_secret,
            scheduler_deployment=args.scheduler_deployment,
            scheduler_container=args.scheduler_container,
            scheduler_results_template=args.scheduler_results_template,
            target_node=args.target_node,
            plan_sha256=expanded_plan["plan_sha256"],
            cluster_environment=run_environment,
            submit_manifest_directory=submit_manifest_directory,
            require_article_environment=(
                args.environment_profile == "article-exact"
            ),
        )
        completed += 1
    api_client.close()
    print(f"Completed and strictly collected {completed} selected runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
