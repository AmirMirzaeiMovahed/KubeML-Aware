import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.analyze import (
    _mean_ci95,
    analyze,
    compare_article_reference,
    load_article_reference,
)
from experiments.controls import execution_controls_contract
from experiments.run_cluster import (
    DEFAULT_PLAN,
    ClusterRunError,
    RunSpec,
    _canonical_sha256,
    _write_or_validate_plan_lock,
    expand_plan,
    materialize_run,
    plan_document,
    probe_scheduler_run,
    validate_result_for_spec,
    validate_scheduler_argument_contract,
    validate_scheduler_record,
)
from experiments.schema import make_result_document
from experiments.statistics import paired_effect_table
from k8s.work_model import WORK_MODEL_VERSION

TRAINER_IMAGE = "registry.example/ml-sim@sha256:" + "a" * 64


def _scheduler_values():
    return {
        "name": "ml-aware-scheduler",
        "targetNode": "node-a",
        "resultsPath": "/results/schedule-{run_id}.json",
        "quietPeriodSeconds": 1.5,
        "burstTimeoutSeconds": 60,
        "pollIntervalSeconds": 0.5,
        "executionTimeoutSeconds": 120,
        "apiTimeoutSeconds": 10,
        "apiRetries": 4,
        "cpuThreshold": 0.85,
        "adaptiveHysteresis": 0.05,
        "maxWaitSeconds": 90,
        "metricsMaxAgeSeconds": 30,
    }


def _runtime_metadata():
    values = _scheduler_values()
    return {
        "runtime_contract_version": "1.0",
        "quiet_period_seconds": values["quietPeriodSeconds"],
        "burst_timeout_seconds": values["burstTimeoutSeconds"],
        "poll_interval_seconds": values["pollIntervalSeconds"],
        "execution_timeout_seconds": values["executionTimeoutSeconds"],
        "api_timeout_seconds": values["apiTimeoutSeconds"],
        "api_retries": values["apiRetries"],
        "cpu_threshold": values["cpuThreshold"],
        "adaptive_hysteresis": values["adaptiveHysteresis"],
        "max_wait_seconds": values["maxWaitSeconds"],
        "metrics_max_age_seconds": values["metricsMaxAgeSeconds"],
    }


def _audit_snapshot():
    prewarm = {
        "schema_version": "1.0",
        "status": "passed",
        "pod_name": "prewarm",
        "pod_uid": "prewarm-uid",
        "target_node": "node-a",
        "requested_image": TRAINER_IMAGE,
        "runtime_image_id": "containerd://image@sha256:" + "d" * 64,
        "creation_timestamp": "2026-01-01T00:00:00Z",
        "start_timestamp": "2026-01-01T00:00:01Z",
        "finished_timestamp": "2026-01-01T00:00:02Z",
        "attestation": {
            "event": "PREWARM_ATTESTATION",
            "blas_runtime": {
                "expected_threads": 1,
                "libraries": [{"user_api": "blas", "num_threads": 1}],
            },
        },
    }
    prewarm["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            prewarm, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "target_node": {"name": "node-a"},
        "kubernetes_version": {},
        "helm": {
            "version": "test",
            "computed_values": {"scheduler": _scheduler_values()},
        },
        "reproduction_policy": {
            "profile": "article-exact",
            "article_claim_eligible": True,
            "errors": [],
        },
        "minikube": {
            "profile": "paper",
            "driver": "docker",
            "profile_status": "Running",
        },
        "execution_controls": {
            "contract": execution_controls_contract(),
            "prewarm": prewarm,
            "pre_run_cooldown": {
                "elapsed_seconds": 30.0,
                "clean_polls": 3,
                "workload_pods_observed": 0,
                "scheduler_continuity": True,
                "node_pressure_clear": True,
            },
        },
    }


def test_registered_plan_has_exact_70_and_optional_90_runs():
    base = expand_plan(DEFAULT_PLAN, include_adaptive=False)
    extended = expand_plan(DEFAULT_PLAN, include_adaptive=True)
    assert len(base) == 70
    assert len(extended) == 90
    assert len({run.run_id for run in extended}) == 90
    assert sum(run.group == "pacing" for run in base) == 25
    assert sum(run.group == "main" for run in base) == 45


def test_paired_configs_materialize_identical_workload_features(tmp_path: Path):
    specs = expand_plan(DEFAULT_PLAN)
    default = next(
        run
        for run in specs
        if run.group == "pacing" and run.repetition == 0 and run.config == "default"
    )
    custom = next(
        run
        for run in specs
        if run.group == "pacing"
        and run.repetition == 0
        and run.config == "custom-baseline"
    )
    assert (default.scenario, default.repetition, default.seed) == (
        custom.scenario,
        custom.repetition,
        custom.seed,
    )
    materialize_run(
        default,
        work_root=tmp_path,
        image="ml-sim:v1",
        namespace="experiment",
        overwrite=False,
    )
    materialize_run(
        custom,
        work_root=tmp_path,
        image="ml-sim:v1",
        namespace="experiment",
        overwrite=False,
    )
    import json

    first = json.loads(
        (tmp_path / default.run_id / "jobs.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (tmp_path / custom.run_id / "jobs.json").read_text(encoding="utf-8")
    )
    assert first["jobs"] == second["jobs"]


def test_reviewed_materialization_is_reused_only_when_exact(tmp_path: Path):
    spec = next(run for run in expand_plan(DEFAULT_PLAN) if run.jobs == 12)
    first = materialize_run(
        spec,
        work_root=tmp_path,
        image="registry.example/ml-sim@sha256:" + "a" * 64,
        namespace="experiment",
        overwrite=False,
        image_pull_secrets=["registry-credentials"],
    )
    second = materialize_run(
        spec,
        work_root=tmp_path,
        image="registry.example/ml-sim@sha256:" + "a" * 64,
        namespace="experiment",
        overwrite=False,
        image_pull_secrets=["registry-credentials"],
    )
    assert second == first
    manifest = next(first.glob("*.yaml"))
    payload = manifest.read_text(encoding="utf-8").replace(
        "ml.scheduler/config:", "ml.scheduler/tampered-config:", 1
    )
    manifest.write_text(payload, encoding="utf-8")
    with pytest.raises(ClusterRunError, match="drifted"):
        materialize_run(
            spec,
            work_root=tmp_path,
            image="registry.example/ml-sim@sha256:" + "a" * 64,
            namespace="experiment",
            overwrite=False,
            image_pull_secrets=["registry-credentials"],
        )


def test_plan_lock_cannot_be_silently_replaced(tmp_path: Path):
    specs = expand_plan(DEFAULT_PLAN)
    document = plan_document(specs, include_adaptive=False)
    path = tmp_path / "plan.json"
    _write_or_validate_plan_lock(path, document, require_existing=False)
    _write_or_validate_plan_lock(path, document, require_existing=True)
    changed = {**document, "run_count": 999}
    with pytest.raises(ClusterRunError, match="plan lock differs"):
        _write_or_validate_plan_lock(path, changed, require_existing=True)


def test_confidence_interval_collapses_for_one_value():
    assert _mean_ci95([3.0]) == (3.0, 3.0, 3.0)


def test_confidence_interval_surrounds_mean():
    mean, lower, upper = _mean_ci95([1, 2, 3, 4, 5])
    assert mean == 3.0
    assert lower < mean < upper


def test_article_effects_use_correct_paired_references_and_confidence_intervals():
    rows = []
    for repetition, (default, intended, reversed_jct) in enumerate(
        [(10.0, 8.0, 12.0), (20.0, 10.0, 15.0), (40.0, 20.0, 30.0)]
    ):
        for config, avg_jct in (
            ("default", default),
            ("custom-baseline", intended),
            ("reversed", reversed_jct),
        ):
            rows.append(
                {
                    "scenario": "48-normal",
                    "config": config,
                    "rep": repetition,
                    "seed": 2000 + repetition,
                    "avg_jct": avg_jct,
                    "makespan": avg_jct * 2,
                }
            )
    paired = paired_effect_table(pd.DataFrame(rows)).set_index("config")
    assert len(paired) == 2
    improvement = paired.loc["custom-baseline"]
    assert improvement.reference_config == "default"
    assert improvement.effect_kind == "improvement_vs_default"
    assert improvement.pairs == 3
    assert improvement.avg_jct_effect_mean == pytest.approx(32 / 3)
    assert improvement.avg_jct_effect_pct_mean == pytest.approx(40.0)
    assert (
        improvement.avg_jct_effect_pct_ci95_low
        < improvement.avg_jct_effect_pct_mean
        < improvement.avg_jct_effect_pct_ci95_high
    )
    degradation = paired.loc["reversed"]
    assert degradation.reference_config == "custom-baseline"
    assert degradation.effect_kind == "degradation_vs_intended"
    assert degradation.avg_jct_effect_mean == pytest.approx(19 / 3)
    assert degradation.avg_jct_effect_pct_mean == pytest.approx(50.0)


def test_paired_effects_reject_missing_reference_pair():
    summary = pd.DataFrame(
        [
            {
                "scenario": "12-normal",
                "config": "default",
                "rep": 0,
                "seed": 1,
                "avg_jct": 10,
                "makespan": 12,
            },
            {
                "scenario": "12-normal",
                "config": "custom-baseline",
                "rep": 0,
                "seed": 1,
                "avg_jct": 8,
                "makespan": 10,
            },
            {
                "scenario": "12-normal",
                "config": "reversed",
                "rep": 1,
                "seed": 2,
                "avg_jct": 9,
                "makespan": 11,
            },
        ]
    )
    with pytest.raises(ValueError, match="unpaired runs"):
        paired_effect_table(summary)


def test_published_article_reference_preserves_table_and_ablation_semantics():
    reference = load_article_reference()
    table = reference["results"]["pacing"]["48-half-pacing"]
    assert table["default"]["metrics"] == {
        "avg_ilt": 0.67,
        "avg_jct": 230.19,
        "makespan": 443.0,
    }
    assert table["custom-baseline"]["metrics"]["avg_jct"] == 165.79

    summary = pd.DataFrame(
        [
            {
                "scenario": "12-normal",
                "config": "default",
                "avg_jct": 234.0,
                "tail_jct_p95": 438.0,
                "max_jct": 451.0,
                "min_jct": 16.0,
                "makespan": 451.0,
            },
            {
                "scenario": "12-normal",
                "config": "custom-baseline",
                "avg_jct": 178.0,
                "tail_jct_p95": 401.0,
                "max_jct": 412.0,
                "min_jct": 7.0,
                "makespan": 412.0,
            },
            {
                "scenario": "12-normal",
                "config": "reversed",
                "avg_jct": 178.0 * 1.49,
                "tail_jct_p95": 450.0,
                "max_jct": 470.0,
                "min_jct": 10.0,
                "makespan": 500.0,
            },
        ]
    )
    comparison = compare_article_reference(summary, reference)
    degradation = comparison[
        (comparison.row_type == "published_effect")
        & (comparison.effect_kind == "degradation_vs_intended")
    ].iloc[0]
    assert degradation.reference_config == "custom-baseline"
    assert degradation.observed_value == pytest.approx(49.0)
    assert degradation.absolute_delta == pytest.approx(0.0)


def test_complete_70_run_documents_analyze_end_to_end(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    specs = expand_plan(DEFAULT_PLAN)
    plan_hash = plan_document(specs, include_adaptive=False)["plan_sha256"]
    for spec in specs:
        jobs = []
        for index in range(spec.jobs):
            start = index * 0.01
            end = start + 1.0
            jobs.append(
                {
                    "job_id": f"job-{index}",
                    "category": "test",
                    "features": {"T": 1, "R": 1, "M": 1, "G": 1, "C": 1, "P": 1},
                    "rank": 0.5,
                    "submission_time": 0.0,
                    "execution_start_time": start,
                    "completion_time": end,
                    "jct_s": end,
                    "status": "completed",
                    "node_name": "node-a",
                    "error": None,
                    "trainer_evidence": {
                        "work_model_version": WORK_MODEL_VERSION,
                        "blas_threads": 1,
                        "blas_library_count": 1,
                    },
                }
            )
        submission = {
            "schema_version": "1.0",
            "mode": "concurrent-client-barrier",
            "seed": spec.seed,
            "expected_jobs": spec.jobs,
            "worker_count": spec.jobs,
            "configured_worker_ceiling": 64,
            "burst_started_at": 0.0,
            "burst_completed_at": 0.1,
            "client_request_spread_seconds": 0.0,
            "server_creation_spread_seconds": 0.0,
            "max_creation_spread_seconds": 5.0,
            "randomized_job_order": [job["job_id"] for job in jobs],
            "jobs": [
                {
                    "submission_index": index,
                    "job_id": job["job_id"],
                    "client_requested_at": 0.0,
                    "client_requested_offset_seconds": 0.0,
                    "client_completed_at": 0.1,
                    "client_duration_seconds": 0.1,
                    "status": "created",
                    "error": None,
                    "uid": f"uid-{index}",
                    "server_creation_timestamp": 0.0,
                }
                for index, job in enumerate(jobs)
            ],
        }
        cluster_snapshot = _audit_snapshot()
        document = make_result_document(
            run_id=spec.run_id,
            scenario=spec.scenario,
            config=spec.config,
            repetition=spec.repetition,
            seed=spec.seed,
            expected_jobs=spec.jobs,
            jobs=jobs,
            source="kubernetes",
            environment={
                "orchestration": {
                    "plan_sha256": plan_hash,
                    "target_node": "node-a",
                    "trainer_image": TRAINER_IMAGE,
                    "kubectl_context": "test-context",
                    "scheduler_deployment": "scheduler",
                    "artifact_sha256": {
                        "jobs.json": "b" * 64,
                        **{f"pods/{job['job_id']}.yaml": "c" * 64 for job in jobs},
                    },
                    "cluster_snapshot_sha256": _canonical_sha256(cluster_snapshot),
                    "submission_sha256": _canonical_sha256(submission),
                },
                "cluster_snapshot": cluster_snapshot,
                "kubernetes": {
                    "workload_pods": [
                        {"name": job["job_id"], "node_name": "node-a"} for job in jobs
                    ]
                },
                "submission": submission,
            },
        )
        if spec.scheduler_name != "default-scheduler":
            events = []
            for index, job in enumerate(jobs[:-1]):
                started = 100.0 + index * 10
                events.extend(
                    [
                        {
                            "event": "pacing_wait_started",
                            "after_job_id": job["job_id"],
                            "before_job_id": jobs[index + 1]["job_id"],
                            "mode": spec.pacing_mode,
                            "fixed_delay_seconds": spec.fixed_delay_seconds,
                            "timestamp": started,
                        },
                        {
                            "event": "pacing_wait_completed",
                            "after_job_id": job["job_id"],
                            "before_job_id": jobs[index + 1]["job_id"],
                            "mode": spec.pacing_mode,
                            "timestamp": started + spec.fixed_delay_seconds,
                        },
                    ]
                )
            document["scheduler_record"] = {
                "schema_version": 3,
                "status": "completed",
                "error": None,
                "metadata": {
                    "profile": "article-manual-bind",
                    "run_id": spec.run_id,
                    "expected_count": spec.jobs,
                    "scheduler_name": spec.scheduler_name,
                    "target_node": "node-a",
                    "pacing_mode": spec.pacing_mode,
                    "fixed_delay": spec.fixed_delay_seconds,
                    "reverse": spec.reverse,
                    **_runtime_metadata(),
                },
                "records": [
                    {
                        "job_id": job["job_id"],
                        "pod_uid": f"uid-{index}",
                        "order": index + 1,
                        "rank": job["rank"],
                        "status": "execution_started",
                        "bind_time": job["execution_start_time"] - 0.001,
                        "release_time": None,
                        "exec_start_time": job["execution_start_time"],
                        "error": None,
                    }
                    for index, job in enumerate(jobs)
                ],
                "events": events,
            }
        (runs / f"{spec.run_id}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    report = analyze(runs_dir=runs, output_dir=tmp_path / "analysis", make_plots=False)
    assert report["run_count"] == 70
    assert report["strictly_complete"] is True
    assert report["pairing_keys"] == ["scenario", "repetition", "seed"]
    assert report["paired_effects"]
    assert report["article_reference"]["acceptance_threshold"] is None
    assert report["article_reference"]["comparison_rows"] == 54
    assert (tmp_path / "analysis" / "aggregate_metrics.csv").is_file()
    assert (tmp_path / "analysis" / "paired_effects.csv").is_file()
    assert (tmp_path / "analysis" / "article_reference_comparison.csv").is_file()


def _one_job_spec(*, reverse=False, pacing_mode="none", fixed_delay=0.0):
    return RunSpec(
        sequence=1,
        group="test",
        run_id="run-1",
        scenario="test",
        jobs=1,
        load="normal",
        repetition=0,
        seed=7,
        config="reversed" if reverse else "custom-baseline",
        scheduler_name="ml-aware-scheduler",
        manifest_set="custom",
        pacing_mode=pacing_mode,
        fixed_delay_seconds=fixed_delay,
        reverse=reverse,
    )


def test_scheduler_evidence_is_validated_against_collected_jobs():
    spec = _one_job_spec()
    result = make_result_document(
        run_id=spec.run_id,
        scenario=spec.scenario,
        config=spec.config,
        repetition=spec.repetition,
        seed=spec.seed,
        expected_jobs=1,
        source="kubernetes",
        environment={
            "cluster_snapshot": {
                "helm": {"computed_values": {"scheduler": _scheduler_values()}}
            }
        },
        jobs=[
            {
                "job_id": "job-a",
                "category": "test",
                "features": {name: 1 for name in ("T", "R", "M", "G", "C", "P")},
                "rank": 0.5,
                "submission_time": 99.0,
                "execution_start_time": 101.0,
                "completion_time": 102.0,
                "jct_s": 3.0,
                "status": "completed",
                "node_name": "node-a",
                "error": None,
            }
        ],
    )
    schedule = {
        "schema_version": 3,
        "status": "completed",
        "error": None,
        "metadata": {
            "profile": "article-manual-bind",
            "run_id": spec.run_id,
            "expected_count": 1,
            "scheduler_name": spec.scheduler_name,
            "target_node": "node-a",
            "pacing_mode": "none",
            "fixed_delay": 0.0,
            "reverse": False,
            **_runtime_metadata(),
        },
        "records": [
            {
                "job_id": "job-a",
                "pod_uid": "uid-job-a",
                "order": 1,
                "rank": 0.5,
                "status": "execution_started",
                "bind_time": 100.0,
                "release_time": None,
                "exec_start_time": 101.0,
                "error": None,
            }
        ],
        "events": [],
    }
    validate_scheduler_record(spec, schedule, result, target_node="node-a")
    schedule["records"][0]["rank"] = 0.1
    with pytest.raises(ClusterRunError, match="rank differs"):
        validate_scheduler_record(spec, schedule, result, target_node="node-a")


def test_live_scheduler_arguments_must_match_computed_helm_values():
    values = _scheduler_values()
    arguments = [
        "--scheduler-name",
        values["name"],
        "--target-node",
        values["targetNode"],
        "--results",
        values["resultsPath"],
        "--quiet-period",
        str(values["quietPeriodSeconds"]),
        "--burst-timeout",
        str(values["burstTimeoutSeconds"]),
        "--poll-interval",
        str(values["pollIntervalSeconds"]),
        "--execution-timeout",
        str(values["executionTimeoutSeconds"]),
        "--api-timeout",
        str(values["apiTimeoutSeconds"]),
        "--api-retries",
        str(values["apiRetries"]),
        "--cpu-threshold",
        str(values["cpuThreshold"]),
        "--adaptive-hysteresis",
        str(values["adaptiveHysteresis"]),
        "--max-wait",
        str(values["maxWaitSeconds"]),
        "--metrics-max-age",
        str(values["metricsMaxAgeSeconds"]),
    ]
    validate_scheduler_argument_contract(
        arguments,
        values,
        target_node="node-a",
        results_template="/results/schedule-{run_id}.json",
    )
    arguments[arguments.index("--quiet-period") + 1] = "9"
    with pytest.raises(ClusterRunError, match="--quiet-period"):
        validate_scheduler_argument_contract(
            arguments,
            values,
            target_node="node-a",
            results_template="/results/schedule-{run_id}.json",
        )


def test_resume_validation_rejects_result_from_different_plan():
    spec = RunSpec(
        sequence=1,
        group="test",
        run_id="run-default",
        scenario="test",
        jobs=1,
        load="normal",
        repetition=0,
        seed=7,
        config="default",
        scheduler_name="default-scheduler",
        manifest_set="default",
        pacing_mode="none",
        fixed_delay_seconds=0.0,
        reverse=False,
    )
    result = make_result_document(
        run_id=spec.run_id,
        scenario=spec.scenario,
        config=spec.config,
        repetition=spec.repetition,
        seed=spec.seed,
        expected_jobs=1,
        source="kubernetes",
        environment={
            "orchestration": {"plan_sha256": "wrong", "target_node": "node-a"}
        },
        jobs=[
            {
                "job_id": "job-a",
                "category": "test",
                "features": {name: 1 for name in ("T", "R", "M", "G", "C", "P")},
                "rank": 0.5,
                "submission_time": 1.0,
                "execution_start_time": 2.0,
                "completion_time": 3.0,
                "jct_s": 2.0,
                "status": "completed",
                "node_name": "node-a",
                "error": None,
            }
        ],
    )
    with pytest.raises(ClusterRunError, match="plan_sha256"):
        validate_result_for_spec(
            result, spec, plan_sha256="expected", target_node="node-a"
        )


def test_scheduler_failure_probe_aborts_before_workload_timeout():
    spec = _one_job_spec()

    class FakeKubectl:
        def pod_json(self, namespace, selector):
            return {
                "items": [
                    {
                        "metadata": {"uid": "scheduler-uid"},
                        "status": {
                            "phase": "Running",
                            "containerStatuses": [
                                {
                                    "name": "scheduler",
                                    "ready": True,
                                    "restartCount": 0,
                                }
                            ],
                        },
                    }
                ]
            }

        def run(self, *args, **kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout='{"status":"failed","error":"bad annotation"}',
            )

    with pytest.raises(ClusterRunError, match="bad annotation"):
        probe_scheduler_run(
            FakeKubectl(),
            namespace="test",
            deployment_name="scheduler",
            container_name="scheduler",
            scheduler_selector="component=scheduler",
            expected_pod_uid="scheduler-uid",
            results_template="/results/schedule-{run_id}.json",
            spec=spec,
        )
