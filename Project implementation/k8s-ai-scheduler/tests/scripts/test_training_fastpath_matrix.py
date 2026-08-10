from types import SimpleNamespace

import pytest

from scheduler.constants import ANNOTATION_MAP, RELEASE_GATE
from scripts import run_training_fastpath_pilot as pilot


def _job():
    values = {key: index + 1 for index, key in enumerate(ANNOTATION_MAP)}
    return SimpleNamespace(job_id="job-00", **values)


def _manifest(arm: str):
    return pilot.pod_manifest(
        arm=arm,
        repetition=0,
        run_id=f"run-{arm}",
        job=_job(),
        index=0,
        image="example.invalid/trainer:test",
        cpu="1",
        expected_jobs=12,
    )


def test_native_default_is_an_ungated_default_scheduler_baseline():
    manifest = _manifest("native-default")

    assert manifest["metadata"]["namespace"] == pilot.BASELINE_NAMESPACE
    assert manifest["spec"]["schedulerName"] == "default-scheduler"
    assert "schedulingGates" not in manifest["spec"]


def test_synchronized_baseline_keeps_the_launch_barrier():
    manifest = _manifest("baseline")

    assert manifest["metadata"]["namespace"] == pilot.BASELINE_NAMESPACE
    assert manifest["spec"]["schedulingGates"] == [{"name": RELEASE_GATE}]


def test_every_controlled_ablation_uses_the_controller_namespace_and_gate():
    for arm in set(pilot.ARM_SETTINGS) - pilot.UNCONTROLLED_ARMS:
        manifest = _manifest(arm)
        assert manifest["metadata"]["namespace"] == pilot.CUSTOM_NAMESPACE
        assert manifest["spec"]["schedulingGates"] == [{"name": RELEASE_GATE}]


def test_aggregate_prefers_the_native_default_reference():
    metrics = {
        "avg_jct_seconds": 10.0,
        "p95_jct_seconds": 12.0,
        "makespan_seconds": 14.0,
    }
    runs = [
        {"arm": "native-default", "repetition": 0, "metrics": metrics},
        {
            "arm": "kubeml",
            "repetition": 0,
            "metrics": {key: value / 2 for key, value in metrics.items()},
        },
    ]

    report = pilot.aggregate(runs, 1, ["native-default", "kubeml"])

    assert report["reference_arm"] == "native-default"
    assert report["mean_improvement_vs_baseline_pct"]["kubeml"] == {
        "avg_jct_seconds": 50.0,
        "makespan_seconds": 50.0,
        "p95_jct_seconds": 50.0,
    }


def test_resume_requires_the_exact_immutable_run_configuration(tmp_path):
    config = {"arms": ["native-default", "kubeml"], "repetitions": 30}
    pilot.write_json_atomic(tmp_path / "run-config.json", config)
    pilot.write_json_atomic(
        tmp_path / "runs.partial.json",
        [{"arm": "native-default", "repetition": 0}],
    )

    assert pilot.resume_runs(tmp_path, config) == [
        {"arm": "native-default", "repetition": 0}
    ]
    with pytest.raises(RuntimeError, match="does not match"):
        pilot.resume_runs(tmp_path, {**config, "repetitions": 20})


def test_resume_rejects_duplicate_completed_arms(tmp_path):
    config = {"arms": ["native-default", "kubeml"], "repetitions": 30}
    pilot.write_json_atomic(tmp_path / "run-config.json", config)
    duplicate = {"arm": "native-default", "repetition": 0}
    pilot.write_json_atomic(tmp_path / "runs.partial.json", [duplicate, duplicate])

    with pytest.raises(RuntimeError, match="duplicate"):
        pilot.resume_runs(tmp_path, config)
