import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import results.metrics_collector as metrics_collector
from experiments.schema import IncompleteRunError, percentile
from k8s.work_model import WORK_MODEL_VERSION, estimate_work
from results.metrics_collector import collect, parse_log_events
from workload.generate_workload import deterministic_job_seed


def _pod(name="light-000-12345678", phase="Succeeded"):
    annotations = {
        "ml.scheduler/estimated-training-time": "10",
        "ml.scheduler/loss-reduction-rate": "0.1",
        "ml.scheduler/matrix-size": "64",
        "ml.scheduler/gradient-update-size": "2",
        "ml.scheduler/checkpoint-interval": "50",
        "ml.scheduler/model-partitions": "1",
        "ml.scheduler/run-id": "run-1",
        "ml.scheduler/pacing-mode": "none",
        "ml.scheduler/fixed-delay-seconds": "0.0",
        "ml.scheduler/reverse": "false",
        "ml.scheduler/expected-jobs": "1",
        "ml.scheduler/seed": "1",
        "ml.scheduler/load-profile": "normal",
        "ml.scheduler/work-model-version": WORK_MODEL_VERSION,
        "ml.scheduler/blas-threads": "1",
    }
    labels = {
        "ml.scheduler/run-id": "run-1",
        "ml.scheduler/scenario": "12-normal",
        "ml.scheduler/config": "default",
        "ml.scheduler/repetition": "0",
    }
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            annotations=annotations,
            labels=labels,
            creation_timestamp=datetime.fromtimestamp(100, tz=timezone.utc),
        ),
        spec=SimpleNamespace(
            node_name="node-a",
            scheduler_name="default-scheduler",
            scheduling_gates=[],
            image_pull_secrets=[SimpleNamespace(name="registry-credentials")],
            containers=[
                SimpleNamespace(
                    name="train",
                    image="registry.example/ml-sim@sha256:" + "a" * 64,
                    env=[
                        SimpleNamespace(
                            name="JOB_ID",
                            value=None,
                            value_from=SimpleNamespace(
                                field_ref=SimpleNamespace(field_path="metadata.name")
                            ),
                        ),
                        SimpleNamespace(
                            name="JOB_SEED",
                            value=str(deterministic_job_seed(1, name)),
                            value_from=None,
                        ),
                        SimpleNamespace(
                            name="ML_WORK_MODEL_VERSION",
                            value=WORK_MODEL_VERSION,
                            value_from=None,
                        ),
                        SimpleNamespace(
                            name="ML_NUM_THREADS",
                            value="1",
                            value_from=None,
                        ),
                        *[
                            SimpleNamespace(
                                name=f"JOB_{feature}",
                                value=annotations[key],
                                value_from=None,
                            )
                            for feature, key in {
                                "T": "ml.scheduler/estimated-training-time",
                                "R": "ml.scheduler/loss-reduction-rate",
                                "M": "ml.scheduler/matrix-size",
                                "G": "ml.scheduler/gradient-update-size",
                                "C": "ml.scheduler/checkpoint-interval",
                                "P": "ml.scheduler/model-partitions",
                            }.items()
                        ],
                    ],
                )
            ],
        ),
        status=SimpleNamespace(
            phase=phase,
            reason=None,
            container_statuses=[
                SimpleNamespace(
                    name="train",
                    image="registry.example/ml-sim@sha256:" + "a" * 64,
                    image_id="containerd://registry.example/ml-sim@sha256:" + "b" * 64,
                    restart_count=0,
                    state=SimpleNamespace(terminated=None, waiting=None),
                )
            ],
        ),
    )


def _success_logs(pod, *, started=101.0, completed=105.0):
    annotations = pod.metadata.annotations
    estimate = estimate_work(
        R=float(annotations["ml.scheduler/loss-reduction-rate"]),
        M=int(annotations["ml.scheduler/matrix-size"]),
        G=float(annotations["ml.scheduler/gradient-update-size"]),
        C=int(annotations["ml.scheduler/checkpoint-interval"]),
        P=int(annotations["ml.scheduler/model-partitions"]),
    )
    completion = {
        "event": "EXECUTION_COMPLETED",
        "timestamp": completed,
        "job_id": pod.metadata.name,
        "steps": estimate.planned_steps,
        "final_loss": 0.01,
        "termination_reason": estimate.termination_reason,
        "work_model_version": WORK_MODEL_VERSION,
        "step_budget": estimate.step_budget,
        "convergence_steps": estimate.convergence_steps,
        "gradient_bytes": estimate.gradient_bytes,
        "checkpoint_bytes": estimate.checkpoint_bytes,
        "checkpoint_count": estimate.checkpoint_count,
        "checkpoint_seconds": 0.01,
        "duration_seconds": completed - started,
        "blas_threads": 1,
        "blas_library_count": 1,
    }
    return "\n".join(
        [
            json.dumps(
                {
                    "event": "INITIALIZATION_COMPLETED",
                    "timestamp": started - 0.1,
                    "job_id": pod.metadata.name,
                    "work_model": estimate.to_dict(),
                    "blas_runtime": {
                        "expected_threads": 1,
                        "libraries": [{"user_api": "blas", "num_threads": 1}],
                    },
                }
            ),
            json.dumps(
                {
                    "event": "EXECUTION_STARTED",
                    "timestamp": started,
                    "job_id": pod.metadata.name,
                }
            ),
            json.dumps(completion),
        ]
    )


class FakeCoreApi:
    def __init__(self, pods, logs):
        self.pods = pods
        self.logs = logs

    def list_namespaced_pod(self, namespace, label_selector):
        return SimpleNamespace(items=self.pods)

    def read_namespaced_pod_log(self, name, namespace, container):
        return self.logs[name]

    def list_node(self):
        return SimpleNamespace(items=[])


def test_standard_linear_p95():
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)


def test_structured_and_legacy_markers_are_parsed():
    logs = "\n".join(
        [
            json.dumps(
                {"event": "EXECUTION_STARTED", "timestamp": 101.0, "job_id": "a"}
            ),
            "[105.500000] EXECUTION_COMPLETED steps=2",
        ]
    )
    events = parse_log_events(logs)
    assert events["EXECUTION_STARTED"]["timestamp"] == 101.0
    assert events["EXECUTION_COMPLETED"]["timestamp"] == 105.5


def test_strict_collection_returns_common_schema():
    pod = _pod()
    logs = {pod.metadata.name: _success_logs(pod)}
    document = collect(
        "test",
        "ml.scheduler/run-id=run-1",
        expected_count=1,
        run_id="run-1",
        scenario="12-normal",
        config_name="default",
        repetition=0,
        seed=1,
        core_api=FakeCoreApi([pod], logs),
        environment={"test": True},
    )
    assert document["run"]["status"] == "completed"
    assert document["summary"]["avg_jct"] == 5.0
    assert document["jobs"][0]["rank"] == pytest.approx(0.5)
    assert (
        document["jobs"][0]["trainer_evidence"]["work_model_version"]
        == WORK_MODEL_VERSION
    )


def test_strict_collection_validates_full_pod_contract():
    pod = _pod()
    logs = {pod.metadata.name: _success_logs(pod)}
    contract = {
        "run_id": "run-1",
        "scheduler_name": "default-scheduler",
        "pacing_mode": "none",
        "fixed_delay_seconds": 0.0,
        "reverse": False,
        "expected_jobs": 1,
        "seed": 1,
        "load_profile": "normal",
        "work_model_version": WORK_MODEL_VERSION,
        "scheduling_gate": None,
        "image": "registry.example/ml-sim@sha256:" + "a" * 64,
        "image_pull_secrets": ["registry-credentials"],
    }
    collect(
        "test",
        "ml.scheduler/run-id=run-1",
        expected_count=1,
        run_id="run-1",
        scenario="12-normal",
        config_name="default",
        repetition=0,
        seed=1,
        core_api=FakeCoreApi([pod], logs),
        expected_pod_contract=contract,
    )
    pod.spec.scheduling_gates = [SimpleNamespace(name="ml.scheduler/release")]
    with pytest.raises(IncompleteRunError) as caught:
        collect(
            "test",
            "ml.scheduler/run-id=run-1",
            expected_count=1,
            run_id="run-1",
            scenario="12-normal",
            config_name="default",
            repetition=0,
            seed=1,
            core_api=FakeCoreApi([pod], logs),
            expected_pod_contract=contract,
        )
    assert "scheduling gates" in caught.value.document["jobs"][0]["error"]


def test_stale_trainer_model_evidence_fails_closed():
    pod = _pod()
    logs = _success_logs(pod).replace(WORK_MODEL_VERSION, "stale-model")
    with pytest.raises(IncompleteRunError) as caught:
        collect(
            "test",
            expected_count=1,
            run_id="run-1",
            scenario="12-normal",
            config_name="default",
            repetition=0,
            seed=1,
            core_api=FakeCoreApi([pod], {pod.metadata.name: logs}),
        )
    assert "work model version mismatch" in caught.value.document["jobs"][0]["error"]


def test_explicit_kube_context_is_forwarded(monkeypatch):
    called = {}
    fake = SimpleNamespace(
        load_kube_config=lambda **kwargs: called.update(kwargs),
        load_incluster_config=lambda: None,
    )
    monkeypatch.setattr(metrics_collector, "config", fake)
    metrics_collector._load_kubernetes_configuration(
        "kubeconfig", context="experiment-context"
    )
    assert called == {"context": "experiment-context"}


def test_missing_marker_fails_instead_of_dropping_pod():
    pod = _pod()
    api = FakeCoreApi([pod], {pod.metadata.name: "{}"})
    with pytest.raises(IncompleteRunError) as caught:
        collect(
            "test",
            expected_count=1,
            run_id="run-1",
            scenario="12-normal",
            config_name="default",
            repetition=0,
            seed=1,
            core_api=api,
        )
    document = caught.value.document
    assert len(document["jobs"]) == 1
    assert document["jobs"][0]["status"] == "failed"
    assert "marker missing" in document["jobs"][0]["error"]


def test_count_mismatch_is_a_run_failure():
    with pytest.raises(IncompleteRunError) as caught:
        collect(
            "test",
            expected_count=2,
            run_id="run-1",
            scenario="12-normal",
            config_name="default",
            repetition=0,
            seed=1,
            core_api=FakeCoreApi([], {}),
        )
    assert caught.value.document["failures"][0]["code"] == "POD_COUNT_MISMATCH"
