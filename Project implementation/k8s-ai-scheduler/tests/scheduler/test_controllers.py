import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

from scheduler.burst import RunSettings
from scheduler.config import ConfigurationError, SchedulerConfig
from scheduler.constants import (
    ANNOTATION_MAP,
    EXPECTED_JOBS_ANNOTATION,
    PACING_MODE_ANNOTATION,
    RELEASE_GATE,
    RUN_ID_ANNOTATION,
    RUN_ID_LABEL,
)
from scheduler.custom_scheduler import ManualBindingSafetyError, MLAwareScheduler
from scheduler.gate_controller import ControllerStopping, SchedulingGateController


def node():
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="node-a", labels={"kubernetes.io/hostname": "node-a"}, deletion_timestamp=None
        ),
        spec=SimpleNamespace(unschedulable=False),
        status=SimpleNamespace(
            allocatable={"cpu": "4"},
            conditions=[SimpleNamespace(type="Ready", status="True")],
        ),
    )


def annotations(*, best=False):
    if best:
        values = dict(T="1", R="9", M="1", G="1", C="9", P="1")
    else:
        values = dict(T="9", R="1", M="9", G="9", C="1", P="9")
    result = {ANNOTATION_MAP[key]: value for key, value in values.items()}
    result[EXPECTED_JOBS_ANNOTATION] = "2"
    return result


def pod(name, *, best=False, gated=False):
    gates = [SimpleNamespace(name=RELEASE_GATE)] if gated else []
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            uid=f"uid-{name}",
            resource_version="1",
            creation_timestamp=None,
            labels={RUN_ID_LABEL: "run-1"},
            annotations=annotations(best=best),
        ),
        spec=SimpleNamespace(
            scheduler_name="ml-aware-scheduler" if not gated else "default-scheduler",
            node_name=None,
            scheduling_gates=gates,
            affinity=None,
            topology_spread_constraints=None,
            volumes=[],
            containers=[SimpleNamespace(name="train")],
            node_selector={},
        ),
        status=SimpleNamespace(phase="Pending"),
    )


class ManualCore:
    def __init__(self):
        self.binds = []

    def read_node(self, *_args, **_kwargs):
        return node()

    def create_namespaced_binding(self, namespace, body, **_kwargs):
        self.binds.append((namespace, body.metadata.name, body.target.name))


def make_manual(tmp_path, core=None):
    return MLAwareScheduler(
        "ml-aware-scheduler",
        "test",
        "none",
        0,
        0.85,
        5,
        0.1,
        False,
        str(tmp_path / "manual.json"),
        target_node="node-a",
        core_api=core or ManualCore(),
        custom_api=SimpleNamespace(),
        load_config=False,
        enable_health_server=False,
        api_retries=0,
    )


def test_manual_profile_requires_explicit_target_node(tmp_path):
    with pytest.raises(ConfigurationError, match="explicit target_node"):
        MLAwareScheduler(
            "ml-aware-scheduler",
            "test",
            "none",
            0,
            0.85,
            5,
            1,
            False,
            str(tmp_path / "out.json"),
            core_api=ManualCore(),
            custom_api=SimpleNamespace(),
            load_config=False,
            enable_health_server=False,
        )


def test_manual_process_run_is_ranked_incremental_and_complete(tmp_path):
    core = ManualCore()
    scheduler = make_manual(tmp_path, core)
    pods = [pod("worst"), pod("best", best=True)]
    scheduler._collect = lambda _settings: pods
    scheduler._wait_for_execution_start = lambda name, timeout=None: {
        "best": 101.0,
        "worst": 102.0,
    }[name]
    records = scheduler.process_run(RunSettings("run-1", 2, "none", 0, False), one_shot=True)
    assert [record.job_id for record in records] == ["best", "worst"]
    assert [item[1] for item in core.binds] == ["best", "worst"]
    document = json.loads((tmp_path / "manual.json").read_text())
    assert document["status"] == "completed"
    assert all(record["status"] == "execution_started" for record in document["records"])
    assert [event["event"] for event in document["events"]] == [
        "pacing_wait_started",
        "pacing_wait_completed",
    ]


def test_manual_profile_rejects_features_that_need_normal_scheduler(tmp_path):
    scheduler = make_manual(tmp_path)
    unsafe = pod("unsafe")
    unsafe.spec.affinity = SimpleNamespace()
    with pytest.raises(ManualBindingSafetyError, match="affinity"):
        scheduler._validate_pod_for_manual_binding(unsafe)

    scheduler.node.spec.taints = [
        SimpleNamespace(key="dedicated", value="gpu", effect="NoSchedule")
    ]
    safe_shape = pod("untolerated")
    safe_shape.spec.tolerations = []
    with pytest.raises(ManualBindingSafetyError, match="does not tolerate"):
        scheduler._validate_pod_for_manual_binding(safe_shape)
    safe_shape.spec.tolerations = [
        SimpleNamespace(
            key="dedicated", value="gpu", effect="NoSchedule", operator="Equal"
        )
    ]
    scheduler._validate_pod_for_manual_binding(safe_shape)


def test_binding_conflict_is_idempotent_only_on_expected_node(tmp_path):
    class ConflictCore(ManualCore):
        def create_namespaced_binding(self, *_args, **_kwargs):
            raise ApiException(status=409)

        def read_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(spec=SimpleNamespace(node_name="node-a"))

    scheduler = make_manual(tmp_path, ConflictCore())
    scheduler._bind_pod("job-a")


def test_unhandled_collection_failure_is_preserved(tmp_path):
    scheduler = make_manual(tmp_path)
    settings = RunSettings("bad-run", 2, "none", 0, False)
    scheduler._preserve_unhandled_failure(settings, RuntimeError("invalid burst"))
    document = json.loads((tmp_path / "manual-bad-run.json").read_text())
    assert document["status"] == "failed"
    assert document["error"] == "invalid burst"
    assert document["metadata"]["run_id"] == "bad-run"


class GateCore:
    def __init__(self):
        self.current = pod("job-a", gated=True)
        self.patches = []

    def list_namespaced_pod(self, *_args, **_kwargs):
        return SimpleNamespace(items=[])

    def read_namespaced_pod(self, *_args, **_kwargs):
        return self.current

    def patch_namespaced_pod(self, name, namespace, body, **kwargs):
        self.patches.append((name, namespace, body, kwargs))
        self.current.spec.scheduling_gates = []
        return self.current


def make_gate(tmp_path, core=None):
    config = SchedulerConfig(
        scheduler_name="default-scheduler",
        namespace="test",
        quiet_period=0.1,
        burst_timeout=2,
        poll_interval=0.1,
        api_timeout=1,
        api_retries=0,
        results_path=str(tmp_path / "gate.json"),
    )
    return SchedulingGateController(
        config,
        core_api=core or GateCore(),
        custom_api=SimpleNamespace(),
        load_config=False,
        enable_health_server=False,
    )


def test_gate_removal_uses_guarded_json_patch_and_is_idempotent(tmp_path):
    core = GateCore()
    controller = make_gate(tmp_path, core)
    controller._remove_gate("job-a")
    assert len(core.patches) == 1
    patch = core.patches[0][2]
    assert patch[0] == {
        "op": "test",
        "path": "/metadata/resourceVersion",
        "value": "1",
    }
    assert patch[-1]["path"] == "/spec/schedulingGates/0"
    assert core.patches[0][3]["_content_type"] == "application/json-patch+json"
    controller._remove_gate("job-a")
    assert len(core.patches) == 1


def test_gate_process_keeps_normal_scheduler_and_records_release(tmp_path):
    controller = make_gate(tmp_path)
    pods = [pod("worst", gated=True), pod("best", best=True, gated=True)]
    released = []
    controller._collect = lambda _settings: pods
    controller._remove_gate = released.append
    controller._wait_for_execution_start = (
        lambda name, **_kwargs: {"best": 101, "worst": 102}[name]
    )
    records = controller.process_run(
        RunSettings("run-1", 2, "none", 0, False), one_shot=True
    )
    assert released == ["best", "worst"]
    assert [record.status for record in records] == ["execution_started", "execution_started"]
    assert all(item.spec.scheduler_name == "default-scheduler" for item in pods)
    document = json.loads((tmp_path / "gate.json").read_text())
    assert document["metadata"]["profile"] == "production-scheduling-gate"
    assert document["status"] == "completed"


def test_gate_controller_resumes_partial_run_without_duplicate_release(tmp_path):
    pods = [pod("worst", gated=True), pod("best", best=True, gated=True)]
    by_name = {item.metadata.name: item for item in pods}
    first_releases = []
    first = make_gate(tmp_path)
    first._collect = lambda _settings: pods
    first._wait_for_execution_start = lambda name, **_kwargs: {
        "best": 101,
        "worst": 102,
    }[name]

    def interrupted_release(name):
        first_releases.append(name)
        if name == "worst":
            raise KeyboardInterrupt("simulated container restart")
        by_name[name].spec.scheduling_gates = []

    first._remove_gate = interrupted_release
    settings = RunSettings("run-1", 2, "none", 0, False)
    with pytest.raises(KeyboardInterrupt):
        first.process_run(settings, one_shot=True)
    assert first_releases == ["best", "worst"]
    assert by_name["best"].spec.scheduling_gates == []
    assert by_name["worst"].spec.scheduling_gates

    recovered_releases = []
    recovered = make_gate(tmp_path)
    recovered._collect = lambda _settings: pods
    recovered._wait_for_execution_start = lambda name, **_kwargs: {
        "best": 101,
        "worst": 102,
    }[name]

    def recovered_release(name):
        recovered_releases.append(name)
        by_name[name].spec.scheduling_gates = []

    recovered._remove_gate = recovered_release
    records = recovered.process_run(settings, one_shot=True)
    assert recovered_releases == ["worst"]
    assert [record.status for record in records] == [
        "execution_started",
        "execution_started",
    ]
    document = json.loads((tmp_path / "gate.json").read_text())
    assert document["schema_version"] == 3
    assert document["status"] == "completed"
    assert any(
        event["event"] == "controller_resumed" for event in document["events"]
    )
    assert {record["pod_uid"] for record in document["records"]} == {
        "uid-best",
        "uid-worst",
    }


def test_gate_failure_marks_readiness_degraded(tmp_path):
    controller = make_gate(tmp_path)
    pods = [pod("worst", gated=True), pod("best", best=True, gated=True)]
    controller._collect = lambda _settings: pods
    controller._remove_gate = lambda _name: None
    controller._wait_for_execution_start = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("marker unavailable")
    )
    with pytest.raises(RuntimeError, match="marker unavailable"):
        controller.process_run(
            RunSettings("run-1", 2, "none", 0, False), one_shot=True
        )
    _live, ready, reason = controller.health.snapshot()
    assert ready is False
    assert "run run-1 failed" in reason


def test_gate_shutdown_leaves_resumable_running_state(tmp_path):
    controller = make_gate(tmp_path)
    pods = [pod("worst", gated=True), pod("best", best=True, gated=True)]
    controller._collect = lambda _settings: pods
    controller.request_stop()
    with pytest.raises(ControllerStopping):
        controller.process_run(
            RunSettings("run-1", 2, "none", 0, False), one_shot=True
        )
    document = json.loads((tmp_path / "gate.json").read_text())
    assert document["status"] == "running"
    assert document["error"] is None
    assert document["events"][-1]["event"] == "controller_stopped"


def test_gate_discovery_isolates_malformed_run_and_continues(tmp_path):
    controller = make_gate(tmp_path)
    bad = pod("bad", gated=True)
    bad.metadata.labels[RUN_ID_LABEL] = "bad-run"
    bad.metadata.annotations[PACING_MODE_ANNOTATION] = "invalid"
    conflicting = pod("conflicting", gated=True)
    conflicting.metadata.labels[RUN_ID_LABEL] = "label-run"
    conflicting.metadata.annotations[RUN_ID_ANNOTATION] = "annotation-run"
    good = [pod("good-a", gated=True), pod("good-b", gated=True)]
    for item in good:
        item.metadata.labels[RUN_ID_LABEL] = "good-run"
    controller._list_run_pods = lambda: [conflicting, bad, *good]

    settings = controller._discover_settings(wait_forever=False)
    assert settings.run_id == "good-run"
    assert "bad-run" in controller._failed_runs
    _live, ready, reason = controller.health.snapshot()
    assert ready is False
    assert "bad-run" in reason
