import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

from scheduler.execution import (
    ExecutionStartError,
    execution_container_for_pod,
    parse_execution_marker,
    wait_for_execution_start,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def running_pod():
    return SimpleNamespace(
        status=SimpleNamespace(phase="Running", reason=None, message=None, container_statuses=[])
    )


def completed_pod(exit_code=0, reason="Completed"):
    return SimpleNamespace(
        status=SimpleNamespace(
            phase="Succeeded" if exit_code == 0 else "Running",
            reason=None,
            message=None,
            container_statuses=[
                SimpleNamespace(
                    name="train",
                    state=SimpleNamespace(
                        terminated=SimpleNamespace(exit_code=exit_code, reason=reason)
                    ),
                )
            ],
        )
    )


def test_execution_container_is_explicit_for_ambiguous_pods():
    pod = SimpleNamespace(
        metadata=SimpleNamespace(annotations={}),
        spec=SimpleNamespace(
            containers=[SimpleNamespace(name="model"), SimpleNamespace(name="sidecar")]
        ),
    )
    with pytest.raises(ExecutionStartError, match="execution-container"):
        execution_container_for_pod(pod)
    pod.metadata.annotations["ml.scheduler/execution-container"] = "model"
    assert execution_container_for_pod(pod) == "model"
    pod.metadata.annotations["ml.scheduler/execution-container"] = "missing"
    with pytest.raises(ExecutionStartError, match="does not name"):
        execution_container_for_pod(pod)


def test_parser_accepts_json_and_legacy_markers_only():
    marker = json.dumps(
        {"event": "EXECUTION_STARTED", "timestamp": 123.5, "job_id": "job-a"}
    )
    assert parse_execution_marker(marker, expected_job_id="job-a") == 123.5
    assert parse_execution_marker(marker, expected_job_id="other") is None
    assert parse_execution_marker("[99.25] EXECUTION_STARTED") == 99.25
    assert parse_execution_marker("prefix EXECUTION_STARTED 99") is None
    assert parse_execution_marker('{"event":"EXECUTION_STARTED","timestamp":"nan"}') is None


def test_wait_polls_until_valid_marker():
    clock = FakeClock()
    logs = iter(["initializing", "[123.0] EXECUTION_STARTED"])
    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: next(logs),
        read_namespaced_pod_status=lambda *_args, **_kwargs: running_pod(),
    )
    assert wait_for_execution_start(
        core,
        "job-a",
        "test",
        timeout=2,
        api_timeout_seconds=1,
        api_retries=0,
        poll_interval=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) == 123.0
    assert clock.value == pytest.approx(0.25)


def test_wait_allows_final_log_flush_after_clean_completion():
    clock = FakeClock()
    logs = iter(["initializing", '[123.0] EXECUTION_STARTED'])
    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: next(logs),
        read_namespaced_pod_status=lambda *_args, **_kwargs: completed_pod(),
    )
    assert wait_for_execution_start(
        core,
        "job-a",
        "test",
        timeout=2,
        api_timeout_seconds=1,
        api_retries=0,
        poll_interval=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) == 123.0
    assert clock.value == pytest.approx(0.25)


def test_nonzero_termination_still_fails_immediately():
    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: "no marker",
        read_namespaced_pod_status=lambda *_args, **_kwargs: completed_pod(
            exit_code=2, reason="Error"
        ),
    )
    with pytest.raises(ExecutionStartError, match="exit_code=2"):
        wait_for_execution_start(
            core,
            "job-a",
            "test",
            timeout=1,
            api_timeout_seconds=1,
            api_retries=0,
        )


def test_terminal_failure_and_forbidden_logs_fail_immediately():
    failed = SimpleNamespace(
        status=SimpleNamespace(
            phase="Failed", reason="Error", message="boom", container_statuses=[]
        )
    )
    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: "no marker",
        read_namespaced_pod_status=lambda *_args, **_kwargs: failed,
    )
    with pytest.raises(ExecutionStartError, match="Failed phase"):
        wait_for_execution_start(
            core,
            "job-a",
            "test",
            timeout=1,
            api_timeout_seconds=1,
            api_retries=0,
        )

    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ApiException(status=403)
        ),
        read_namespaced_pod_status=lambda *_args, **_kwargs: running_pod(),
    )
    with pytest.raises(ExecutionStartError, match="forbidden"):
        wait_for_execution_start(
            core,
            "job-a",
            "test",
            timeout=1,
            api_timeout_seconds=1,
            api_retries=0,
        )


def test_timeout_reports_last_observation():
    clock = FakeClock()
    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: "no marker",
        read_namespaced_pod_status=lambda *_args, **_kwargs: running_pod(),
    )
    with pytest.raises(ExecutionStartError, match="timed out after 1.000s"):
        wait_for_execution_start(
            core,
            "job-a",
            "test",
            timeout=1,
            api_timeout_seconds=1,
            api_retries=0,
            poll_interval=0.25,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert clock.value == pytest.approx(1.0)


def test_execution_wait_honors_shutdown_before_api_calls():
    core = SimpleNamespace(
        read_namespaced_pod_log=lambda *_args, **_kwargs: pytest.fail(
            "log API should not be called"
        ),
        read_namespaced_pod_status=lambda *_args, **_kwargs: pytest.fail(
            "status API should not be called"
        ),
    )
    with pytest.raises(InterruptedError, match="shutdown"):
        wait_for_execution_start(
            core,
            "job-a",
            "test",
            timeout=1,
            api_timeout_seconds=1,
            api_retries=0,
            stop_requested=lambda: True,
        )
