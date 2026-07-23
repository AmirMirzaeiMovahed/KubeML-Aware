import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

from scheduler.execution import (
    ExecutionStartError,
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
