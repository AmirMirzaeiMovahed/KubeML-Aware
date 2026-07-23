from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException

from scheduler import kube
from scheduler.kube import (
    ApiFailureKind,
    KubernetesOperationError,
    NodeValidationError,
    call_with_retries,
    classify_api_exception,
    parse_cpu_quantity,
    validate_target_node,
)


@pytest.mark.parametrize(
    "quantity, expected",
    [
        ("250m", 0.25),
        ("123456789n", 0.123456789),
        ("500u", 0.0005),
        ("2", 2.0),
        (".5", 0.5),
        ("1e-3", 0.001),
        ("+1", 1.0),
        ("2k", 2000.0),
    ],
)
def test_parse_cpu_quantity(quantity, expected):
    assert parse_cpu_quantity(quantity) == pytest.approx(expected)


@pytest.mark.parametrize("quantity", ["", "-1", "1Mi", "nan", "1.2.3", None])
def test_parse_cpu_quantity_rejects_invalid_values(quantity):
    with pytest.raises(ValueError):
        parse_cpu_quantity(quantity)


def test_api_exception_classification():
    assert classify_api_exception(ApiException(status=503)) is ApiFailureKind.TRANSIENT
    assert classify_api_exception(ApiException(status=409)) is ApiFailureKind.CONFLICT
    assert classify_api_exception(ApiException(status=404)) is ApiFailureKind.NOT_FOUND
    assert classify_api_exception(ApiException(status=403)) is ApiFailureKind.FORBIDDEN
    waiting = ApiException(status=400, reason="container is waiting to start")
    assert classify_api_exception(waiting, log_read=True) is ApiFailureKind.TRANSIENT
    assert classify_api_exception(waiting, log_read=False) is ApiFailureKind.INVALID


def test_retry_is_bounded_and_only_for_transient_failures():
    attempts = []
    sleeps = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise ApiException(status=503)
        return "ok"

    assert call_with_retries(
        operation,
        operation="test",
        retries=2,
        sleep=sleeps.append,
        random_value=lambda: 0.5,
    ) == "ok"
    assert len(attempts) == 3
    assert sleeps == pytest.approx([0.2, 0.4])

    with pytest.raises(KubernetesOperationError) as error:
        call_with_retries(
            lambda: (_ for _ in ()).throw(ApiException(status=403)),
            operation="forbidden",
            retries=99,
        )
    assert error.value.kind is ApiFailureKind.FORBIDDEN


def test_authentication_prefers_in_cluster_and_falls_back(monkeypatch):
    calls = []
    monkeypatch.setattr(kube.config, "load_incluster_config", lambda: calls.append("in"))
    monkeypatch.setattr(kube.config, "load_kube_config", lambda: calls.append("file"))
    assert kube.load_kubernetes_configuration() == "in-cluster"
    assert calls == ["in"]

    calls.clear()

    def unavailable():
        calls.append("in")
        raise ConfigException("not in a pod")

    monkeypatch.setattr(kube.config, "load_incluster_config", unavailable)
    assert kube.load_kubernetes_configuration() == "kubeconfig"
    assert calls == ["in", "file"]


def make_node(*, ready=True, unschedulable=False, cpu="4", deleting=False):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="node-a",
            deletion_timestamp="now" if deleting else None,
        ),
        spec=SimpleNamespace(unschedulable=unschedulable),
        status=SimpleNamespace(
            conditions=[SimpleNamespace(type="Ready", status="True" if ready else "False")],
            allocatable={"cpu": cpu},
        ),
    )


def test_target_node_must_exist_be_ready_schedulable_and_have_cpu():
    core = SimpleNamespace(read_node=lambda *_args, **_kwargs: make_node())
    assert validate_target_node(core, "node-a", timeout=1, retries=0).metadata.name == "node-a"

    for node, message in [
        (make_node(ready=False), "not Ready"),
        (make_node(unschedulable=True), "unschedulable"),
        (make_node(cpu="0"), "no allocatable CPU"),
        (make_node(deleting=True), "being deleted"),
    ]:
        core = SimpleNamespace(read_node=lambda *_args, _node=node, **_kwargs: _node)
        with pytest.raises(NodeValidationError, match=message):
            validate_target_node(core, "node-a", timeout=1, retries=0)
