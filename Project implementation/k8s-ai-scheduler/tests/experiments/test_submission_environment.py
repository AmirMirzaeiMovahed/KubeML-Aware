from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.environment import (
    ArticleEnvironmentError,
    parse_quantity,
    validate_article_environment,
)
from experiments.submission import BurstSubmissionError, submit_burst


def _write_pods(directory: Path, names: list[str]) -> None:
    directory.mkdir()
    for name in names:
        payload = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": "experiment"},
            "spec": {"containers": [{"name": "train", "image": "example.invalid/x"}]},
        }
        (directory / f"{name}.yaml").write_text(
            yaml.safe_dump(payload), encoding="utf-8"
        )


def test_concurrent_submission_is_seeded_and_category_neutral(tmp_path: Path):
    names = ["heavy-000", "io-intensive-001", "light-002", "slow-converging-003"]
    _write_pods(tmp_path / "pods", names)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    def create(namespace, body):
        assert namespace == "experiment"
        name = body["metadata"]["name"]
        return SimpleNamespace(
            metadata=SimpleNamespace(
                uid=f"uid-{name}",
                creation_timestamp=base + timedelta(milliseconds=names.index(name)),
            )
        )

    first = submit_burst(
        tmp_path / "pods",
        expected_namespace="experiment",
        expected_count=4,
        seed=91,
        create_pod=create,
        max_workers=4,
        max_creation_spread_seconds=1.0,
    )
    second = submit_burst(
        tmp_path / "pods",
        expected_namespace="experiment",
        expected_count=4,
        seed=91,
        create_pod=create,
        max_workers=4,
        max_creation_spread_seconds=1.0,
    )
    assert first["mode"] == "concurrent-client-barrier"
    assert first["randomized_job_order"] == second["randomized_job_order"]
    assert first["randomized_job_order"] != sorted(names)
    assert {row["job_id"] for row in first["jobs"]} == set(names)
    assert first["server_creation_spread_seconds"] == pytest.approx(0.003, abs=1e-6)


def test_submission_fails_closed_when_server_spread_is_too_wide(tmp_path: Path):
    names = ["heavy-000", "light-001"]
    _write_pods(tmp_path / "pods", names)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    def create(_namespace, body):
        offset = 0 if body["metadata"]["name"] == names[0] else 10
        return SimpleNamespace(
            metadata=SimpleNamespace(
                uid="uid", creation_timestamp=base + timedelta(seconds=offset)
            )
        )

    with pytest.raises(BurstSubmissionError, match="creation spread") as captured:
        submit_burst(
            tmp_path / "pods",
            expected_namespace="experiment",
            expected_count=2,
            seed=1,
            create_pod=create,
            max_workers=2,
            max_creation_spread_seconds=1.0,
        )
    assert captured.value.evidence["server_creation_spread_seconds"] == 10.0


def _snapshot() -> dict:
    return {
        "kubectl_context": "minikube",
        "minikube": {
            "profile": "minikube",
            "driver": "docker",
            "profile_status": "Running",
        },
        "cluster_nodes": [{"name": "minikube", "uid": "node-uid"}],
        "target_node": {
            "name": "minikube",
            "labels": {"minikube.k8s.io/primary": "true"},
            "capacity": {"cpu": "4", "memory": "8Gi"},
            "conditions": [
                {"type": "Ready", "status": "True"},
                {"type": "MemoryPressure", "status": "False"},
                {"type": "DiskPressure", "status": "False"},
                {"type": "PIDPressure", "status": "False"},
            ],
        },
        "scheduler_deployment": {
            "pods": [{"name": "scheduler-abc", "uid": "scheduler-uid"}]
        },
        "target_node_pods": [
            {"namespace": "kube-system", "name": "coredns", "phase": "Running"},
            {"namespace": "experiment", "name": "scheduler-abc", "phase": "Running"},
        ],
    }


def test_article_environment_is_fail_closed_and_record_only_is_ineligible():
    snapshot = _snapshot()
    accepted = validate_article_environment(
        snapshot, experiment_namespace="experiment", profile="article-exact"
    )
    assert accepted["article_claim_eligible"] is True

    snapshot["cluster_nodes"].append({"name": "other", "uid": "other"})
    snapshot["target_node_pods"].append(
        {"namespace": "vpn", "name": "openvpn", "phase": "Running"}
    )
    with pytest.raises(ArticleEnvironmentError) as captured:
        validate_article_environment(
            snapshot, experiment_namespace="experiment", profile="article-exact"
        )
    assert any("exactly one" in error for error in captured.value.errors)
    assert any("openvpn" in error for error in captured.value.errors)

    recorded = validate_article_environment(
        snapshot, experiment_namespace="experiment", profile="record-only"
    )
    assert recorded["article_claim_eligible"] is False
    assert recorded["errors"]


def test_kubernetes_quantity_parser_covers_node_capacity_forms():
    assert parse_quantity("4") == 4.0
    assert parse_quantity("4000m") == 4.0
    assert parse_quantity("8Gi") == 8 * 1024**3
    assert parse_quantity("8192Mi") == 8 * 1024**3
