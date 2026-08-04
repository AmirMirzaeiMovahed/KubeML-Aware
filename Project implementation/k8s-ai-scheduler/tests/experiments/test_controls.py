import pytest

from experiments.controls import (
    execution_controls_contract,
    prewarm_pod_manifest,
    validate_cooldown_evidence,
    validate_execution_controls_evidence,
    validate_minikube_attestation,
    validate_prewarm_observation,
)
from experiments.run_cluster import wait_for_cooldown

IMAGE = "registry.example/ml-sim@sha256:" + "a" * 64


def test_prewarm_manifest_is_pinned_secure_and_node_specific():
    manifest = prewarm_pod_manifest(
        namespace="experiment",
        target_node="minikube",
        image=IMAGE,
        image_pull_secrets=["registry"],
    )
    assert manifest["spec"]["nodeName"] == "minikube"
    assert manifest["spec"]["containers"][0]["image"] == IMAGE
    assert manifest["spec"]["containers"][0]["imagePullPolicy"] == "IfNotPresent"
    assert manifest["spec"]["automountServiceAccountToken"] is False
    assert manifest["spec"]["imagePullSecrets"] == [{"name": "registry"}]


def test_prewarm_observation_requires_runtime_blas_and_image_evidence():
    pod = {
        "metadata": {
            "name": "prewarm",
            "uid": "uid",
            "creationTimestamp": "2026-01-01T00:00:00Z",
        },
        "spec": {"nodeName": "minikube", "containers": [{"image": IMAGE}]},
        "status": {
            "phase": "Succeeded",
            "startTime": "2026-01-01T00:00:01Z",
            "containerStatuses": [{
                "restartCount": 0,
                "imageID": "containerd://registry/ml-sim@sha256:" + "b" * 64,
                "state": {"terminated": {"finishedAt": "2026-01-01T00:00:02Z"}},
            }],
        },
    }
    logs = (
        '{"event":"PREWARM_ATTESTATION","matrix_shape":[128,128],'
        '"blas_runtime":{"expected_threads":1,"libraries":'
        '[{"user_api":"blas","num_threads":1}]}}'
    )
    evidence = validate_prewarm_observation(
        pod, logs, expected_image=IMAGE, target_node="minikube"
    )
    assert evidence["status"] == "passed"
    assert len(evidence["evidence_sha256"]) == 64

    controls = {
        "contract": execution_controls_contract(),
        "prewarm": evidence,
        "pre_run_cooldown": {
            "elapsed_seconds": 30.0,
            "clean_polls": 3,
            "workload_pods_observed": 0,
            "scheduler_continuity": True,
            "node_pressure_clear": True,
        },
    }
    assert validate_execution_controls_evidence(
        controls, expected_image=IMAGE, target_node="minikube"
    ) == []
    controls["prewarm"]["target_node"] = "wrong"
    assert "target node" in "; ".join(
        validate_execution_controls_evidence(
            controls, expected_image=IMAGE, target_node="minikube"
        )
    )

    with pytest.raises(ValueError, match="BLAS"):
        validate_prewarm_observation(
            pod,
            logs.replace('"num_threads":1', '"num_threads":2'),
            expected_image=IMAGE,
            target_node="minikube",
        )


def test_cooldown_evidence_is_fail_closed():
    accepted = {
        "elapsed_seconds": 30.0,
        "clean_polls": 3,
        "workload_pods_observed": 0,
        "scheduler_continuity": True,
        "node_pressure_clear": True,
    }
    assert validate_cooldown_evidence(accepted) == []
    accepted["clean_polls"] = 2
    assert "clean_polls" in "; ".join(validate_cooldown_evidence(accepted))


def test_minikube_attestation_requires_running_docker_profile():
    profiles = {
        "valid": [{
            "Name": "paper",
            "Status": "Running",
            "Config": {"Driver": "docker"},
        }]
    }
    evidence = validate_minikube_attestation(
        profiles,
        {
            "Name": "paper",
            "Host": "Running",
            "Kubelet": "Running",
            "APIServer": "Running",
            "Kubeconfig": "Configured",
        },
        profile="paper",
        version="minikube version: v1.40.0",
    )
    assert evidence["driver"] == "docker"

    profiles["valid"][0]["Config"]["Driver"] = "virtualbox"
    with pytest.raises(ValueError, match="docker"):
        validate_minikube_attestation(
            profiles, {}, profile="paper", version="v1.40.0"
        )


def test_runner_cooldown_checks_clean_cluster_node_and_scheduler():
    class FakeKubectl:
        def pod_json(self, _namespace, selector):
            if selector == "app.kubernetes.io/name=ml-sim-job":
                return {"items": []}
            return {
                "items": [{
                    "metadata": {"uid": "scheduler-uid"},
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [{"ready": True, "restartCount": 0}],
                    },
                }]
            }

        def json(self, *args, **_kwargs):
            assert args[:2] == ("get", "node")
            return {
                "status": {
                    "conditions": [
                        {"type": "Ready", "status": "True"},
                        {"type": "MemoryPressure", "status": "False"},
                        {"type": "DiskPressure", "status": "False"},
                        {"type": "PIDPressure", "status": "False"},
                    ]
                }
            }

    evidence = wait_for_cooldown(
        FakeKubectl(),
        namespace="experiment",
        target_node="minikube",
        scheduler_selector="component=scheduler",
        expected_scheduler_uid="scheduler-uid",
        cooldown_seconds=0.0,
        minimum_clean_polls=1,
    )
    assert evidence["clean_polls"] == 1
    assert evidence["scheduler_continuity"] is True
