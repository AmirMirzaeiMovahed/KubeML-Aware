from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return yaml.safe_load(
        (ROOT / "deploy" / "knative" / name).read_text(encoding="utf-8")
    )


def test_knative_inference_service_is_bounded_and_non_root():
    document = load("inference-service.yaml")
    assert document["apiVersion"] == "serving.knative.dev/v1"
    assert document["kind"] == "Service"
    template = document["spec"]["template"]
    annotations = template["metadata"]["annotations"]
    assert annotations["autoscaling.knative.dev/min-scale"] == "0"
    assert int(annotations["autoscaling.knative.dev/max-scale"]) <= 2
    container = template["spec"]["containers"][0]
    assert container["image"] != "latest"
    assert not container["image"].endswith(":latest")
    assert container["resources"]["requests"]
    assert container["resources"]["limits"]
    security = container["securityContext"]
    assert security["runAsNonRoot"] is True
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]


def test_knative_namespace_enforces_restricted_pod_security():
    document = load("namespace.yaml")
    labels = document["metadata"]["labels"]
    assert labels["pod-security.kubernetes.io/enforce"] == "restricted"


def test_france_ingress_bridges_existing_nginx_to_cluster_local_kourier():
    document = load("france-ingress.yaml")
    assert document["kind"] == "Ingress"
    assert document["metadata"]["namespace"] == "kourier-system"
    assert document["spec"]["ingressClassName"] == "nginx"
    rule = document["spec"]["rules"][0]
    assert rule["host"].endswith(".167-104-216-211.nip.io")
    backend = rule["http"]["paths"][0]["backend"]["service"]
    assert backend == {"name": "kourier", "port": {"number": 80}}
