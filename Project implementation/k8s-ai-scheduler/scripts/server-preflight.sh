#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_CONTEXT:?Set EXPECTED_CONTEXT to the exact kubectl context you intend to inspect}"

for command_name in kubectl helm docker curl python3; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 1
  }
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
"${PYTHON_BIN}" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version; print(sys.version)'
"${PYTHON_BIN}" -c 'import jsonschema,kubernetes,kubernetes_validate,matplotlib,numpy,pandas,prometheus_client,yaml; print("runtime imports: OK")'
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -c 'from experiments.run_cluster import DEFAULT_PLAN,expand_plan; assert len(expand_plan(DEFAULT_PLAN)) == 70; assert len(expand_plan(DEFAULT_PLAN, include_adaptive=True)) == 90; print("experiment plans: 70/90 OK")'

actual_context="$(kubectl config current-context)"
if [[ "${actual_context}" != "${EXPECTED_CONTEXT}" ]]; then
  echo "refusing to continue: current context '${actual_context}' != EXPECTED_CONTEXT '${EXPECTED_CONTEXT}'" >&2
  exit 1
fi

kubectl version -o yaml
kubectl cluster-info
kubectl get nodes -o wide
kubectl get --raw='/readyz?verbose'
helm version
docker version
helm lint deploy/helm/ml-ai-scheduler
production_render="$(mktemp)"
reproduction_render="$(mktemp)"
matrix_render="$(mktemp)"
trap 'rm -f -- "${production_render}" "${reproduction_render}" "${matrix_render}"' EXIT
helm template ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace ai-scheduler \
  --values deploy/helm/ml-ai-scheduler/values-production.yaml >"${production_render}"
helm template ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace ai-scheduler \
  --values deploy/helm/ml-ai-scheduler/values-reproduction.yaml \
  --set-string scheduler.targetNode=preflight-node >"${reproduction_render}"
helm template ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace ai-scheduler \
  --values deploy/helm/ml-ai-scheduler/values-reproduction-matrix.yaml \
  --set-string scheduler.targetNode=preflight-node >"${matrix_render}"
"${PYTHON_BIN}" scripts/validate_manifests.py \
  "${production_render}" "${reproduction_render}" "${matrix_render}" \
  --kubernetes-version 1.36.0

if kubectl get apiservice v1beta1.metrics.k8s.io >/dev/null 2>&1; then
  kubectl get --raw='/apis/metrics.k8s.io/v1beta1/nodes' >/dev/null
  echo "metrics.k8s.io: available"
else
  echo "metrics.k8s.io: unavailable (required only for adaptive pacing)"
fi

echo "Server preflight completed. No cluster objects were changed."
