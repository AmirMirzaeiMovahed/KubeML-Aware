#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_CONTEXT:?Set EXPECTED_CONTEXT to the exact target context}"
: "${REGISTRY:?Set REGISTRY, for example registry.example.com/team}"
: "${TAG:?Set the immutable application tag}"
: "${IMAGE_DIGEST:?Set IMAGE_DIGEST to the pushed scheduler sha256 digest}"

MODE="${MODE:-production}"
NAMESPACE="${NAMESPACE:-ai-scheduler}"
RELEASE="${RELEASE:-ml-ai-scheduler}"
TARGET_NODE="${TARGET_NODE:-}"
IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-}"
LOCAL_VALUES_FILE="${LOCAL_VALUES_FILE:-}"

if [[ "${MODE}" != "production" && "${MODE}" != "reproduction" && "${MODE}" != "reproduction-matrix" ]]; then
  echo "MODE must be production, reproduction, or reproduction-matrix" >&2
  exit 1
fi
if [[ "$(kubectl config current-context)" != "${EXPECTED_CONTEXT}" ]]; then
  echo "refusing to deploy to unexpected kubectl context" >&2
  exit 1
fi
if [[ "${MODE}" != "production" && -z "${TARGET_NODE}" ]]; then
  echo "TARGET_NODE is required in reproduction modes" >&2
  exit 1
fi

values_file="deploy/helm/ml-ai-scheduler/values-${MODE}.yaml"
helm_args=(
  upgrade --install "${RELEASE}" deploy/helm/ml-ai-scheduler
  --namespace "${NAMESPACE}" --create-namespace
  --values "${values_file}"
  --set-string "image.repository=${REGISTRY}/ml-aware-scheduler"
  --set-string "image.tag=${TAG}"
  --set-string "image.digest=${IMAGE_DIGEST}"
  --atomic --wait --timeout 5m
)

if [[ -n "${LOCAL_VALUES_FILE}" ]]; then
  if [[ ! -f "${LOCAL_VALUES_FILE}" ]]; then
    echo "LOCAL_VALUES_FILE does not exist: ${LOCAL_VALUES_FILE}" >&2
    exit 1
  fi
  helm_args+=(--values "${LOCAL_VALUES_FILE}")
fi
if [[ -n "${IMAGE_PULL_SECRET}" ]]; then
  helm_args+=(--set-string "imagePullSecrets[0].name=${IMAGE_PULL_SECRET}")
fi

if [[ -n "${TARGET_NODE}" ]]; then
  helm_args+=(--set-string "scheduler.targetNode=${TARGET_NODE}")
fi

helm "${helm_args[@]}"
kubectl -n "${NAMESPACE}" rollout status "deployment/${RELEASE}-ml-ai-scheduler" --timeout=180s
kubectl -n "${NAMESPACE}" get deployment,pod,service
