#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-scheduler}"
SERVICE="${SERVICE:-ml-ai-scheduler-ml-ai-scheduler}"
LOCAL_PORT="${LOCAL_PORT:-18080}"

port_forward_log="$(mktemp)"
metrics_output="$(mktemp)"
kubectl -n "${NAMESPACE}" port-forward "service/${SERVICE}" "${LOCAL_PORT}:8080" >"${port_forward_log}" 2>&1 &
port_forward_pid=$!
cleanup() {
  kill "${port_forward_pid}" >/dev/null 2>&1 || true
  wait "${port_forward_pid}" >/dev/null 2>&1 || true
  rm -f -- "${port_forward_log}" "${metrics_output}"
}
trap cleanup EXIT

for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/livez" >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${LOCAL_PORT}/livez"; then
  cat "${port_forward_log}" >&2
  exit 1
fi
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/readyz"
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/metrics" -o "${metrics_output}"
head -n 20 "${metrics_output}"

kubectl -n "${NAMESPACE}" get events --sort-by=.metadata.creationTimestamp
echo "Smoke test completed successfully."
