#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${KUBEML_OUTPUT_ROOT:-/root/kubeml-reviewer-matrix}"
ENVIRONMENT="${KUBEML_ENVIRONMENT:-shared-production}"
REPETITIONS="${KUBEML_REPETITIONS:-30}"
RESUME_RUN="${KUBEML_RESUME_RUN:-}"

cd "$ROOT"
kubectl get namespace ai-scheduler >/dev/null
kubectl -n ai-scheduler rollout status \
  deployment/kubeml-scheduler-ml-ai-scheduler --timeout=120s
kubectl top node >/dev/null

args=(
  --reviewer-matrix
  --matrix-repetitions "$REPETITIONS"
  --environment "$ENVIRONMENT"
  --output-root "$OUTPUT_ROOT"
  --timeout-seconds 300
)
if [[ -n "$RESUME_RUN" ]]; then
  args+=(--resume-run "$RESUME_RUN")
fi

python3 scripts/run_training_fastpath_pilot.py "${args[@]}"

if [[ -n "$RESUME_RUN" ]]; then
  run_directory="$RESUME_RUN"
else
  run_directory="$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi
result="$run_directory/paired-training-results.json"
test -s "$result"

analysis_args=("$result" --output-dir "$run_directory/analysis")
if [[ "$ENVIRONMENT" != "dedicated" ]]; then
  analysis_args+=(--allow-nonclaimable)
fi
python3 scripts/analyze_reviewer_matrix.py "${analysis_args[@]}"

printf 'RESULT=%s\n' "$result"
printf 'ANALYSIS=%s\n' "$run_directory/analysis"
