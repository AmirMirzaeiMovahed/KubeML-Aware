"""Shared metadata contracts for workload pods and controllers."""

ANNOTATION_MAP = {
    "T": "ml.scheduler/estimated-training-time",
    "R": "ml.scheduler/loss-reduction-rate",
    "M": "ml.scheduler/matrix-size",
    "G": "ml.scheduler/gradient-update-size",
    "C": "ml.scheduler/checkpoint-interval",
    "P": "ml.scheduler/model-partitions",
}

WORKLOAD_KIND_ANNOTATION = "ml.scheduler/workload-kind"
INFERENCE_ANNOTATION_MAP = {
    "latency_slo_ms": "ml.scheduler/latency-slo-ms",
    "predicted_latency_ms": "ml.scheduler/predicted-latency-ms",
    "request_rate_rps": "ml.scheduler/request-rate-rps",
    "memory_mib": "ml.scheduler/memory-mib",
    "cold_start_ms": "ml.scheduler/cold-start-ms",
    "priority": "ml.scheduler/priority",
}

RUN_ID_LABEL = "ml.scheduler/run-id"
RUN_ID_ANNOTATION = "ml.scheduler/run-id"
SCENARIO_LABEL = "ml.scheduler/scenario"
CONFIG_LABEL = "ml.scheduler/config"
REPETITION_LABEL = "ml.scheduler/repetition"
EXPECTED_COUNT_ANNOTATION = "ml.scheduler/expected-count"
EXPECTED_JOBS_ANNOTATION = "ml.scheduler/expected-jobs"
PACING_MODE_ANNOTATION = "ml.scheduler/pacing-mode"
FIXED_DELAY_ANNOTATION = "ml.scheduler/fixed-delay-seconds"
REVERSE_ANNOTATION = "ml.scheduler/reverse"
RANK_POLICY_ANNOTATION = "ml.scheduler/rank-policy"
TAIL_BALANCE_ANNOTATION = "ml.scheduler/tail-balance-enabled"
FAST_PATH_ANNOTATION = "ml.scheduler/fast-path-enabled"
RELEASE_GATE = "ml.scheduler/release"
EXECUTION_CONTAINER_ANNOTATION = "ml.scheduler/execution-container"
EXECUTION_EVENT = "EXECUTION_STARTED"
EXECUTION_COMPLETED_EVENT = "EXECUTION_COMPLETED"
EXECUTION_FAILED_EVENT = "EXECUTION_FAILED"
