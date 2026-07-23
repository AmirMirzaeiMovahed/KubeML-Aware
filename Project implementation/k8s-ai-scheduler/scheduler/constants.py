"""Shared metadata contracts for workload pods and controllers."""

ANNOTATION_MAP = {
    "T": "ml.scheduler/estimated-training-time",
    "R": "ml.scheduler/loss-reduction-rate",
    "M": "ml.scheduler/matrix-size",
    "G": "ml.scheduler/gradient-update-size",
    "C": "ml.scheduler/checkpoint-interval",
    "P": "ml.scheduler/model-partitions",
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
RELEASE_GATE = "ml.scheduler/release"
EXECUTION_EVENT = "EXECUTION_STARTED"
