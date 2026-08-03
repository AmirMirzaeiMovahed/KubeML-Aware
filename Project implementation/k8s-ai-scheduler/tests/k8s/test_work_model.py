import pytest

from k8s.work_model import (
    WORK_MODEL_VERSION,
    estimate_work,
    model_assumptions,
    partition_row_ranges,
)


def test_partition_ranges_cover_every_row_once():
    ranges = partition_row_ranges(11, 4)
    assert ranges == ((0, 3), (3, 6), (6, 9), (9, 11))
    assert [row for start, end in ranges for row in range(start, end)] == list(
        range(11)
    )


def test_convergence_rate_controls_steps_and_termination():
    fast = estimate_work(R=0.2, M=128, G=10, C=50, P=1)
    slow = estimate_work(R=0.001, M=128, G=10, C=50, P=1)
    assert fast.planned_steps < slow.planned_steps
    assert fast.termination_reason == "converged"
    assert slow.termination_reason == "max_steps"
    assert slow.planned_steps == slow.step_budget


def test_features_have_distinct_physical_effects():
    base = estimate_work(R=0.05, M=128, G=2, C=50, P=1)
    larger_matrix = estimate_work(R=0.05, M=256, G=2, C=50, P=1)
    larger_gradient = estimate_work(R=0.05, M=128, G=8, C=50, P=1)
    partitioned = estimate_work(R=0.05, M=128, G=2, C=50, P=4)
    frequent_checkpoints = estimate_work(R=0.05, M=128, G=2, C=5, P=1)

    assert larger_matrix.matrix_compute_seconds > base.matrix_compute_seconds
    assert larger_gradient.gradient_bytes > base.gradient_bytes
    assert larger_gradient.gradient_update_seconds > base.gradient_update_seconds
    assert partitioned.matrix_compute_seconds == pytest.approx(
        base.matrix_compute_seconds
    )
    assert partitioned.partition_sync_seconds > 0
    assert frequent_checkpoints.checkpoint_count > base.checkpoint_count
    assert frequent_checkpoints.checkpoint_seconds > base.checkpoint_seconds


def test_model_assumptions_are_versioned_and_serializable():
    assumptions = model_assumptions()
    assert assumptions["model_version"] == WORK_MODEL_VERSION
    assert assumptions["min_step_budget"] < assumptions["max_step_budget"]


def test_partitions_cannot_exceed_matrix_rows():
    with pytest.raises(ValueError, match="must not exceed"):
        partition_row_ranges(2, 3)
