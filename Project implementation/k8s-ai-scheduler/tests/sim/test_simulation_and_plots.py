import numpy as np
import pandas as pd
import pytest

from scheduler.rank import JobFeatures
from sim.plot_results import ARTICLE_FIGURE_CONFIGS, PACING_CONFIG_ORDER, mean_ecdf_iqr
from sim.run_experiments import SimulationSettings, main_comparison
from sim.simulate import (
    SimulationIncompleteError,
    TrainerWorkModel,
    estimate_trainer_work,
    run_default,
    run_paced,
)


def _jobs(count=3):
    return [
        JobFeatures(f"job-{index}", 10 + index, 0.1, 32 + index, 1, 50, 1)
        for index in range(count)
    ]


def test_completion_timestamp_is_interpolated_within_step():
    job = _jobs(1)[0]
    model = TrainerWorkModel(matmul_seconds_at_reference=0.001)
    work = estimate_trainer_work(job, model)
    result = run_default(
        [job], n_cores=1, dt=1.0, mean_ilt=0.0, alpha=0.0, work_model=model,
    )[0]
    assert result.exec_start_t == 0.0
    assert result.completion_t == pytest.approx(work)
    assert 0 < result.completion_t < 1.0


def test_default_parameter_is_mean_inter_launch_time_not_total_window():
    model = TrainerWorkModel(matmul_seconds_at_reference=0.0001)
    results = run_default(
        _jobs(), n_cores=4, dt=0.1, mean_ilt=0.7, alpha=0.0, work_model=model,
    )
    starts = sorted(result.exec_start_t for result in results)
    assert starts == pytest.approx([0.0, 0.7, 1.4])


def test_incomplete_simulation_raises_with_unfinished_ids():
    with pytest.raises(SimulationIncompleteError) as caught:
        run_paced(_jobs(), dt=0.1, max_time=0.01)
    assert set(caught.value.unfinished_job_ids) == {"job-0", "job-1", "job-2"}


def test_trainer_model_uses_matrix_and_partitions():
    base = JobFeatures("base", 10, 0.1, 64, 2, 50, 1)
    larger = JobFeatures("large", 10, 0.1, 128, 2, 50, 1)
    partitioned = JobFeatures("parts", 10, 0.1, 64, 2, 50, 2)
    assert estimate_trainer_work(larger) > estimate_trainer_work(base)
    assert estimate_trainer_work(partitioned) > estimate_trainer_work(base)


def test_zero_dt_is_rejected():
    with pytest.raises(ValueError, match="dt"):
        run_default(_jobs(), dt=0)


def test_mean_ecdf_and_iqr_are_computed_across_runs():
    frame = pd.DataFrame({
        "rep": [0, 0, 1, 1],
        "jct": [1.0, 2.0, 2.0, 4.0],
    })
    grid, mean, lower, upper = mean_ecdf_iqr(frame)
    assert np.array_equal(grid, [1.0, 2.0, 4.0])
    assert mean == pytest.approx([0.25, 0.75, 1.0])
    assert lower == pytest.approx([0.125, 0.625, 1.0])
    assert upper == pytest.approx([0.375, 0.875, 1.0])


def test_pacing_plot_order_contains_each_registered_configuration_once():
    assert len(PACING_CONFIG_ORDER) == len(set(PACING_CONFIG_ORDER))
    assert PACING_CONFIG_ORDER[:5] == (
        "default",
        "custom-baseline",
        "custom-delay-1s",
        "custom-delay-2s",
        "custom-delay-5s",
    )


def test_article_figures_do_not_mix_in_extension_or_ablation_configs():
    assert ARTICLE_FIGURE_CONFIGS == ("default", "custom-baseline")


def _tail_jobs():
    # Equal R, M, G, C, P so rank is driven purely by T (smaller T ranks first);
    # the T=30 job is therefore ranked last and lands in the balanced tail.
    return [
        JobFeatures(f"job-{index}", duration, 0.1, 64, 5, 50, 1)
        for index, duration in enumerate((1, 2, 3, 4, 5, 30))
    ]


def _duration_weighted_model():
    # estimated_time_weight=1.0 makes simulated work equal to T exactly, so the
    # T-based tail balancer and the sim share one duration model.
    return TrainerWorkModel(estimated_time_weight=1.0)


def test_balance_tail_pulls_longest_job_forward_and_preserves_prefix():
    jobs = _tail_jobs()
    common = dict(
        n_cores=2, dt=0.1, alpha=0.0, inherent_gap=1.0,
        work_model=_duration_weighted_model(),
    )
    baseline = {r.job_id: r for r in run_paced(jobs, mode="fixed", **common)}
    balanced = {
        r.job_id: r
        for r in run_paced(jobs, mode="fixed", balance_tail=True, tail_window=3, **common)
    }
    # The longest, lowest-ranked job starts strictly earlier once the tail is balanced.
    assert balanced["job-5"].exec_start_t < baseline["job-5"].exec_start_t
    # The protected high-priority prefix is frozen, so mean-JCT behavior is preserved.
    for frozen in ("job-0", "job-1", "job-2"):
        assert balanced[frozen].exec_start_t == pytest.approx(baseline[frozen].exec_start_t)
    # No job is dropped or duplicated.
    assert set(balanced) == set(baseline)


def test_balance_tail_is_skipped_for_reversed_ablation():
    jobs = _tail_jobs()
    common = dict(
        n_cores=2, dt=0.1, alpha=0.0, inherent_gap=1.0, reverse=True,
        work_model=_duration_weighted_model(),
    )
    plain = {r.job_id: r.exec_start_t for r in run_paced(jobs, mode="fixed", **common)}
    balanced = {
        r.job_id: r.exec_start_t
        for r in run_paced(jobs, mode="fixed", balance_tail=True, tail_window=3, **common)
    }
    assert balanced == pytest.approx(plain)


def test_run_paced_rejects_invalid_tail_window():
    with pytest.raises(ValueError, match="tail_window"):
        run_paced(_tail_jobs(), dt=0.1, balance_tail=True, tail_window=1)


def test_tail_balanced_run_records_balancing_evidence(tmp_path):
    _dataframe, _summary, documents, _improvements = main_comparison(
        settings=SimulationSettings(dt=0.1),
        results_dir=tmp_path,
        include_adaptive=False,
        repeats=1,
    )
    balanced = documents["sim-12-normal-custom-tail-balanced-r0"]
    tail = balanced["environment"]["simulation"]["tail_balance"]
    assert set(tail) >= {
        "selected",
        "reason",
        "original_tail",
        "balanced_tail",
        "predicted_makespan_before",
        "predicted_makespan_after",
    }
    # Non-balanced configs must not carry balancing evidence.
    baseline = documents["sim-12-normal-custom-baseline-r0"]
    assert "tail_balance" not in baseline["environment"]["simulation"]


def test_simulation_outputs_paired_effects_and_exploratory_label(tmp_path):
    dataframe, _summary, documents, improvements = main_comparison(
        settings=SimulationSettings(dt=0.1),
        results_dir=tmp_path,
        include_adaptive=False,
        repeats=1,
    )
    # default, custom-baseline, custom-tail-balanced, reversed across 3 scenarios
    assert len(dataframe) == 12
    assert improvements["comparisons"]
    assert (tmp_path / "paired_effects.csv").is_file()
    assert all(
        document["environment"]["simulation"]["claim_eligibility"]
        == "exploratory-only"
        for document in documents.values()
    )
