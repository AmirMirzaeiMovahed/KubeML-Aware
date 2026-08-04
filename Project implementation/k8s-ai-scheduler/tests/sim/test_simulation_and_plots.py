import numpy as np
import pandas as pd
import pytest

from scheduler.rank import JobFeatures
from sim.plot_results import PACING_CONFIG_ORDER, mean_ecdf_iqr
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


def test_simulation_outputs_paired_effects_and_exploratory_label(tmp_path):
    dataframe, _summary, documents, improvements = main_comparison(
        settings=SimulationSettings(dt=0.1),
        results_dir=tmp_path,
        include_adaptive=False,
        repeats=1,
    )
    assert len(dataframe) == 9
    assert improvements["comparisons"]
    assert (tmp_path / "paired_improvements.csv").is_file()
    assert all(
        document["environment"]["simulation"]["claim_eligibility"]
        == "exploratory-only"
        for document in documents.values()
    )
