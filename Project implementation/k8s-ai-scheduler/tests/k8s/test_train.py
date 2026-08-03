import json

import pytest

from k8s import train


def test_invalid_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("JOB_ID", "job-a")
    monkeypatch.setenv("JOB_M", "not-an-integer")
    with pytest.raises(ValueError, match="JOB_M"):
        train.load_config()


def test_work_model_version_mismatch_is_rejected(monkeypatch):
    monkeypatch.setenv("ML_WORK_MODEL_VERSION", "stale-model")
    with pytest.raises(ValueError, match="does not match"):
        train.load_config()


def test_marker_is_after_initialization_and_logs_are_json(monkeypatch, capsys):
    config = train.TrainingConfig(
        job_id="job-a",
        seed=7,
        T=10.0,
        R=10.0,
        M=1,
        G=0.001,
        C=50,
        P=1,
    )
    result = train.run_training(config)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    event_names = [record["event"] for record in events]
    assert event_names[:2] == ["INITIALIZATION_COMPLETED", "EXECUTION_STARTED"]
    assert event_names[-1] == "EXECUTION_COMPLETED"
    assert events[1]["job_id"] == "job-a"
    assert result["steps"] == 1
    assert result["termination_reason"] == "converged"
    assert result["work_model_version"] == train.WORK_MODEL_VERSION


def test_partition_and_checkpoint_work_is_observable(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ML_CHECKPOINT_DIR", str(tmp_path))
    config = train.TrainingConfig(
        job_id="checkpoint-job",
        seed=11,
        T=1.0,
        R=10.0,
        M=5,
        G=0.001,
        C=1,
        P=3,
    )
    result = train.run_training(config)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    initialized = events[0]
    checkpoint = next(event for event in events if event["event"] == "CHECKPOINT")

    assert initialized["partition_rows"] == [[0, 2], [2, 4], [4, 5]]
    assert checkpoint["bytes_written"] == result["checkpoint_bytes"]
    assert result["checkpoint_count"] == 1
    checkpoint_files = list(tmp_path.glob("ml-checkpoint-*.bin"))
    assert len(checkpoint_files) == 1
    assert checkpoint_files[0].stat().st_size == result["checkpoint_bytes"]


def test_slow_convergence_reaches_finite_step_budget(capsys):
    config = train.TrainingConfig(
        job_id="bounded-job",
        seed=3,
        T=1.0,
        R=0.001,
        M=1,
        G=0.001,
        C=1000,
        P=1,
    )
    result = train.run_training(config)
    capsys.readouterr()
    assert result["termination_reason"] == "max_steps"
    assert result["steps"] == result["step_budget"]


def test_stable_default_seed(monkeypatch):
    monkeypatch.setenv("JOB_ID", "stable-job")
    monkeypatch.delenv("JOB_SEED", raising=False)
    assert train.load_config().seed == train.load_config().seed
