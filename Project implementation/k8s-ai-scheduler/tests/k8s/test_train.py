import json

import pytest

from k8s import train


def test_invalid_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("JOB_ID", "job-a")
    monkeypatch.setenv("JOB_M", "not-an-integer")
    with pytest.raises(ValueError, match="JOB_M"):
        train.load_config()


def test_marker_is_after_initialization_and_logs_are_json(monkeypatch, capsys):
    config = train.TrainingConfig(
        job_id="job-a",
        seed=7,
        T=10.0,
        R=10.0,
        M=1,
        G=1.0,
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


def test_stable_default_seed(monkeypatch):
    monkeypatch.setenv("JOB_ID", "stable-job")
    monkeypatch.delenv("JOB_SEED", raising=False)
    assert train.load_config().seed == train.load_config().seed
