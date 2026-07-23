import json
import random
from pathlib import Path

import pytest
import yaml

from results.metrics_collector import FEATURE_ANNOTATIONS as COLLECTOR_FEATURE_ANNOTATIONS
from scheduler.constants import (
    ANNOTATION_MAP,
    CONFIG_LABEL,
    REPETITION_LABEL,
    RUN_ID_LABEL,
    SCENARIO_LABEL,
)
from workload.generate_workload import (
    FEATURE_ANNOTATIONS,
    LABEL_KEYS,
    OUTPUT_SENTINEL,
    _sample_category,
    generate_burst,
    prepare_output_directory,
    to_pod_yaml,
)


def test_metadata_contracts_cannot_drift_between_components():
    assert FEATURE_ANNOTATIONS == ANNOTATION_MAP == COLLECTOR_FEATURE_ANNOTATIONS
    assert LABEL_KEYS == {
        "run_id": RUN_ID_LABEL,
        "scenario": SCENARIO_LABEL,
        "config": CONFIG_LABEL,
        "repetition": REPETITION_LABEL,
    }


def test_burst_is_deterministic_and_ids_depend_on_seed_and_index():
    first = generate_burst(25, seed=123, load="normal")
    second = generate_burst(25, seed=123, load="normal")
    different = generate_burst(25, seed=124, load="normal")
    assert first == second
    assert [job.job_id for job in first] != [job.job_id for job in different]
    assert len({job.job_id for job in first}) == 25


def test_half_profile_only_scales_matrix_dimension():
    normal = generate_burst(40, seed=987, load="normal")
    half = generate_burst(40, seed=987, load="half")
    for full_job, half_job in zip(normal, half):
        assert full_job.job_id == half_job.job_id
        assert half_job.M == max(1, int(round(full_job.M * 0.5)))
        for feature in ("T", "R", "G", "C", "P"):
            assert getattr(half_job, feature) == getattr(full_job, feature)


def test_partition_sampling_includes_upper_bound():
    rng = random.Random(42)
    sampled = {
        _sample_category("heavy", rng, seed=42, index=index).P
        for index in range(500)
    }
    assert sampled == {2, 3, 4}


def test_pod_yaml_is_valid_and_contains_run_contract():
    job = generate_burst(1, seed=4)[0]
    manifest = yaml.safe_load(to_pod_yaml(
        job,
        "ml-aware-scheduler",
        "registry.example/ml-sim:v1",
        namespace="experiment",
        run_id="run-4",
        scenario="12-normal",
        config="custom-delay-1s",
        repetition=2,
        expected_jobs=12,
        seed=4,
        pacing_mode="fixed",
        fixed_delay_seconds=1.0,
        image_pull_secrets=["registry-credentials"],
    ))
    assert manifest["kind"] == "Pod"
    assert manifest["metadata"]["labels"] == {
        "app": "ml-sim-job",
        "app.kubernetes.io/name": "ml-sim-job",
        "ml.scheduler/run-id": "run-4",
        "ml.scheduler/scenario": "12-normal",
        "ml.scheduler/config": "custom-delay-1s",
        "ml.scheduler/repetition": "2",
    }
    annotations = manifest["metadata"]["annotations"]
    assert annotations["ml.scheduler/pacing-mode"] == "fixed"
    assert annotations["ml.scheduler/fixed-delay-seconds"] == "1.0"
    assert annotations["ml.scheduler/reverse"] == "false"
    assert annotations["ml.scheduler/expected-jobs"] == "12"
    assert manifest["spec"]["automountServiceAccountToken"] is False
    assert manifest["spec"]["imagePullSecrets"] == [
        {"name": "registry-credentials"}
    ]


def test_invalid_image_pull_secret_is_rejected():
    job = generate_burst(1, seed=4)[0]
    with pytest.raises(ValueError, match="image pull secret"):
        to_pod_yaml(
            job,
            "ml-aware-scheduler",
            "registry.example/ml-sim:v1",
            image_pull_secrets=["Invalid Secret"],
        )


def test_nonempty_output_is_rejected_and_overwrite_requires_sentinel(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_directory(output, overwrite=False)
    with pytest.raises(RuntimeError, match="sentinel"):
        prepare_output_directory(output, overwrite=True)
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_generator_owned_directory_can_be_safely_regenerated(tmp_path: Path):
    output = tmp_path / "run"
    prepare_output_directory(output, overwrite=False)
    (output / "jobs.json").write_text("{}", encoding="utf-8")
    pods = output / "pods_custom"
    pods.mkdir()
    (pods / "old.yaml").write_text("kind: Pod", encoding="utf-8")
    prepare_output_directory(output, overwrite=True)
    assert (output / OUTPUT_SENTINEL).is_file()
    assert not (output / "jobs.json").exists()
    assert not pods.exists()
