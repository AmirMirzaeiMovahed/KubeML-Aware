import json

import pytest

from profiling.cli import main
from profiling.model import InferenceSample, build_inference_profile
from scheduler.constants import INFERENCE_ANNOTATION_MAP, WORKLOAD_KIND_ANNOTATION


def sample(latency, *, requests=10, duration=1.0, memory=128, cold=50):
    return InferenceSample(latency, requests, duration, memory, cold)


def test_profile_uses_p95_peak_resources_and_observed_rate():
    profile = build_inference_profile(
        [sample(10, memory=128), sample(30, memory=256, cold=100)],
        job_id="classifier",
        latency_slo_ms=50,
        priority=2,
    )
    assert profile.features.predicted_latency_ms == pytest.approx(29.0)
    assert profile.features.request_rate_rps == pytest.approx(10.0)
    assert profile.features.memory_mib == 256
    assert profile.features.cold_start_ms == 100
    annotations = profile.annotations()
    assert annotations[WORKLOAD_KIND_ANNOTATION] == "inference"
    assert annotations[INFERENCE_ANNOTATION_MAP["predicted_latency_ms"]] == "29"


def test_profile_rejects_empty_or_invalid_measurements():
    with pytest.raises(ValueError, match="at least one"):
        build_inference_profile([], job_id="empty", latency_slo_ms=100)
    with pytest.raises(ValueError, match="requests must be"):
        sample(10, requests=0).validate()


def test_profile_cli_writes_auditable_annotation_document(tmp_path):
    source = tmp_path / "samples.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "latency_ms": latency,
                    "requests": 5,
                    "duration_seconds": 0.5,
                    "memory_mib": 64,
                    "cold_start_ms": 20,
                }
            )
            for latency in (8, 12)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"
    assert main(
        [
            "--input",
            str(source),
            "--job-id",
            "serve-a",
            "--latency-slo-ms",
            "25",
            "--output",
            str(output),
        ]
    ) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["profile_kind"] == "inference"
    assert document["sample_count"] == 2
    assert document["features"]["job_id"] == "serve-a"
