from types import SimpleNamespace

import pytest

from scheduler.burst import (
    AnnotationValidationError,
    BurstCollector,
    BurstContractError,
    extract_features,
    pod_run_id,
    run_settings_for_pods,
)
from scheduler.constants import (
    ANNOTATION_MAP,
    EXPECTED_JOBS_ANNOTATION,
    FIXED_DELAY_ANNOTATION,
    PACING_MODE_ANNOTATION,
    REVERSE_ANNOTATION,
    RUN_ID_ANNOTATION,
    RUN_ID_LABEL,
)


def make_pod(name="job-a", run_id="run-1", **annotation_overrides):
    annotations = {
        ANNOTATION_MAP["T"]: "10",
        ANNOTATION_MAP["R"]: "0.1",
        ANNOTATION_MAP["M"]: "128",
        ANNOTATION_MAP["G"]: "5",
        ANNOTATION_MAP["C"]: "50",
        ANNOTATION_MAP["P"]: "1",
        RUN_ID_ANNOTATION: run_id,
        EXPECTED_JOBS_ANNOTATION: "2",
    }
    annotations.update(annotation_overrides)
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            uid=f"uid-{name}",
            resource_version="1",
            labels={RUN_ID_LABEL: run_id},
            annotations=annotations,
        )
    )


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def test_strict_feature_extraction():
    features = extract_features(make_pod())
    assert features.job_id == "job-a"
    assert features.P == 1.0

    missing = make_pod()
    del missing.metadata.annotations[ANNOTATION_MAP["T"]]
    with pytest.raises(AnnotationValidationError, match="missing required annotation"):
        extract_features(missing)

    with pytest.raises(AnnotationValidationError, match="finite and > 0"):
        extract_features(make_pod(**{ANNOTATION_MAP["R"]: "-1"}))
    with pytest.raises(AnnotationValidationError, match="positive integer"):
        extract_features(make_pod(**{ANNOTATION_MAP["P"]: "1.2"}))


def test_conflicting_run_label_and_annotation_is_rejected():
    pod = make_pod()
    pod.metadata.annotations[RUN_ID_ANNOTATION] = "different"
    with pytest.raises(AnnotationValidationError, match="conflicting run-id"):
        pod_run_id(pod)


def test_per_run_settings_override_fallback_and_must_match():
    annotations = {
        PACING_MODE_ANNOTATION: "fixed",
        FIXED_DELAY_ANNOTATION: "2.5",
        REVERSE_ANNOTATION: "true",
    }
    pods = [make_pod("a", **annotations), make_pod("b", **annotations)]
    settings = run_settings_for_pods(
        pods,
        fallback_run_id=None,
        fallback_expected_count=None,
        fallback_pacing_mode="none",
        fallback_fixed_delay=0,
        fallback_reverse=False,
    )
    assert settings.run_id == "run-1"
    assert settings.expected_count == 2
    assert settings.pacing_mode == "fixed"
    assert settings.fixed_delay == 2.5
    assert settings.reverse is True

    pods[1].metadata.annotations[FIXED_DELAY_ANNOTATION] = "3"
    with pytest.raises(AnnotationValidationError, match="inconsistent"):
        run_settings_for_pods(
            pods,
            fallback_run_id=None,
            fallback_expected_count=None,
            fallback_pacing_mode="none",
            fallback_fixed_delay=0,
            fallback_reverse=False,
        )


def test_collector_requires_exact_count_and_quiet_period():
    clock = FakeClock()
    pods = [make_pod("a"), make_pod("b")]
    collector = BurstCollector(
        lambda: pods,
        quiet_period=1.0,
        timeout=5.0,
        poll_interval=0.25,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert collector.collect("run-1", 2) == pods
    assert clock.value == pytest.approx(1.0)


def test_collector_rejects_overfull_and_times_out_incomplete_bursts():
    clock = FakeClock()
    pods = [make_pod("a"), make_pod("b"), make_pod("c")]
    collector = BurstCollector(
        lambda: pods,
        quiet_period=1,
        timeout=2,
        poll_interval=0.5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with pytest.raises(BurstContractError, match="expected 2 pods but observed 3"):
        collector.collect("run-1", 2)

    clock = FakeClock()
    collector = BurstCollector(
        lambda: pods[:1],
        quiet_period=1,
        timeout=2,
        poll_interval=0.5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with pytest.raises(BurstContractError, match="did not reach exactly 2"):
        collector.collect("run-1", 2)
    assert clock.value == pytest.approx(2.0)
