"""Turn measured inference samples into auditable scheduler inputs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from scheduler.constants import INFERENCE_ANNOTATION_MAP, WORKLOAD_KIND_ANNOTATION
from scheduler.rank import InferenceFeatures, validate_inference_jobs

PROFILE_SCHEMA_VERSION = "1.0"
PERCENTILE_METHOD = "linear-interpolation"


def _positive(name: str, value: object, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    valid = numeric >= 0 if allow_zero else numeric > 0
    if not math.isfinite(numeric) or not valid:
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {operator}")
    return numeric


@dataclass(frozen=True)
class InferenceSample:
    """One observation window produced by a service or load generator."""

    latency_ms: float
    requests: int
    duration_seconds: float
    memory_mib: float
    cold_start_ms: float

    def validate(self) -> "InferenceSample":
        _positive("latency_ms", self.latency_ms)
        if isinstance(self.requests, bool) or not isinstance(self.requests, int):
            raise ValueError("requests must be a positive integer")
        if self.requests <= 0:
            raise ValueError("requests must be a positive integer")
        _positive("duration_seconds", self.duration_seconds)
        _positive("memory_mib", self.memory_mib)
        _positive("cold_start_ms", self.cold_start_ms)
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "InferenceSample":
        try:
            raw_requests = data["requests"]
            if isinstance(raw_requests, bool) or not isinstance(raw_requests, int):
                raise ValueError("requests must be a positive integer")
            sample = cls(
                latency_ms=float(data["latency_ms"]),
                requests=raw_requests,
                duration_seconds=float(data["duration_seconds"]),
                memory_mib=float(data["memory_mib"]),
                cold_start_ms=float(data["cold_start_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid inference sample fields") from exc
        return sample.validate()


@dataclass(frozen=True)
class InferenceProfile:
    """Profile evidence plus the exact features consumed by the scheduler."""

    features: InferenceFeatures
    sample_count: int
    total_requests: int
    percentile: float = 95.0
    schema_version: str = PROFILE_SCHEMA_VERSION
    percentile_method: str = PERCENTILE_METHOD

    def annotations(self) -> dict[str, str]:
        values = {
            "latency_slo_ms": self.features.latency_slo_ms,
            "predicted_latency_ms": self.features.predicted_latency_ms,
            "request_rate_rps": self.features.request_rate_rps,
            "memory_mib": self.features.memory_mib,
            "cold_start_ms": self.features.cold_start_ms,
            "priority": self.features.priority,
        }
        return {
            WORKLOAD_KIND_ANNOTATION: "inference",
            **{
                INFERENCE_ANNOTATION_MAP[name]: format(float(value), ".12g")
                for name, value in values.items()
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_kind": "inference",
            "sample_count": self.sample_count,
            "total_requests": self.total_requests,
            "latency_percentile": self.percentile,
            "percentile_method": self.percentile_method,
            "features": asdict(self.features),
            "annotations": self.annotations(),
        }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile without samples")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def build_inference_profile(
    samples: Iterable[InferenceSample],
    *,
    job_id: str,
    latency_slo_ms: float,
    priority: float = 1.0,
    percentile: float = 95.0,
) -> InferenceProfile:
    """Aggregate samples using p95 latency, peak memory and observed demand."""

    measured = [sample.validate() for sample in samples]
    if not measured:
        raise ValueError("at least one inference sample is required")
    total_requests = sum(sample.requests for sample in measured)
    total_duration = sum(sample.duration_seconds for sample in measured)
    features = InferenceFeatures(
        job_id=job_id,
        latency_slo_ms=_positive("latency_slo_ms", latency_slo_ms),
        predicted_latency_ms=_percentile(
            [sample.latency_ms for sample in measured], percentile
        ),
        request_rate_rps=total_requests / total_duration,
        memory_mib=max(sample.memory_mib for sample in measured),
        cold_start_ms=max(sample.cold_start_ms for sample in measured),
        priority=_positive("priority", priority),
    )
    validate_inference_jobs([features])
    return InferenceProfile(
        features=features,
        sample_count=len(measured),
        total_requests=total_requests,
        percentile=percentile,
    )
