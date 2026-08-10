"""Validated scheduler configuration shared by both execution profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class ConfigurationError(ValueError):
    pass


@dataclass
class SchedulerConfig:
    scheduler_name: str = "ml-aware-scheduler"
    namespace: str = "default"
    run_id: Optional[str] = None
    expected_count: Optional[int] = None
    quiet_period: float = 1.5
    burst_timeout: float = 120.0
    poll_interval: float = 0.5
    pacing_mode: str = "none"
    fixed_delay: float = 0.0
    cpu_threshold: float = 0.85
    adaptive_hysteresis: float = 0.05
    max_wait: float = 30.0
    metrics_max_age: float = 30.0
    fast_path_enabled: bool = False
    # Keep the library default identical to the production Helm profile and
    # the threshold recorded in the paired pilot evidence.
    fast_path_cpu_threshold: float = 0.85
    execution_timeout: float = 120.0
    api_timeout: float = 10.0
    api_retries: int = 4
    target_node: Optional[str] = None
    reverse: bool = False
    results_path: str = "../results/schedule_run.json"
    health_host: str = "0.0.0.0"
    health_port: int = 8080

    def runtime_metadata(self) -> Dict[str, object]:
        """Return the non-secret runtime contract persisted with every run."""

        return {
            "runtime_contract_version": "1.0",
            "quiet_period_seconds": self.quiet_period,
            "burst_timeout_seconds": self.burst_timeout,
            "poll_interval_seconds": self.poll_interval,
            "execution_timeout_seconds": self.execution_timeout,
            "api_timeout_seconds": self.api_timeout,
            "api_retries": self.api_retries,
            "cpu_threshold": self.cpu_threshold,
            "adaptive_hysteresis": self.adaptive_hysteresis,
            "max_wait_seconds": self.max_wait,
            "metrics_max_age_seconds": self.metrics_max_age,
            "fast_path_enabled": self.fast_path_enabled,
            "fast_path_cpu_threshold": self.fast_path_cpu_threshold,
        }

    def validate(self, *, manual_binding: bool = False) -> "SchedulerConfig":
        if not self.scheduler_name.strip():
            raise ConfigurationError("scheduler_name must not be empty")
        if not self.namespace.strip():
            raise ConfigurationError("namespace must not be empty")
        if self.run_id is not None and not self.run_id.strip():
            raise ConfigurationError("run_id must be omitted or non-empty")
        if self.expected_count is not None and self.expected_count <= 0:
            raise ConfigurationError("expected_count must be > 0")
        if self.run_id is None and self.expected_count is not None:
            raise ConfigurationError("expected_count requires an explicit run_id")
        if self.run_id is not None and self.expected_count is None:
            raise ConfigurationError("run_id requires expected_count")
        for name in (
            "quiet_period",
            "burst_timeout",
            "poll_interval",
            "max_wait",
            "metrics_max_age",
            "execution_timeout",
            "api_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be > 0")
        if self.fixed_delay < 0:
            raise ConfigurationError("fixed_delay must be >= 0")
        if self.pacing_mode not in {"none", "fixed", "adaptive"}:
            raise ConfigurationError("pacing_mode must be none, fixed, or adaptive")
        if not 0 < self.cpu_threshold <= 1:
            raise ConfigurationError("cpu_threshold must be in (0, 1]")
        if not 0 < self.fast_path_cpu_threshold <= 1:
            raise ConfigurationError("fast_path_cpu_threshold must be in (0, 1]")
        if not 0 <= self.adaptive_hysteresis < self.cpu_threshold:
            raise ConfigurationError(
                "adaptive_hysteresis must be >= 0 and smaller than cpu_threshold"
            )
        if self.api_retries < 0:
            raise ConfigurationError("api_retries must be >= 0")
        if not 1 <= self.health_port <= 65535:
            raise ConfigurationError("health_port must be in [1, 65535]")
        if manual_binding and not self.target_node:
            raise ConfigurationError("manual-binding reproduction requires an explicit target_node")
        return self
