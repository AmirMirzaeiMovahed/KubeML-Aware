"""Dependency-free structured logs, health checks and Prometheus metrics."""

from __future__ import annotations

import json
import math
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Mapping, Optional, Tuple


class JsonEventLogger:
    def __init__(self, stream=None, static_fields: Optional[Mapping[str, object]] = None):
        self.stream = stream or sys.stdout
        self.static_fields = dict(static_fields or {})
        self._lock = threading.Lock()

    def emit(self, event: str, level: str = "INFO", **fields: object) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "event": event,
            **self.static_fields,
            **fields,
        }
        with self._lock:
            print(json.dumps(payload, sort_keys=True, default=str), file=self.stream, flush=True)

    def info(self, event: str, **fields: object) -> None:
        self.emit(event, "INFO", **fields)

    def warning(self, event: str, **fields: object) -> None:
        self.emit(event, "WARNING", **fields)

    def error(self, event: str, **fields: object) -> None:
        self.emit(event, "ERROR", **fields)


def _labels_key(labels: Optional[Mapping[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class MetricsRegistry:
    """Small registry covering the scheduler's counters and gauges."""

    def __init__(self):
        self._values: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self._types: Dict[str, str] = {}
        self._help: Dict[str, str] = {}
        self._lock = threading.Lock()

    def define(self, name: str, metric_type: str, help_text: str) -> None:
        if metric_type not in {"counter", "gauge"}:
            raise ValueError("metric_type must be counter or gauge")
        with self._lock:
            self._types[name] = metric_type
            self._help[name] = help_text

    def inc(
        self, name: str, amount: float = 1.0, labels: Optional[Mapping[str, str]] = None
    ) -> None:
        if amount < 0:
            raise ValueError("counter increment must not be negative")
        with self._lock:
            self._types.setdefault(name, "counter")
            key = (name, _labels_key(labels))
            self._values[key] = self._values.get(key, 0.0) + amount

    def set(
        self, name: str, value: float, labels: Optional[Mapping[str, str]] = None
    ) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        with self._lock:
            self._types.setdefault(name, "gauge")
            self._values[(name, _labels_key(labels))] = value

    def render(self) -> str:
        lines = []
        with self._lock:
            names = sorted(set(self._types) | {key[0] for key in self._values})
            for name in names:
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} {self._types.get(name, 'gauge')}")
                for (metric_name, labels), value in sorted(self._values.items()):
                    if metric_name != name:
                        continue
                    label_text = ""
                    if labels:
                        rendered = ",".join(
                            f'{key}="{_escape_label(label)}"' for key, label in labels
                        )
                        label_text = "{" + rendered + "}"
                    lines.append(f"{name}{label_text} {value:.12g}")
        return "\n".join(lines) + "\n"


@dataclass
class HealthState:
    live: bool = True
    ready: bool = False
    reason: str = "initializing"

    def __post_init__(self):
        self._lock = threading.Lock()

    def set_ready(self, ready: bool, reason: str) -> None:
        with self._lock:
            self.ready = bool(ready)
            self.reason = str(reason)

    def snapshot(self) -> Tuple[bool, bool, str]:
        with self._lock:
            return self.live, self.ready, self.reason


class HealthServer:
    def __init__(self, host: str, port: int, state: HealthState, metrics: MetricsRegistry):
        self.host = host
        self.port = port
        self.state = state
        self.metrics = metrics
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        state = self.state
        metrics = self.metrics

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                live, ready, reason = state.snapshot()
                if self.path == "/metrics":
                    body = metrics.render().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                elif self.path == "/livez":
                    body = json.dumps({"live": live}).encode("utf-8")
                    self.send_response(200 if live else 503)
                    self.send_header("Content-Type", "application/json")
                elif self.path == "/readyz":
                    body = json.dumps({"ready": ready, "reason": reason}).encode("utf-8")
                    self.send_response(200 if ready else 503)
                    self.send_header("Content-Type", "application/json")
                else:
                    body = b'{"error":"not found"}'
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="scheduler-health", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None


def default_metrics() -> MetricsRegistry:
    registry = MetricsRegistry()
    definitions = {
        "ml_scheduler_bursts_total": ("counter", "Burst processing attempts."),
        "ml_scheduler_burst_jobs": ("gauge", "Jobs observed in the current burst."),
        "ml_scheduler_releases_total": ("counter", "Successfully bound or released jobs."),
        "ml_scheduler_failures_total": ("counter", "Scheduler failures by stage."),
        "ml_scheduler_api_retries_total": ("counter", "Kubernetes API retries."),
        "ml_scheduler_cpu_utilization_ratio": ("gauge", "Latest target-node CPU ratio."),
        "ml_scheduler_metrics_age_seconds": ("gauge", "Age of the latest node metric."),
        "ml_scheduler_ready": ("gauge", "Whether the scheduler is ready."),
    }
    for name, (kind, help_text) in definitions.items():
        registry.define(name, kind, help_text)
    return registry

