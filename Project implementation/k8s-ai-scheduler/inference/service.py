"""Small deterministic inference service with built-in profiling evidence."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Sequence

import numpy as np

MODEL_VERSION = "synthetic-linear-v1"
MAX_BODY_BYTES = 1_048_576
MAX_BATCH_SIZE = 128
PROCESS_STARTED = time.perf_counter()


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _memory_mib() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows-only test fallback
        return 1e-6
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. The container runtime is Linux.
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return max(float(maximum_rss) / divisor, 1e-6)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class InferenceModel:
    """Deterministic linear-softmax model used as a portable serving target."""

    def __init__(self, input_dim: int, output_dim: int, seed: int):
        if input_dim <= 0 or output_dim <= 1:
            raise ValueError("model dimensions are invalid")
        generator = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = generator.normal(0, 0.25, size=(input_dim, output_dim))
        self.bias = generator.normal(0, 0.05, size=(output_dim,))

    def predict(self, instances: Sequence[Sequence[float]]) -> list[list[float]]:
        if not instances or len(instances) > MAX_BATCH_SIZE:
            raise ValueError(f"instances must contain 1..{MAX_BATCH_SIZE} rows")
        array = np.asarray(instances, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ValueError(f"every instance must have exactly {self.input_dim} features")
        if not np.isfinite(array).all():
            raise ValueError("instances must contain only finite numeric values")
        logits = array @ self.weights + self.bias
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities.tolist()


class ProfileState:
    def __init__(self, cold_start_ms: float):
        self.cold_start_ms = max(cold_start_ms, 1e-6)
        self.started_at = time.monotonic()
        self.request_count = 0
        self.batch_items = 0
        self.latencies_ms: deque[float] = deque(maxlen=2048)
        self.lock = threading.Lock()

    def observe(self, latency_ms: float, batch_items: int) -> dict[str, float | int]:
        with self.lock:
            self.request_count += 1
            self.batch_items += batch_items
            self.latencies_ms.append(latency_ms)
        sample = {
            "latency_ms": latency_ms,
            "requests": batch_items,
            "duration_seconds": max(latency_ms / 1000.0, 1e-9),
            "memory_mib": _memory_mib(),
            "cold_start_ms": self.cold_start_ms,
        }
        return sample

    def snapshot(self) -> dict[str, float | int | str]:
        with self.lock:
            latencies = list(self.latencies_ms)
            requests = self.request_count
            items = self.batch_items
        uptime = max(time.monotonic() - self.started_at, 1e-9)
        return {
            "model_version": MODEL_VERSION,
            "request_count": requests,
            "batch_items": items,
            "observed_items_per_second": items / uptime,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            "memory_mib": _memory_mib(),
            "cold_start_ms": self.cold_start_ms,
        }


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: InferenceModel):
        initialized = time.perf_counter()
        super().__init__(address, InferenceHandler)
        self.model = model
        self.profile = ProfileState((initialized - PROCESS_STARTED) * 1000.0)


class InferenceHandler(BaseHTTPRequestHandler):
    server: InferenceServer

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/healthz", "/readyz"}:
            self._json(HTTPStatus.OK, {"status": "ok", "model_version": MODEL_VERSION})
            return
        if self.path == "/v1/profile":
            self._json(HTTPStatus.OK, self.server.profile.snapshot())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/predict":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= MAX_BODY_BYTES:
                raise ValueError(f"request body must be 1..{MAX_BODY_BYTES} bytes")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
                raise ValueError("JSON body must contain an instances array")
            started = time.perf_counter()
            predictions = self.server.model.predict(payload["instances"])
            latency_ms = max((time.perf_counter() - started) * 1000.0, 1e-6)
            sample = self.server.profile.observe(latency_ms, len(predictions))
            print(
                json.dumps(
                    {"event": "inference_sample", **sample},
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                flush=True,
            )
            self._json(
                HTTPStatus.OK,
                {
                    "model_version": MODEL_VERSION,
                    "predictions": predictions,
                    "sample": sample,
                },
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, _format: str, *_args: object) -> None:
        return


def build_server(host: str, port: int) -> InferenceServer:
    input_dim = _env_int("MODEL_INPUT_DIM", 4, 1, 4096)
    output_dim = _env_int("MODEL_OUTPUT_DIM", 2, 2, 4096)
    seed = _env_int("MODEL_SEED", 2025, 0, 2**31 - 1)
    return InferenceServer((host, port), InferenceModel(input_dim, output_dim, seed))


def main(_argv: Optional[Sequence[str]] = None) -> int:
    host = os.environ.get("HOST", "0.0.0.0")
    port = _env_int("PORT", 8080, 1, 65535)
    server = build_server(host, port)

    def stop(*_args: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
