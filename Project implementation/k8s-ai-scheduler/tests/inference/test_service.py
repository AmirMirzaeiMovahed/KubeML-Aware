import json
import threading
import urllib.error
import urllib.request

import pytest

from inference.service import InferenceModel, InferenceServer


def test_model_is_deterministic_finite_and_normalized():
    first = InferenceModel(3, 2, 7).predict([[1, 2, 3]])
    second = InferenceModel(3, 2, 7).predict([[1, 2, 3]])
    assert first == second
    assert sum(first[0]) == pytest.approx(1.0)


@pytest.fixture
def server():
    instance = InferenceServer(("127.0.0.1", 0), InferenceModel(3, 2, 7))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def request_json(url, *, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read())


def test_http_prediction_and_profile_contract(server):
    status, health = request_json(f"{server}/readyz")
    assert status == 200
    assert health["status"] == "ok"

    status, prediction = request_json(
        f"{server}/v1/predict", payload={"instances": [[1, 2, 3], [3, 2, 1]]}
    )
    assert status == 200
    assert len(prediction["predictions"]) == 2
    assert prediction["sample"]["requests"] == 2
    assert prediction["sample"]["cold_start_ms"] > 0

    _, profile = request_json(f"{server}/v1/profile")
    assert profile["request_count"] == 1
    assert profile["batch_items"] == 2
    assert profile["latency_p95_ms"] > 0


def test_http_rejects_invalid_shape(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request_json(f"{server}/v1/predict", payload={"instances": [[1, 2]]})
    assert error.value.code == 400
