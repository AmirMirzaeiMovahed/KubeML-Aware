import io
import json
from urllib.request import urlopen

import pytest

from scheduler.records import AtomicRecordStore, RecordStoreError, ScheduleRecord
from scheduler.telemetry import HealthServer, HealthState, JsonEventLogger, MetricsRegistry


def test_record_store_is_incremental_atomic_and_versioned(tmp_path):
    path = tmp_path / "run.json"
    store = AtomicRecordStore(str(path), {"run_id": "r1"})
    store.initialize()
    record = ScheduleRecord("job-a", 1, 0.75, pod_uid="uid-job-a")
    store.upsert(record)
    record.status = "bound"
    record.bind_time = 123.0
    store.upsert(record)
    store.append_event({"event": "pacing_wait_completed", "timestamp": 124.0})
    store.set_status("completed")

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 3
    assert document["metadata"]["run_id"] == "r1"
    assert document["status"] == "completed"
    assert document["events"][0]["event"] == "pacing_wait_completed"
    assert "recorded_at" in document["events"][0]
    assert document["records"] == [
        {
            "bind_time": 123.0,
            "error": None,
            "exec_start_time": None,
            "job_id": "job-a",
            "order": 1,
            "pod_uid": "uid-job-a",
            "rank": 0.75,
            "release_time": None,
            "status": "bound",
        }
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_record_store_resumes_only_exact_versioned_state(tmp_path):
    path = tmp_path / "resume.json"
    original = AtomicRecordStore(str(path), {"run_id": "r1"})
    original.initialize()
    original.set_status("running")
    original.upsert(ScheduleRecord("job-a", 1, 0.5, pod_uid="uid-a"))

    resumed = AtomicRecordStore(str(path), {"run_id": "r1"})
    assert resumed.load_existing() is True
    assert resumed.status == "running"
    assert resumed.records[0]["pod_uid"] == "uid-a"

    with pytest.raises(RecordStoreError, match="metadata"):
        AtomicRecordStore(str(path), {"run_id": "other"}).load_existing()

    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 2
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RecordStoreError, match="schema"):
        AtomicRecordStore(str(path), {"run_id": "r1"}).load_existing()


def test_json_logging_and_prometheus_rendering():
    stream = io.StringIO()
    logger = JsonEventLogger(stream=stream, static_fields={"component": "test"})
    logger.info("hello", value=3)
    payload = json.loads(stream.getvalue())
    assert payload["event"] == "hello"
    assert payload["component"] == "test"
    assert payload["value"] == 3

    registry = MetricsRegistry()
    registry.define("jobs_total", "counter", "Jobs processed.")
    registry.inc("jobs_total", labels={"run": 'a"b'})
    rendered = registry.render()
    assert "# TYPE jobs_total counter" in rendered
    assert 'jobs_total{run="a\\"b"} 1' in rendered


def test_health_server_exposes_liveness_readiness_and_metrics():
    state = HealthState(ready=False, reason="warming")
    registry = MetricsRegistry()
    registry.set("ready", 0)
    server = HealthServer("127.0.0.1", 0, state, registry)
    server.start()
    try:
        port = server._server.server_address[1]
        assert json.loads(urlopen(f"http://127.0.0.1:{port}/livez").read())["live"] is True
        state.set_ready(True, "ok")
        assert json.loads(urlopen(f"http://127.0.0.1:{port}/readyz").read())["ready"] is True
        assert b"ready 0" in urlopen(f"http://127.0.0.1:{port}/metrics").read()
    finally:
        server.stop()
