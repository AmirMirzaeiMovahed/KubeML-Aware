"""Crash-safe, incremental schedule records."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional


class RecordStoreError(RuntimeError):
    """Raised when persisted scheduler state is unsafe to resume."""


@dataclass
class ScheduleRecord:
    job_id: str
    order: int
    rank: float
    pod_uid: Optional[str] = None
    status: str = "ranked"
    bind_time: Optional[float] = None
    release_time: Optional[float] = None
    exec_start_time: Optional[float] = None
    error: Optional[str] = None


class AtomicRecordStore:
    SCHEMA_VERSION = 3
    VALID_STATUSES = {"initializing", "running", "completed", "failed"}

    def __init__(self, path: str, metadata: Optional[Dict[str, object]] = None):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        self._document = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
            "status": "initializing",
            "error": None,
            "records": [],
            "events": [],
        }

    @property
    def records(self) -> List[Dict[str, object]]:
        return list(self._document["records"])

    @property
    def events(self) -> List[Dict[str, object]]:
        return list(self._document["events"])

    @property
    def status(self) -> str:
        return str(self._document["status"])

    @property
    def metadata(self) -> Dict[str, object]:
        return dict(self._document["metadata"])

    def initialize(self) -> None:
        self._write()

    def load_existing(self) -> bool:
        """Load and validate an existing state file without modifying it."""

        with self._lock:
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    document = json.load(handle)
            except FileNotFoundError:
                return False
            except (OSError, json.JSONDecodeError) as exc:
                raise RecordStoreError(
                    f"scheduler state is unreadable: {self.path}: {exc}"
                ) from exc
            if not isinstance(document, dict):
                raise RecordStoreError("scheduler state must be a JSON object")
            if document.get("schema_version") != self.SCHEMA_VERSION:
                raise RecordStoreError(
                    "scheduler state schema is incompatible: "
                    f"{document.get('schema_version')!r}"
                )
            if document.get("metadata") != self._document["metadata"]:
                raise RecordStoreError("scheduler state metadata does not match the run")
            if document.get("status") not in self.VALID_STATUSES:
                raise RecordStoreError("scheduler state has an invalid status")
            records = document.get("records")
            events = document.get("events")
            if not isinstance(records, list) or not isinstance(events, list):
                raise RecordStoreError("scheduler state records/events must be arrays")
            job_ids = []
            for record in records:
                if not isinstance(record, dict):
                    raise RecordStoreError("scheduler state contains a non-object record")
                job_id = record.get("job_id")
                pod_uid = record.get("pod_uid")
                if not isinstance(job_id, str) or not job_id:
                    raise RecordStoreError("scheduler state record has no job_id")
                if not isinstance(pod_uid, str) or not pod_uid:
                    raise RecordStoreError(
                        f"scheduler state record {job_id!r} has no pod_uid"
                    )
                job_ids.append(job_id)
            if len(job_ids) != len(set(job_ids)):
                raise RecordStoreError("scheduler state contains duplicate job records")
            self._document = document
            return True

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._document["status"] = status
            self._document["error"] = error
            self._write_locked()

    def upsert(self, record: ScheduleRecord) -> None:
        data = asdict(record)
        with self._lock:
            records = self._document["records"]
            for index, existing in enumerate(records):
                if existing["job_id"] == record.job_id:
                    records[index] = data
                    break
            else:
                records.append(data)
            self._write_locked()

    def append_event(self, event: Mapping[str, object]) -> None:
        data = dict(event)
        data.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._document["events"].append(data)
            self._write_locked()

    def _write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        self._document["updated_at"] = datetime.now(timezone.utc).isoformat()
        fd, temporary = tempfile.mkstemp(
            prefix=os.path.basename(self.path) + ".", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(self._document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
