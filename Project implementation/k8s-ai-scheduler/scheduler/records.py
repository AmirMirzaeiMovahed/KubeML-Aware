"""Crash-safe, incremental schedule records."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional


@dataclass
class ScheduleRecord:
    job_id: str
    order: int
    rank: float
    status: str = "ranked"
    bind_time: Optional[float] = None
    release_time: Optional[float] = None
    exec_start_time: Optional[float] = None
    error: Optional[str] = None


class AtomicRecordStore:
    SCHEMA_VERSION = 2

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

    def initialize(self) -> None:
        self._write()

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
