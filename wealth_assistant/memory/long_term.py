"""Long-term memory: what the system remembers about a client across runs.

* run summaries, used to describe what changed since the last review
* the last good CRM profile, used as a fallback when the CRM is down
* findings a human reviewer dismissed as expected, so they are not flagged again

Storage is one JSON file per client: human-inspectable, trivially portable, and good
enough for a single-process assistant. Writes are atomic (temp file + ``os.replace``) so
a crash mid-write never corrupts history. A corrupt file is quarantined, not fatal.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger, log_event

logger = get_logger("memory")

MAX_SUMMARIES = 20
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _empty_record() -> dict[str, Any]:
    return {"summaries": [], "crm_cache": None, "suppressions": {}}


class LongTermMemory(ABC):
    """Storage-agnostic interface. Agents depend on this, never on the file format."""

    @abstractmethod
    def _load(self, client_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def _save(self, client_id: str, record: dict[str, Any]) -> None: ...

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # --- run summaries -------------------------------------------------------
    def last_summary(self, client_id: str) -> dict[str, Any] | None:
        summaries = self._load(client_id)["summaries"]
        return deepcopy(summaries[-1]) if summaries else None

    def summaries(self, client_id: str) -> list[dict[str, Any]]:
        return deepcopy(self._load(client_id)["summaries"])

    def append_summary(self, client_id: str, summary: dict[str, Any]) -> None:
        with self._lock:
            record = self._load(client_id)
            record["summaries"] = (record["summaries"] + [summary])[-MAX_SUMMARIES:]
            self._save(client_id, record)

    # --- CRM cache ------------------------------------------------------------
    def cache_crm_profile(self, client_id: str, profile: dict[str, Any]) -> None:
        with self._lock:
            record = self._load(client_id)
            record["crm_cache"] = {"profile": profile, "cached_at": _now()}
            self._save(client_id, record)

    def cached_crm_profile(self, client_id: str) -> dict[str, Any] | None:
        return deepcopy(self._load(client_id)["crm_cache"])

    # --- reviewer suppressions ----------------------------------------------------
    def suppress(self, client_id: str, fingerprint: str, reason: str, reviewer: str) -> None:
        with self._lock:
            record = self._load(client_id)
            record["suppressions"][fingerprint] = {"reason": reason, "reviewer": reviewer, "at": _now()}
            self._save(client_id, record)

    def suppressions(self, client_id: str) -> dict[str, dict[str, Any]]:
        return deepcopy(self._load(client_id)["suppressions"])

    def clear(self, client_id: str) -> None:
        with self._lock:
            self._save(client_id, _empty_record())


class InMemoryLongTermMemory(LongTermMemory):
    def __init__(self) -> None:
        super().__init__()
        self._records: dict[str, dict[str, Any]] = {}

    def _load(self, client_id: str) -> dict[str, Any]:
        return deepcopy(self._records.get(client_id, _empty_record()))

    def _save(self, client_id: str, record: dict[str, Any]) -> None:
        self._records[client_id] = deepcopy(record)


class JsonFileMemoryStore(LongTermMemory):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, client_id: str) -> Path:
        if not _SAFE_ID.match(client_id):
            raise ValueError(f"unsafe client id for memory path: {client_id!r}")
        return self.directory / f"{client_id}.json"

    def _load(self, client_id: str) -> dict[str, Any]:
        path = self._path(client_id)
        if not path.exists():
            return _empty_record()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("memory root is not an object")
            record = _empty_record()
            record.update({k: data[k] for k in record if k in data})
            return record
        except (ValueError, OSError) as exc:
            quarantine = path.with_suffix(f".corrupt-{datetime.now():%Y%m%d%H%M%S}.json")
            try:
                path.replace(quarantine)
            except OSError:
                pass
            log_event(logger, logging.ERROR, "memory.corrupt",
                      f"long-term memory for {client_id} unreadable ({exc}); quarantined to {quarantine.name}",
                      client_id=client_id)
            return _empty_record()

    def _save(self, client_id: str, record: dict[str, Any]) -> None:
        path = self._path(client_id)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=f".{client_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, default=str)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
