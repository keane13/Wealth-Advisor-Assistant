"""Short-term (session) memory: the shared working context for one run.

It works as a blackboard. Each agent reads its inputs from it and writes its output
back under a well-known key. The orchestrator uses the keys to check a step's
dependencies before routing to it. Every write records which agent wrote it and when,
so the run can be debugged after the fact.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class MemoryEvent:
    at: str
    key: str
    written_by: str


class SessionMemory:
    def __init__(self, max_events: int = 500) -> None:
        self._data: dict[str, Any] = {}
        self._events: deque[MemoryEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def put(self, key: str, value: Any, written_by: str) -> None:
        with self._lock:
            self._data[key] = value
            self._events.append(MemoryEvent(datetime.now(timezone.utc).isoformat(), key, written_by))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def require(self, key: str) -> Any:
        with self._lock:
            if key not in self._data:
                raise KeyError(f"session memory has no '{key}'")
            return self._data[key]

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def history(self) -> list[MemoryEvent]:
        with self._lock:
            return list(self._events)
