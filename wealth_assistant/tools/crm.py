"""CRM integration.

``MockCRMService`` stands in for a remote REST API. It has realistic failure modes
(latency, random 503s, full outages, 404s, malformed payloads) so the resilience path
can be exercised. ``CRMProfileTool`` is the adapter agents see: it turns HTTP-style
errors into the ``ToolError`` hierarchy, so retry and fallback decisions are made on
error *semantics* and not on status codes.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from ..core.errors import DataValidationError, NotFoundError, ToolError, TransientToolError
from ..core.tools import Tool
from ..memory.long_term import LongTermMemory


class CRMHttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class MockCRMService:
    def __init__(
        self,
        profiles_path: Path,
        *,
        failure_rate: float = 0.0,
        outage: bool = False,
        latency_s: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.profiles_path = profiles_path
        self.failure_rate = max(0.0, min(1.0, failure_rate))
        self.outage = outage
        self.latency_s = max(0.0, latency_s)
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()

    def get_profile(self, client_id: str) -> dict[str, Any]:
        time.sleep(self.latency_s)
        if self.outage:
            raise CRMHttpError(503, "CRM unavailable (simulated outage)")
        with self._rng_lock:
            flaky = self._rng.random() < self.failure_rate
        if flaky:
            raise CRMHttpError(503, "CRM temporarily unavailable (simulated)")
        try:
            profiles = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CRMHttpError(500, f"CRM backend error: {exc}") from exc
        if client_id not in profiles:
            raise CRMHttpError(404, f"client {client_id} not found in CRM")
        return profiles[client_id]


class CRMProfileTool(Tool):
    name = "crm.get_profile"
    description = "Fetch the client's CRM profile (risk tolerance, goals, KYC status, advisor notes)."

    def __init__(self, service: MockCRMService) -> None:
        self.service = service

    def run(self, *, client_id: str) -> dict[str, Any]:
        try:
            payload = self.service.get_profile(client_id)
        except CRMHttpError as exc:
            if exc.status == 404:
                raise NotFoundError(str(exc), tool=self.name) from exc
            if exc.status == 429 or exc.status >= 500:
                raise TransientToolError(str(exc), tool=self.name) from exc
            raise ToolError(str(exc), tool=self.name) from exc
        if not isinstance(payload, dict):
            raise DataValidationError("CRM returned a non-object payload", tool=self.name)
        return {"profile": payload, "source": "crm", "cached_at": None}


class CachedCRMProfileTool(Tool):
    """Fallback for ``crm.get_profile``: the last good profile from long-term memory."""

    name = "crm.cached_profile"
    description = "Last successfully fetched CRM profile, served from long-term memory."

    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory

    def run(self, *, client_id: str) -> dict[str, Any]:
        cached = self.memory.cached_crm_profile(client_id)
        if not cached:
            raise NotFoundError(f"no cached CRM profile for {client_id}", tool=self.name)
        return {"profile": cached["profile"], "source": "cache", "cached_at": cached["cached_at"]}
