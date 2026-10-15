"""Per-run execution context handed to every agent.

It is the only way agents reach the outside world: tools, memory and settings are all
injected here. That keeps agents unit-testable with fakes and free of global state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..config import Settings
from ..memory.long_term import LongTermMemory
from ..memory.short_term import SessionMemory
from .tools import ToolRegistry


@dataclass
class RunContext:
    client_id: str
    tools: ToolRegistry
    long_term: LongTermMemory
    settings: Settings
    session: SessionMemory = field(default_factory=SessionMemory)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
