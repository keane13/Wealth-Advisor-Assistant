from __future__ import annotations

from pathlib import Path

import pytest

from wealth_assistant.config import PROJECT_ROOT, Settings
from wealth_assistant.logging_setup import configure_logging
from wealth_assistant.memory.long_term import InMemoryLongTermMemory


@pytest.fixture(autouse=True)
def _quiet_logging():
    configure_logging(None, console=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=PROJECT_ROOT / "data",
        memory_dir=tmp_path / "memory",
        log_dir=tmp_path / "logs",
        output_dir=tmp_path / "out",
        crm_latency_s=0.0,
        random_seed=7,
    )


@pytest.fixture
def memory() -> InMemoryLongTermMemory:
    return InMemoryLongTermMemory()
