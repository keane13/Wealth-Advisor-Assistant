"""Structured logging.

Two sinks:
* ``logs/run.jsonl`` holds one JSON object per event (machine-readable, greppable by run_id).
* The console gets a short human-readable line.

Every record carries the ``run_id`` and ``agent`` from context variables, so no call
site has to pass them. Worker threads must be started with ``contextvars.copy_context()``
(see ``core.concurrency``) so they inherit the variables.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")
agent_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent", default="-")

_ROOT = "wealth_assistant"
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        record.agent = agent_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "agent": getattr(record, "agent", "-"),
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        agent = getattr(record, "agent", "-")
        event = getattr(record, "event", "") or ""
        prefix = f"{record.levelname:<7} [{agent}]"
        return f"{prefix} {event + ': ' if event else ''}{record.getMessage()}"


def configure_logging(log_dir: Path | None, level: str = "INFO", console: bool = True) -> None:
    """Configure the package logger. Safe to call more than once (handlers are replaced)."""
    logger = logging.getLogger(_ROOT)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    context_filter = _ContextFilter()
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.jsonl", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level.upper())
        stream_handler.setFormatter(ConsoleFormatter())
        stream_handler.addFilter(context_filter)
        logger.addHandler(stream_handler)


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{component}")


def log_event(logger: logging.Logger, level: int, event: str, message: str, **fields: Any) -> None:
    """Log a structured event. ``fields`` become top-level keys in the JSON log line."""
    logger.log(level, message, extra={"event": event, **fields})
