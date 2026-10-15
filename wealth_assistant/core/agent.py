"""Agent abstraction.

An agent declares the task types it can handle (``capabilities``) and implements
``_run``. The base class is a template method. It owns the cross-cutting concerns
every agent needs (logging context, timing, and containing unexpected exceptions) so
concrete agents contain only domain logic.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from ..logging_setup import agent_var, get_logger, log_event
from .context import RunContext


class AgentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"    # produced output but some sub-step degraded or failed
    PENDING = "pending"    # waiting on something external (e.g. a human reviewer)
    SKIPPED = "skipped"    # nothing to do, or a dependency was missing
    FAILED = "failed"      # produced no usable output


@dataclass(frozen=True)
class AgentTask:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent: str
    task: str
    status: AgentStatus
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "agent": self.agent,
            "task": self.task,
            "status": self.status.value,
            "summary": self.summary,
            "warnings": self.warnings,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 1),
        }


class BaseAgent(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]

    def __init__(self) -> None:
        self.logger = get_logger(f"agents.{self.name}")

    def handle(self, task: AgentTask, ctx: RunContext) -> AgentResult:
        token = agent_var.set(self.name)
        started = time.perf_counter()
        log_event(self.logger, logging.INFO, "agent.start", f"handling task '{task.type}'", task=task.type)
        try:
            try:
                result = self._run(task, ctx)
            except Exception as exc:  # last line of defence: one agent's bug must not crash the run
                self.logger.exception("unhandled error in %s", self.name, extra={"event": "agent.crash"})
                result = AgentResult(self.name, task.type, AgentStatus.FAILED,
                                     summary="agent crashed", errors=[f"{type(exc).__name__}: {exc}"])
            result.duration_ms = (time.perf_counter() - started) * 1000
            level = logging.ERROR if result.status is AgentStatus.FAILED else logging.INFO
            log_event(self.logger, level, "agent.finish",
                      f"{task.type} -> {result.status.value} ({result.duration_ms:.0f}ms) {result.summary}",
                      task=task.type, status=result.status.value,
                      warnings=len(result.warnings), errors=result.errors)
            return result
        finally:
            agent_var.reset(token)

    def result(self, task: AgentTask, status: AgentStatus, summary: str = "",
               warnings: list[str] | None = None, errors: list[str] | None = None) -> AgentResult:
        return AgentResult(self.name, task.type, status, summary, warnings or [], errors or [])

    def decision(self, message: str, **fields: Any) -> None:
        """Log a decision the agent made. Decisions are what an auditor asks about first."""
        log_event(self.logger, logging.INFO, "decision", message, **fields)

    @abstractmethod
    def _run(self, task: AgentTask, ctx: RunContext) -> AgentResult: ...
