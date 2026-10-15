"""Tool abstraction.

Agents never import or construct concrete tools. They ask the ``ToolRegistry`` for a
capability by name (``tools.call("crm.get_profile", client_id=...)``) and get back a
``ToolResult`` that never raises. Resilience concerns live here, in one place, rather
than being re-implemented in every agent:

* timeout per call
* bounded retries with exponential backoff, only for ``TransientToolError``
* an optional fallback tool when the primary is exhausted (the result is marked ``degraded``)
* structured logging of every attempt
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from ..logging_setup import get_logger, log_event
from .concurrency import submit_with_context
from .errors import ToolError, ToolTimeoutError, UnknownToolError

logger = get_logger("tools")


class Tool(ABC):
    """A single capability with a stable name. Implementations raise ``ToolError`` subclasses."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Must raise ``ToolError`` (or a subclass) on expected failures."""


@dataclass(frozen=True)
class ToolPolicy:
    max_attempts: int = 1
    backoff_base_s: float = 0.2
    timeout_s: float | None = None
    fallback: str | None = None
    # By default a fallback only covers *transient* failures (outage, timeout). A
    # permanent answer such as "client not found" is real information, and masking it
    # with stale cached data would be wrong.
    fallback_on_permanent: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    error: str | None = None
    error_type: str | None = None
    attempts: int = 1
    duration_ms: float = 0.0
    degraded: bool = False
    served_by: str | None = None  # differs from ``tool`` when a fallback answered

    def unwrap(self) -> Any:
        if not self.ok:
            raise ToolError(self.error or "tool failed", tool=self.tool)
        return self.data


@dataclass
class _Registration:
    name: str
    tool: Tool
    policy: ToolPolicy


@dataclass
class ToolRegistry:
    sleep: Callable[[float], None] = time.sleep
    _tools: dict[str, _Registration] = field(default_factory=dict)

    def register(self, tool: Tool, policy: ToolPolicy | None = None, *, name: str | None = None) -> None:
        """Register ``tool`` under its own name, or under ``name`` to bind an implementation
        to a capability (e.g. the template narrative serving ``report.narrative``)."""
        key = name or tool.name
        if not key:
            raise ValueError(f"{type(tool).__name__} has no name")
        if key in self._tools:
            raise ValueError(f"tool '{key}' already registered")
        self._tools[key] = _Registration(key, tool, policy or ToolPolicy())

    def has(self, name: str) -> bool:
        return name in self._tools

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": n, "implementation": type(r.tool).__name__, "description": r.tool.description,
             "fallback": r.policy.fallback}
            for n, r in sorted(self._tools.items())
        ]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        return self._call(name, kwargs, visited=())

    # ------------------------------------------------------------------ internals

    def _call(self, name: str, kwargs: dict[str, Any], visited: tuple[str, ...]) -> ToolResult:
        if name not in self._tools:
            raise UnknownToolError(f"tool '{name}' is not registered")
        if name in visited:  # guards against fallback cycles (a -> b -> a)
            return ToolResult(tool=name, ok=False, error="fallback cycle detected", error_type="FallbackCycle")

        reg = self._tools[name]
        result, retryable = self._call_with_retries(reg, kwargs)
        if result.ok or not reg.policy.fallback:
            return result
        if not retryable and not reg.policy.fallback_on_permanent:
            log_event(logger, logging.INFO, "tool.no_fallback",
                      f"{name} failed permanently ({result.error_type}); fallback not applicable",
                      tool=name, error_type=result.error_type)
            return result

        log_event(
            logger, logging.WARNING, "tool.fallback",
            f"{name} failed ({result.error_type}); falling back to {reg.policy.fallback}",
            tool=name, fallback=reg.policy.fallback, error=result.error,
        )
        fb = self._call(reg.policy.fallback, kwargs, visited + (name,))
        return ToolResult(
            tool=name,
            ok=fb.ok,
            data=fb.data,
            error=fb.error if not fb.ok else result.error,
            error_type=fb.error_type if not fb.ok else result.error_type,
            attempts=result.attempts + fb.attempts,
            duration_ms=result.duration_ms + fb.duration_ms,
            degraded=True,
            served_by=fb.served_by or fb.tool,
        )

    def _call_with_retries(self, reg: _Registration, kwargs: dict[str, Any]) -> tuple[ToolResult, bool]:
        """Returns the result and whether the final failure (if any) was transient."""
        name, tool, policy = reg.name, reg.tool, reg.policy
        started = time.perf_counter()
        last_exc: BaseException | None = None

        for attempt in range(1, policy.max_attempts + 1):
            log_event(logger, logging.DEBUG, "tool.call", f"{name} attempt {attempt}",
                      tool=name, attempt=attempt, tool_args=_safe_args(kwargs))
            try:
                data = self._execute(name, tool, kwargs, policy.timeout_s)
                elapsed = (time.perf_counter() - started) * 1000
                log_event(logger, logging.DEBUG, "tool.success", f"{name} ok in {elapsed:.0f}ms",
                          tool=name, attempt=attempt, duration_ms=round(elapsed, 1))
                return ToolResult(tool=name, ok=True, data=data, attempts=attempt,
                                  duration_ms=elapsed, served_by=name), False
            except ToolError as exc:
                last_exc = exc
                if exc.retryable and attempt < policy.max_attempts:
                    delay = policy.backoff_base_s * (2 ** (attempt - 1))
                    log_event(logger, logging.WARNING, "tool.retry",
                              f"{name} transient failure: {exc}; retrying in {delay:.2f}s",
                              tool=name, attempt=attempt, error=str(exc))
                    self.sleep(delay)
                    continue
                break
            except Exception as exc:  # unexpected bug inside a tool: contain it, never retry
                last_exc = exc
                logger.exception("tool %s raised an unexpected error", name,
                                 extra={"event": "tool.crash", "tool": name})
                break

        elapsed = (time.perf_counter() - started) * 1000
        log_event(logger, logging.ERROR if not policy.fallback else logging.WARNING, "tool.failure",
                  f"{name} failed after {attempt} attempt(s): {last_exc}",
                  tool=name, attempts=attempt, error_type=type(last_exc).__name__)
        retryable = isinstance(last_exc, ToolError) and last_exc.retryable
        return ToolResult(tool=name, ok=False, error=str(last_exc), error_type=type(last_exc).__name__,
                          attempts=attempt, duration_ms=elapsed), retryable

    @staticmethod
    def _execute(name: str, tool: Tool, kwargs: dict[str, Any], timeout_s: float | None) -> Any:
        if timeout_s is None:
            return tool.run(**kwargs)
        future = submit_with_context(tool.run, pool="tool", **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout:
            # The worker thread cannot be killed; it finishes in the background and its
            # result is discarded. Acceptable for idempotent reads, which all our tools are.
            future.cancel()
            raise ToolTimeoutError(f"timed out after {timeout_s:.2f}s", tool=name) from None


def _safe_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Log scalar arguments only: large payloads (transaction lists) would flood the logs."""
    return {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
