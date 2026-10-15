"""Exception hierarchy.

The key distinction is *retryable* vs *non-retryable*: the tool layer retries only
transient failures (timeouts, 5xx-style outages) and never retries errors that will
fail identically on the next attempt (not found, invalid data).
"""

from __future__ import annotations


class WealthAssistantError(Exception):
    """Base class for all domain errors raised by this package."""


class ToolError(WealthAssistantError):
    """A tool failed. ``retryable`` tells the tool layer whether another attempt may help."""

    retryable: bool = False

    def __init__(self, message: str, *, tool: str | None = None) -> None:
        super().__init__(message)
        self.tool = tool


class TransientToolError(ToolError):
    """Temporary failure (service unavailable, rate limited, ...). Safe to retry."""

    retryable = True


class ToolTimeoutError(TransientToolError):
    """The tool did not answer within its time budget."""


class NotFoundError(ToolError):
    """The requested entity does not exist in the source system."""


class DataValidationError(ToolError):
    """Input or upstream data is malformed beyond what can be repaired."""


class UnknownToolError(WealthAssistantError):
    """An agent asked for a tool that is not registered (a wiring/programming error)."""


class RoutingError(WealthAssistantError):
    """The orchestrator could not find exactly one agent for a task type."""
