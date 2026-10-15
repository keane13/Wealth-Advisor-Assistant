"""The tool layer: retries, timeouts, fallbacks and error containment."""

from __future__ import annotations

import time

import pytest

from wealth_assistant.core.errors import NotFoundError, ToolError, TransientToolError, UnknownToolError
from wealth_assistant.core.tools import Tool, ToolPolicy, ToolRegistry


class Flaky(Tool):
    name = "flaky"

    def __init__(self, failures: int, exc: type[ToolError] = TransientToolError) -> None:
        self.failures, self.exc, self.calls = failures, exc, 0

    def run(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("boom")
        return {"ok": True, **kwargs}


class Constant(Tool):
    def __init__(self, name: str, value) -> None:
        self.name, self.value = name, value

    def run(self, **kwargs):
        return self.value


class Slow(Tool):
    name = "slow"

    def run(self, **kwargs):
        time.sleep(0.5)
        return "late"


class Buggy(Tool):
    name = "buggy"

    def run(self, **kwargs):
        raise ZeroDivisionError("bug")


@pytest.fixture
def registry():
    return ToolRegistry(sleep=lambda s: None)


def test_transient_errors_are_retried_until_success(registry):
    tool = Flaky(failures=2)
    registry.register(tool, ToolPolicy(max_attempts=3))
    result = registry.call("flaky", x=1)
    assert result.ok and result.data == {"ok": True, "x": 1}
    assert result.attempts == 3 and tool.calls == 3


def test_permanent_errors_are_not_retried(registry):
    tool = Flaky(failures=5, exc=NotFoundError)
    registry.register(tool, ToolPolicy(max_attempts=3))
    result = registry.call("flaky")
    assert not result.ok and result.error_type == "NotFoundError"
    assert tool.calls == 1


def test_fallback_serves_result_and_marks_it_degraded(registry):
    registry.register(Flaky(failures=99), ToolPolicy(max_attempts=2, fallback="backup"))
    registry.register(Constant("backup", "cached"))
    result = registry.call("flaky")
    assert result.ok and result.data == "cached"
    assert result.degraded and result.served_by == "backup"
    assert result.attempts == 3


def test_fallback_is_not_used_for_permanent_errors_by_default(registry):
    registry.register(Flaky(failures=99, exc=NotFoundError), ToolPolicy(fallback="backup"))
    registry.register(Constant("backup", "cached"))
    result = registry.call("flaky")
    assert not result.ok and result.error_type == "NotFoundError"


def test_fallback_on_permanent_can_be_opted_into(registry):
    registry.register(Flaky(failures=99, exc=ToolError), ToolPolicy(fallback="backup", fallback_on_permanent=True))
    registry.register(Constant("backup", "template"))
    assert registry.call("flaky").data == "template"


def test_timeout_is_a_transient_failure(registry):
    registry.register(Slow(), ToolPolicy(timeout_s=0.05, max_attempts=1))
    result = registry.call("slow")
    assert not result.ok and result.error_type == "ToolTimeoutError"


def test_unexpected_exceptions_are_contained(registry):
    registry.register(Buggy(), ToolPolicy(max_attempts=3))
    result = registry.call("buggy")
    assert not result.ok and result.error_type == "ZeroDivisionError" and result.attempts == 1


def test_unknown_tool_is_a_programming_error(registry):
    with pytest.raises(UnknownToolError):
        registry.call("nope")


def test_duplicate_registration_rejected_and_alias_supported(registry):
    registry.register(Constant("a", 1))
    with pytest.raises(ValueError):
        registry.register(Constant("a", 2))
    registry.register(Constant("a", 3), name="capability.a")
    assert registry.call("capability.a").data == 3


def test_fallback_cycles_terminate(registry):
    registry.register(Flaky(failures=99), ToolPolicy(fallback="other"))
    other = Flaky(failures=99)
    other.name = "other"
    registry.register(other, ToolPolicy(fallback="flaky"))
    assert not registry.call("flaky").ok
