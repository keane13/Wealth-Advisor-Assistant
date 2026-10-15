"""Thread helpers that preserve logging context (run_id, agent) across threads."""

from __future__ import annotations

import contextvars
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Literal, TypeVar

T = TypeVar("T")

# Tool calls are I/O-bound (simulated network calls, file reads), so threads are a good
# fit without bringing in asyncio. There are two pools: agents fan out work on the
# "task" pool, and tool timeouts run on the "tool" pool. A task waiting on a tool can
# therefore never take the thread that tool needs, which rules out nested-pool deadlock.
_POOLS = {
    "task": ThreadPoolExecutor(max_workers=8, thread_name_prefix="wa-task"),
    "tool": ThreadPoolExecutor(max_workers=8, thread_name_prefix="wa-tool"),
}


def submit_with_context(fn: Callable[..., T], *args: Any, pool: Literal["task", "tool"] = "task",
                        **kwargs: Any) -> Future[T]:
    ctx = contextvars.copy_context()
    return _POOLS[pool].submit(ctx.run, fn, *args, **kwargs)
