# src/mcp_cli/utils/async_utils.py
"""
Tiny helper for “run an async coroutine from possibly-sync code”.

* If no event-loop exists → `asyncio.run`.
* If a loop exists but is **not** running → `loop.run_until_complete`.
* If called **inside** a running loop → we raise, so callers know to
  switch to the `*_async` variant instead of silently returning junk.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


def run_blocking(coro: Awaitable[T]) -> T:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # totally sync context
        # asyncio.run expects a coroutine, not just any awaitable
        if asyncio.iscoroutine(coro):
            coro_result: T = asyncio.run(coro)
            return coro_result
        else:
            # If it's not a coroutine, wrap it
            async def _wrapper() -> T:
                return await coro

            wrapped_result: T = asyncio.run(_wrapper())
            return wrapped_result

    if loop.is_running():
        raise RuntimeError(
            "run_blocking() called inside a running event-loop - "
            "use the async API instead."
        )
    return loop.run_until_complete(coro)
