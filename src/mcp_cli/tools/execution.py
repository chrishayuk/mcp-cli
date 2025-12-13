# mcp_cli/tools/execution.py
"""Parallel and streaming tool execution utilities.

Provides async-native parallel execution with callbacks for tool calls.
Uses chuk-tool-processor's ToolCall/ToolResult models.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


async def execute_tools_parallel(
    manager: ToolManager,
    calls: list[CTPToolCall],
    timeout: float | None = None,
    on_tool_start: Callable[[CTPToolCall], Awaitable[None]] | None = None,
    on_tool_result: Callable[[CTPToolResult], Awaitable[None]] | None = None,
    max_concurrency: int = 4,
) -> list[CTPToolResult]:
    """
    Execute multiple tool calls in parallel with optional callbacks.

    Uses chuk-tool-processor's ToolCall/ToolResult models for consistency.
    Results are returned in completion order (faster tools return first).

    Args:
        manager: ToolManager instance to execute tools
        calls: List of CTPToolCall objects to execute
        timeout: Timeout per tool execution (uses default if not specified)
        on_tool_start: Async callback invoked when each tool starts
        on_tool_result: Async callback invoked when each tool completes
        max_concurrency: Maximum concurrent executions (default: 4)

    Returns:
        List of CTPToolResult objects in completion order
    """
    if not calls:
        return []

    effective_timeout = timeout or manager.tool_timeout
    sem = asyncio.Semaphore(max_concurrency)
    results: list[CTPToolResult] = []

    async def execute_single(call: CTPToolCall) -> CTPToolResult:
        """Execute a single tool call with semaphore control."""
        start_time = datetime.now(UTC)

        async with sem:
            # Invoke start callback
            if on_tool_start:
                try:
                    await on_tool_start(call)
                except Exception as e:
                    logger.warning(
                        f"on_tool_start callback failed for {call.tool}: {e}"
                    )

            # Execute the tool
            tool_result = await manager.execute_tool(
                call.tool,
                call.arguments,
                namespace=call.namespace if call.namespace != "default" else None,
                timeout=effective_timeout,
            )

            end_time = datetime.now(UTC)

            # Convert ToolCallResult to CTPToolResult
            ctp_result = CTPToolResult(
                id=call.id,
                tool=call.tool,
                result=tool_result.result if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
                start_time=start_time,
                end_time=end_time,
                machine=platform.node(),
                pid=os.getpid(),
            )

            # Invoke result callback
            if on_tool_result:
                try:
                    await on_tool_result(ctp_result)
                except Exception as e:
                    logger.warning(
                        f"on_tool_result callback failed for {call.tool}: {e}"
                    )

            return ctp_result

    # Create all tasks and execute in parallel
    tasks = [asyncio.create_task(execute_single(call)) for call in calls]

    # Use as_completed to get results in completion order
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        results.append(result)

    return results


async def stream_execute_tools(
    manager: ToolManager,
    calls: list[CTPToolCall],
    timeout: float | None = None,
    on_tool_start: Callable[[CTPToolCall], Awaitable[None]] | None = None,
    max_concurrency: int = 4,
) -> AsyncIterator[CTPToolResult]:
    """
    Execute multiple tool calls in parallel, yielding results as they complete.

    This is the streaming version of execute_tools_parallel - results are
    yielded immediately when each tool completes, without waiting for all.

    Args:
        manager: ToolManager instance to execute tools
        calls: List of CTPToolCall objects to execute
        timeout: Timeout per tool execution (uses default if not specified)
        on_tool_start: Async callback invoked when each tool starts
        max_concurrency: Maximum concurrent executions (default: 4)

    Yields:
        CTPToolResult objects as each tool completes (in completion order)
    """
    if not calls:
        return

    effective_timeout = timeout or manager.tool_timeout
    sem = asyncio.Semaphore(max_concurrency)
    queue: asyncio.Queue[CTPToolResult] = asyncio.Queue()

    async def execute_single(call: CTPToolCall) -> None:
        """Execute a single tool call and put result in queue."""
        start_time = datetime.now(UTC)

        async with sem:
            # Invoke start callback
            if on_tool_start:
                try:
                    await on_tool_start(call)
                except Exception as e:
                    logger.warning(
                        f"on_tool_start callback failed for {call.tool}: {e}"
                    )

            # Execute the tool
            tool_result = await manager.execute_tool(
                call.tool,
                call.arguments,
                namespace=call.namespace if call.namespace != "default" else None,
                timeout=effective_timeout,
            )

            end_time = datetime.now(UTC)

            # Convert ToolCallResult to CTPToolResult
            ctp_result = CTPToolResult(
                id=call.id,
                tool=call.tool,
                result=tool_result.result if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
                start_time=start_time,
                end_time=end_time,
                machine=platform.node(),
                pid=os.getpid(),
            )

            await queue.put(ctp_result)

    # Start all tasks
    tasks = {asyncio.create_task(execute_single(call)) for call in calls}

    # Yield results as they complete
    results_received = 0
    while results_received < len(calls):
        try:
            result = await queue.get()
            yield result
            results_received += 1
        except asyncio.CancelledError:
            # Cancel remaining tasks on cancellation
            for task in tasks:
                if not task.done():
                    task.cancel()
            break

    # Clean up any remaining tasks
    done, pending = await asyncio.wait(tasks, timeout=0)
    for task in pending:
        task.cancel()
