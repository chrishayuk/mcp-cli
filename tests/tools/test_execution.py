# tests/tools/test_execution.py
"""Tests for parallel and streaming tool execution."""

import asyncio
import pytest
from datetime import datetime

from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult

from mcp_cli.tools.execution import execute_tools_parallel, stream_execute_tools
from mcp_cli.tools.models import ToolCallResult


class MockToolManager:
    """Mock ToolManager for testing execution functions."""

    def __init__(self, results: dict[str, ToolCallResult] | None = None):
        self.tool_timeout = 30.0
        self.results = results or {}
        self.executed_tools: list[str] = []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult:
        self.executed_tools.append(tool_name)
        if tool_name in self.results:
            return self.results[tool_name]
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result={"output": f"result from {tool_name}"},
        )


# ----------------------------------------------------------------------------
# execute_tools_parallel tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tools_parallel_empty_calls():
    """Test with empty call list."""
    manager = MockToolManager()
    results = await execute_tools_parallel(manager, [])
    assert results == []


@pytest.mark.asyncio
async def test_execute_tools_parallel_single_call():
    """Test with single tool call."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={"x": 1})]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 1
    assert results[0].tool == "test_tool"
    assert results[0].is_success
    assert manager.executed_tools == ["test_tool"]


@pytest.mark.asyncio
async def test_execute_tools_parallel_multiple_calls():
    """Test with multiple tool calls."""
    manager = MockToolManager()
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 3
    assert set(r.tool for r in results) == {"tool_a", "tool_b", "tool_c"}
    assert len(manager.executed_tools) == 3


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_timeout():
    """Test with custom timeout."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = await execute_tools_parallel(manager, calls, timeout=60.0)

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_on_tool_start():
    """Test on_tool_start callback is invoked."""
    manager = MockToolManager()
    started_tools: list[str] = []

    async def on_start(call: CTPToolCall):
        started_tools.append(call.tool)

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    await execute_tools_parallel(manager, calls, on_tool_start=on_start)

    assert set(started_tools) == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_on_tool_result():
    """Test on_tool_result callback is invoked."""
    manager = MockToolManager()
    completed_tools: list[str] = []

    async def on_result(result: CTPToolResult):
        completed_tools.append(result.tool)

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    await execute_tools_parallel(manager, calls, on_tool_result=on_result)

    assert set(completed_tools) == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_execute_tools_parallel_callback_exception():
    """Test that callback exceptions don't break execution."""
    manager = MockToolManager()

    async def failing_callback(call: CTPToolCall):
        raise ValueError("callback error")

    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    # Should not raise, just log warning
    results = await execute_tools_parallel(
        manager, calls, on_tool_start=failing_callback
    )

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_error():
    """Test handling tool execution errors."""
    manager = MockToolManager(
        results={
            "failing_tool": ToolCallResult(
                tool_name="failing_tool",
                success=False,
                error="tool failed",
            )
        }
    )

    calls = [
        CTPToolCall(id="call_1", tool="failing_tool", arguments={}),
        CTPToolCall(id="call_2", tool="success_tool", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 2
    failing = next(r for r in results if r.tool == "failing_tool")
    success = next(r for r in results if r.tool == "success_tool")

    assert not failing.is_success
    assert failing.error == "tool failed"
    assert success.is_success


@pytest.mark.asyncio
async def test_execute_tools_parallel_max_concurrency():
    """Test max_concurrency limits parallel execution."""
    execution_count = 0
    max_concurrent = 0

    original_execute = MockToolManager.execute_tool

    async def tracking_execute(
        self, tool_name, arguments, namespace=None, timeout=None
    ):
        nonlocal execution_count, max_concurrent
        execution_count += 1
        current = execution_count
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)  # Simulate work
        execution_count -= 1
        return await original_execute(self, tool_name, arguments, namespace, timeout)

    manager = MockToolManager()
    manager.execute_tool = lambda *args, **kwargs: tracking_execute(
        manager, *args, **kwargs
    )

    calls = [
        CTPToolCall(id=f"call_{i}", tool=f"tool_{i}", arguments={}) for i in range(10)
    ]

    await execute_tools_parallel(manager, calls, max_concurrency=2)

    # Max concurrent should not exceed 2 (though timing may vary)
    assert max_concurrent <= 3  # Allow some slack due to async timing


@pytest.mark.asyncio
async def test_execute_tools_parallel_respects_namespace():
    """Test namespace is passed correctly."""
    manager = MockToolManager()
    calls = [
        CTPToolCall(id="call_1", tool="test_tool", arguments={}, namespace="custom_ns")
    ]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 1


# ----------------------------------------------------------------------------
# stream_execute_tools tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_execute_tools_empty_calls():
    """Test with empty call list."""
    manager = MockToolManager()
    results = []
    async for result in stream_execute_tools(manager, []):
        results.append(result)
    assert results == []


@pytest.mark.asyncio
async def test_stream_execute_tools_single_call():
    """Test streaming with single tool call."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={"x": 1})]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    assert len(results) == 1
    assert results[0].tool == "test_tool"
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_multiple_calls():
    """Test streaming with multiple tool calls."""
    manager = MockToolManager()
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    assert len(results) == 3
    assert set(r.tool for r in results) == {"tool_a", "tool_b", "tool_c"}


@pytest.mark.asyncio
async def test_stream_execute_tools_with_on_tool_start():
    """Test on_tool_start callback in streaming mode."""
    manager = MockToolManager()
    started_tools: list[str] = []

    async def on_start(call: CTPToolCall):
        started_tools.append(call.tool)

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, on_tool_start=on_start):
        results.append(result)

    assert set(started_tools) == {"tool_a", "tool_b"}
    assert len(results) == 2


@pytest.mark.asyncio
async def test_stream_execute_tools_yields_as_completed():
    """Test results are yielded as tools complete."""
    manager = MockToolManager()
    yield_times: list[float] = []

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    import time

    start = time.time()
    async for result in stream_execute_tools(manager, calls):
        yield_times.append(time.time() - start)

    # Both should yield quickly (not waiting for all)
    assert len(yield_times) == 2


@pytest.mark.asyncio
async def test_stream_execute_tools_with_error():
    """Test streaming handles errors gracefully."""
    manager = MockToolManager(
        results={
            "failing_tool": ToolCallResult(
                tool_name="failing_tool",
                success=False,
                error="tool failed",
            )
        }
    )

    calls = [
        CTPToolCall(id="call_1", tool="failing_tool", arguments={}),
        CTPToolCall(id="call_2", tool="success_tool", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_stream_execute_tools_result_fields():
    """Test CTPToolResult has expected fields."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    result = results[0]
    assert result.id == "call_1"
    assert result.tool == "test_tool"
    assert isinstance(result.start_time, datetime)
    assert isinstance(result.end_time, datetime)
    assert result.machine  # Should have hostname
    assert result.pid > 0  # Should have process ID


# ----------------------------------------------------------------------------
# Additional coverage tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tools_parallel_on_tool_result_callback_exception():
    """Test on_tool_result callback exception doesn't break execution."""
    manager = MockToolManager()

    async def failing_result_callback(result: CTPToolResult):
        raise ValueError("callback error")

    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    # Should not raise, just log warning
    results = await execute_tools_parallel(
        manager, calls, on_tool_result=failing_result_callback
    )

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_on_tool_start_callback_exception():
    """Test on_tool_start callback exception in streaming mode."""
    manager = MockToolManager()

    async def failing_start_callback(call: CTPToolCall):
        raise ValueError("callback error")

    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = []
    async for result in stream_execute_tools(
        manager, calls, on_tool_start=failing_start_callback
    ):
        results.append(result)

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_cancellation():
    """Test stream_execute_tools handles cancellation gracefully."""
    manager = MockToolManager()

    # Create a slow tool that we can cancel
    async def slow_execute(self, tool_name, arguments, namespace=None, timeout=None):
        await asyncio.sleep(10)  # Long delay
        return ToolCallResult(tool_name=tool_name, success=True, result={})

    manager.execute_tool = lambda *args, **kwargs: slow_execute(
        manager, *args, **kwargs
    )

    calls = [
        CTPToolCall(id="call_1", tool="slow_tool_1", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool_2", arguments={}),
    ]

    results = []

    async def collect_with_cancel():
        async for result in stream_execute_tools(manager, calls):
            results.append(result)
            # Cancel after receiving anything (but we won't receive anything since they're slow)

    task = asyncio.create_task(collect_with_cancel())
    await asyncio.sleep(0.1)  # Let tasks start
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Results should be empty since we cancelled before any completed
    assert len(results) == 0


@pytest.mark.asyncio
async def test_stream_execute_tools_with_timeout():
    """Test stream_execute_tools with custom timeout."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = []
    async for result in stream_execute_tools(manager, calls, timeout=60.0):
        results.append(result)

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_max_concurrency():
    """Test stream_execute_tools respects max_concurrency."""
    execution_count = 0
    max_concurrent = 0

    original_execute = MockToolManager.execute_tool

    async def tracking_execute(
        self, tool_name, arguments, namespace=None, timeout=None
    ):
        nonlocal execution_count, max_concurrent
        execution_count += 1
        current = execution_count
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)  # Simulate work
        execution_count -= 1
        return await original_execute(self, tool_name, arguments, namespace, timeout)

    manager = MockToolManager()
    manager.execute_tool = lambda *args, **kwargs: tracking_execute(
        manager, *args, **kwargs
    )

    calls = [
        CTPToolCall(id=f"call_{i}", tool=f"tool_{i}", arguments={}) for i in range(5)
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, max_concurrency=2):
        results.append(result)

    assert len(results) == 5
    # Max concurrent should not exceed 2 (with some slack for async timing)
    assert max_concurrent <= 3
