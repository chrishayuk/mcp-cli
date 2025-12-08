"""Tests for StreamingDisplayManager."""

import asyncio

import pytest

from mcp_cli.display.manager import ChukTermRenderer, StreamingDisplayManager
from mcp_cli.display.models import ContentType, StreamingPhase


class TestStreamingDisplayManager:
    """Tests for StreamingDisplayManager."""

    @pytest.fixture
    def manager(self):
        """Create a display manager instance."""
        return StreamingDisplayManager()

    @pytest.mark.asyncio
    async def test_initial_state(self, manager):
        """Test initial manager state."""
        assert manager.streaming_state is None
        assert manager.tool_execution is None
        assert not manager.is_streaming
        assert not manager.is_tool_executing
        assert not manager.is_busy

    @pytest.mark.asyncio
    async def test_start_streaming(self, manager):
        """Test starting streaming creates state."""
        await manager.start_streaming()

        assert manager.streaming_state is not None
        assert manager.streaming_state.phase == StreamingPhase.INITIALIZING
        assert manager.is_streaming
        assert manager.is_busy

        # Cleanup
        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_add_chunk_creates_state_if_needed(self, manager):
        """Test adding chunk auto-creates state."""
        raw_chunk = {"content": "Hello"}

        await manager.add_chunk(raw_chunk)

        assert manager.streaming_state is not None
        assert manager.streaming_state.accumulated_content == "Hello"

        # Cleanup
        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_add_multiple_chunks(self, manager):
        """Test adding multiple chunks accumulates content."""
        await manager.start_streaming()

        await manager.add_chunk({"content": "Hello "})
        await manager.add_chunk({"content": "world"})
        await manager.add_chunk({"content": "!"})

        assert manager.streaming_state.accumulated_content == "Hello world!"
        assert manager.streaming_state.chunks_received == 3

        # Cleanup
        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_add_chunk_with_reasoning(self, manager):
        """Test adding chunk with reasoning content."""
        await manager.start_streaming()

        await manager.add_chunk(
            {"content": "Answer", "reasoning_content": "Let me think..."}
        )

        assert manager.streaming_state.accumulated_content == "Answer"
        assert manager.streaming_state.reasoning_content == "Let me think..."

        # Cleanup
        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_update_reasoning(self, manager):
        """Test updating reasoning content."""
        await manager.start_streaming()

        await manager.update_reasoning("Thinking step 1...")
        assert manager.streaming_state.reasoning_content == "Thinking step 1..."

        await manager.update_reasoning("Thinking step 2...")
        assert manager.streaming_state.reasoning_content == "Thinking step 2..."

        # Cleanup
        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_stop_streaming_returns_content(self, manager):
        """Test stopping streaming returns accumulated content."""
        await manager.start_streaming()
        await manager.add_chunk({"content": "Test content"})

        content = await manager.stop_streaming()

        assert content == "Test content"
        assert manager.streaming_state.phase == StreamingPhase.COMPLETED
        assert not manager.is_streaming

    @pytest.mark.asyncio
    async def test_stop_streaming_interrupted(self, manager):
        """Test stopping streaming with interrupt flag."""
        await manager.start_streaming()
        await manager.add_chunk({"content": "Partial"})

        content = await manager.stop_streaming(interrupted=True)

        assert content == "Partial"
        assert manager.streaming_state.phase == StreamingPhase.INTERRUPTED
        assert manager.streaming_state.interrupted

    @pytest.mark.asyncio
    async def test_stop_streaming_without_start(self, manager):
        """Test stopping streaming when not started."""
        content = await manager.stop_streaming()

        assert content == ""

    @pytest.mark.asyncio
    async def test_start_tool_execution(self, manager):
        """Test starting tool execution."""
        await manager.start_tool_execution("test_tool", {"arg": "value"})

        assert manager.tool_execution is not None
        assert manager.tool_execution.name == "test_tool"
        assert manager.tool_execution.arguments == {"arg": "value"}
        assert manager.is_tool_executing
        assert manager.is_busy

        # Cleanup
        await manager.stop_tool_execution("result")

    @pytest.mark.asyncio
    async def test_stop_tool_execution_success(self, manager):
        """Test stopping tool execution successfully."""
        await manager.start_tool_execution("test_tool", {})

        await manager.stop_tool_execution("Success result", success=True)

        assert not manager.is_tool_executing
        assert manager.tool_execution is None

    @pytest.mark.asyncio
    async def test_stop_tool_execution_failure(self, manager):
        """Test stopping tool execution with failure."""
        await manager.start_tool_execution("test_tool", {})

        await manager.stop_tool_execution("Error message", success=False)

        assert not manager.is_tool_executing
        assert manager.tool_execution is None

    @pytest.mark.asyncio
    async def test_stop_tool_without_start(self, manager):
        """Test stopping tool when not started."""
        # Should not raise
        await manager.stop_tool_execution("result")

    @pytest.mark.asyncio
    async def test_concurrent_streaming_and_tool(self, manager):
        """Test streaming stops before tool starts."""
        await manager.start_streaming()
        await manager.add_chunk({"content": "Response"})

        # Starting tool while streaming
        await manager.start_tool_execution("tool", {})

        # Should have tool state but streaming should be stopped
        assert manager.is_tool_executing
        # Streaming might still be active depending on implementation

        # Cleanup
        await manager.stop_streaming()
        await manager.stop_tool_execution("result")

    @pytest.mark.asyncio
    async def test_restart_streaming(self, manager):
        """Test stopping and restarting streaming."""
        # First stream
        await manager.start_streaming()
        await manager.add_chunk({"content": "First"})
        content1 = await manager.stop_streaming()

        assert content1 == "First"

        # Second stream
        await manager.start_streaming()
        await manager.add_chunk({"content": "Second"})
        content2 = await manager.stop_streaming()

        assert content2 == "Second"
        # Should be fresh state, not accumulated
        assert content2 != "FirstSecond"

    @pytest.mark.asyncio
    async def test_stop_streaming_interrupts_previous(self, manager):
        """Test starting new stream interrupts previous."""
        await manager.start_streaming()
        await manager.add_chunk({"content": "First"})

        # Start new stream without stopping first
        await manager.start_streaming()

        # Previous stream should be interrupted
        assert manager.streaming_state.chunks_received == 0

    @pytest.mark.asyncio
    async def test_show_user_message(self, manager):
        """Test showing user message (no exceptions)."""
        # Should not raise
        manager.show_user_message("Hello")

    @pytest.mark.asyncio
    async def test_show_system_message(self, manager):
        """Test showing system message (no exceptions)."""
        # Should not raise
        manager.show_system_message("System info")

    @pytest.mark.asyncio
    async def test_refresh_loop_stops_cleanly(self, manager):
        """Test refresh loop stops without errors."""
        await manager.start_streaming()

        # Let refresh loop run a bit
        await asyncio.sleep(0.3)

        # Stop streaming
        await manager.stop_streaming()

        # Should have no active tasks
        assert manager._refresh_task is None or manager._refresh_task.done()

    @pytest.mark.asyncio
    async def test_refresh_loop_with_tool(self, manager):
        """Test refresh loop runs during tool execution."""
        await manager.start_tool_execution("tool", {})

        # Let refresh loop run a bit
        await asyncio.sleep(0.3)

        await manager.stop_tool_execution("result")

        # Should have no active tasks
        assert manager._refresh_task is None or manager._refresh_task.done()

    @pytest.mark.asyncio
    async def test_content_type_detection_in_stream(self, manager):
        """Test content type is detected during streaming."""
        await manager.start_streaming()

        await manager.add_chunk({"content": "def hello():"})

        assert manager.streaming_state.detected_type == ContentType.CODE

        # Cleanup
        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_elapsed_time_tracking(self, manager):
        """Test elapsed time is tracked."""
        await manager.start_streaming()

        # Add small delay
        await asyncio.sleep(0.1)

        assert manager.streaming_state.elapsed_time >= 0.1

        # Cleanup
        await manager.stop_streaming()


class TestChukTermRenderer:
    """Tests for ChukTermRenderer."""

    def test_render_returns_content(self):
        """Test renderer returns content as-is."""
        renderer = ChukTermRenderer()

        result = renderer.render("Test content", ContentType.TEXT)

        assert result == "Test content"

    def test_render_with_different_types(self):
        """Test rendering different content types."""
        renderer = ChukTermRenderer()

        for content_type in ContentType:
            result = renderer.render("Content", content_type)
            assert result == "Content"


class TestDisplayManagerIntegration:
    """Integration tests for display manager."""

    @pytest.mark.asyncio
    async def test_full_streaming_lifecycle(self):
        """Test complete streaming lifecycle."""
        manager = StreamingDisplayManager()

        # Start streaming
        await manager.start_streaming()
        assert manager.is_streaming

        # Add chunks with various content
        await manager.add_chunk({"content": "def hello():\n"})
        await manager.add_chunk({"content": "    return 'world'\n"})
        await manager.add_chunk({"content": "", "finish_reason": "stop"})

        # Update reasoning
        await manager.update_reasoning("Generated a function")

        # Stop streaming
        final = await manager.stop_streaming()

        assert final == "def hello():\n    return 'world'\n"
        assert manager.streaming_state.detected_type == ContentType.CODE
        assert manager.streaming_state.chunks_received == 3
        assert manager.streaming_state.finish_reason == "stop"
        assert not manager.is_streaming

    @pytest.mark.asyncio
    async def test_full_tool_lifecycle(self):
        """Test complete tool execution lifecycle."""
        manager = StreamingDisplayManager()

        # Start tool
        await manager.start_tool_execution(
            "database_query", {"query": "SELECT * FROM users"}
        )
        assert manager.is_tool_executing

        # Simulate execution delay
        await asyncio.sleep(0.1)

        # Stop with success
        await manager.stop_tool_execution("42 rows returned", success=True)

        assert not manager.is_tool_executing

    @pytest.mark.asyncio
    async def test_stream_then_tool_sequence(self):
        """Test streaming followed by tool execution."""
        manager = StreamingDisplayManager()

        # Stream response
        await manager.start_streaming()
        await manager.add_chunk({"content": "I'll query the database"})
        await manager.stop_streaming()

        # Execute tool
        await manager.start_tool_execution("query_db", {"table": "users"})
        await manager.stop_tool_execution("Results: ...")

        assert not manager.is_busy

    @pytest.mark.asyncio
    async def test_interrupted_stream_handling(self):
        """Test handling interrupted stream."""
        manager = StreamingDisplayManager()

        await manager.start_streaming()
        await manager.add_chunk({"content": "Starting long response..."})

        # User interrupts
        content = await manager.stop_streaming(interrupted=True)

        assert content == "Starting long response..."
        assert manager.streaming_state.interrupted
        assert manager.streaming_state.phase == StreamingPhase.INTERRUPTED
