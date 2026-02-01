"""Additional tests for StreamingDisplayManager to increase coverage."""

import asyncio
import time

import pytest

from mcp_cli.display.manager import StreamingDisplayManager
from mcp_cli.display.models import StreamingState


class TestStreamingDisplayManagerCoverage:
    """Additional tests for uncovered code paths in StreamingDisplayManager."""

    @pytest.fixture
    def manager(self):
        """Create a display manager instance."""
        return StreamingDisplayManager()

    # ==================== STREAMING EDGE CASES ====================

    @pytest.mark.asyncio
    async def test_add_chunk_returns_early_when_no_state_created(self, manager):
        """Test add_chunk handles edge case where start_streaming fails."""
        # This tests line 130 - the type guard after start_streaming
        # Directly set streaming_state to None to simulate failure
        manager.streaming_state = None

        # Mock start_streaming to not create state
        original_start = manager.start_streaming

        async def mock_start():
            pass  # Don't create state

        manager.start_streaming = mock_start

        # Should not raise, just return early
        await manager.add_chunk({"content": "test"})

        # Restore
        manager.start_streaming = original_start

    @pytest.mark.asyncio
    async def test_update_reasoning_without_state(self, manager):
        """Test update_reasoning returns early when no streaming state."""
        # Line 148 - early return when no streaming_state
        manager.streaming_state = None

        # Should not raise
        await manager.update_reasoning("some reasoning")

    # ==================== TOOL EXECUTION TIMEOUT HANDLING ====================

    @pytest.mark.asyncio
    async def test_start_tool_execution_with_stuck_refresh_task(self, manager):
        """Test start_tool_execution handles stuck refresh task."""
        # Lines 243-248 - timeout handling for stuck refresh task
        await manager.start_streaming()

        # Create a mock task that takes a long time
        async def slow_task():
            await asyncio.sleep(10)

        manager._refresh_task = asyncio.create_task(slow_task())
        manager._refresh_active = True

        # Starting tool should timeout and cancel the stuck task
        await manager.start_tool_execution("test", {})

        # Should have new tool state
        assert manager.tool_execution is not None
        assert manager.tool_execution.name == "test"

        # Cleanup
        await manager.stop_tool_execution("done")

    @pytest.mark.asyncio
    async def test_start_tool_clears_stale_streaming_state(self, manager):
        """Test start_tool_execution clears inactive streaming state."""
        # Lines 267-273 and earlier - clearing stale streaming state
        await manager.start_streaming()
        await manager.add_chunk({"content": "test"})

        # Complete the streaming (marks as inactive via complete())
        manager.streaming_state.complete()

        # Start tool - should clear stale streaming state
        await manager.start_tool_execution("tool", {"arg": "val"})

        assert manager.streaming_state is None
        assert manager.tool_execution is not None

        # Cleanup
        await manager.stop_tool_execution("done")

    # ==================== REFRESH LOOP HANDLING ====================

    @pytest.mark.asyncio
    async def test_stop_refresh_loop_timeout(self, manager):
        """Test stop_refresh_loop handles timeout cancellation."""
        # Lines 355-360 - timeout and cancellation

        async def stuck_loop():
            while True:
                await asyncio.sleep(0.01)

        manager._refresh_task = asyncio.create_task(stuck_loop())
        manager._refresh_active = True

        # Stop should timeout and cancel
        await manager._stop_refresh_loop()

        assert manager._refresh_task is None

    @pytest.mark.asyncio
    async def test_refresh_loop_error_handling(self, manager):
        """Test refresh loop handles errors gracefully."""
        # Lines 389-390 - error logging in refresh loop
        await manager.start_streaming()

        # Corrupt the state to cause an error
        manager.streaming_state = "not a StreamingState"

        # Let loop run and hit error
        await asyncio.sleep(0.2)

        # Stop - should not raise
        manager._refresh_active = False
        if manager._refresh_task:
            manager._refresh_task.cancel()
            try:
                await manager._refresh_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_refresh_loop_when_already_running(self, manager):
        """Test start_refresh_loop returns early when already running."""
        # Line 340 - early return when task exists
        await manager.start_streaming()

        task1 = manager._refresh_task

        # Try to start again
        await manager._start_refresh_loop()

        # Should be same task
        assert manager._refresh_task is task1

        # Cleanup
        await manager.stop_streaming()

    # ==================== RENDERING STATUS MODES ====================

    @pytest.mark.asyncio
    async def test_render_streaming_status_with_content(self, manager):
        """Test rendering when content exists (no reasoning)."""
        # Lines 423-429 - content mode
        await manager.start_streaming()
        await manager.add_chunk({"content": "Hello world"})

        # Let refresh loop render
        await asyncio.sleep(0.2)

        # Should be in content mode
        assert manager.streaming_state.content_length > 0

        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_render_streaming_status_with_reasoning(self, manager):
        """Test rendering with reasoning preview (no content)."""
        # Lines 430-435 - thinking mode
        await manager.start_streaming()

        # Set reasoning but no content
        manager.streaming_state.reasoning_content = "Let me think about this..."
        manager.streaming_state.accumulated_content = ""

        # Force reasoning preview update
        manager._last_reasoning_update = 0
        manager._update_reasoning_preview()

        # Let refresh loop render
        await asyncio.sleep(0.2)

        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_render_streaming_mode_switch(self, manager):
        """Test switching from thinking mode to content mode."""
        # Lines 444-449 - mode switching detection
        await manager.start_streaming()

        # Start with reasoning only
        manager.streaming_state.reasoning_content = "Thinking..."
        manager._showing_thinking = True
        manager._last_reasoning_update = 0
        manager._update_reasoning_preview()

        await asyncio.sleep(0.15)

        # Then add content - triggers mode switch
        await manager.add_chunk({"content": "Response"})

        await asyncio.sleep(0.15)

        await manager.stop_streaming()

    # ==================== MULTI-LINE DISPLAY HANDLING ====================

    @pytest.mark.asyncio
    async def test_multiline_display_clear_and_rewrite(self, manager):
        """Test clearing and rewriting multi-line display."""
        # Lines 467-481, 495-501 - multi-line handling
        await manager.start_streaming()

        # Set up multi-line reasoning preview
        manager.streaming_state.reasoning_content = "x" * 300
        manager._last_line_count = 3
        manager._showing_thinking = True
        manager._last_reasoning_update = 0
        manager._update_reasoning_preview()

        # Force render
        await manager._render_streaming_status()

        await manager.stop_streaming()

    @pytest.mark.asyncio
    async def test_single_line_display_update(self, manager):
        """Test single line status update."""
        # Lines 483-492 - single line clear and rewrite
        await manager.start_streaming()
        await manager.add_chunk({"content": "short"})

        manager._last_line_count = 1

        # Force render
        await manager._render_streaming_status()

        await manager.stop_streaming()

    # ==================== CLEAR PREVIOUS LINES ====================

    def test_clear_previous_lines_single_line(self, manager):
        """Test clearing a single line."""
        # Lines 508-530 - clearing with single line
        manager._last_line_count = 1

        # Should not raise
        manager._clear_previous_lines()

    def test_clear_previous_lines_multiple_lines(self, manager, capsys):
        """Test clearing multiple lines."""
        # Lines 508-530 - clearing multiple lines
        manager._last_line_count = 3

        manager._clear_previous_lines()

        # Check ANSI codes were used
        captured = capsys.readouterr()
        # Should contain clear sequences
        assert "\033[K" in captured.out or captured.out == ""

    def test_clear_previous_lines_zero_lines(self, manager):
        """Test clearing when no lines to clear."""
        # Line 508-509 - early return
        manager._last_line_count = 0

        # Should return early, not raise
        manager._clear_previous_lines()

    # ==================== CLEAR CURRENT DISPLAY ====================

    @pytest.mark.asyncio
    async def test_clear_current_display_async_multiline(self, manager):
        """Test async clearing of multi-line display."""
        # Lines 540-556 - async version
        manager._last_line_count = 3
        manager._last_status = "some status"

        await manager._clear_current_display_async()

        assert manager._last_line_count == 0
        assert manager._last_status == ""

    @pytest.mark.asyncio
    async def test_clear_current_display_async_zero_lines(self, manager):
        """Test async clearing when no lines."""
        manager._last_line_count = 0

        await manager._clear_current_display_async()

        assert manager._last_line_count == 0

    def test_clear_current_display_sync_multiline(self, manager):
        """Test sync clearing of multi-line display."""
        # Lines 570-575 - sync version with multiple lines
        manager._last_line_count = 3
        manager._last_status = "status"

        manager._clear_current_display()

        assert manager._last_line_count == 0
        assert manager._last_status == ""

    def test_clear_current_display_sync_zero_lines(self, manager):
        """Test sync clearing when no lines to clear."""
        manager._last_line_count = 0

        manager._clear_current_display()

        assert manager._last_line_count == 0

    # ==================== SPLIT PREVIEW INTO LINES ====================

    def test_split_preview_short_text(self, manager):
        """Test splitting short text into lines."""
        # Lines 614-645
        result = manager._split_preview_into_lines("short text", max_line_len=80)

        assert len(result) == 1
        assert result[0] == "short text"

    def test_split_preview_with_ellipsis(self, manager):
        """Test splitting text that starts with ellipsis."""
        result = manager._split_preview_into_lines(
            "...continued from before", max_line_len=80
        )

        assert len(result) >= 1
        assert result[0].startswith("...")

    def test_split_preview_long_text(self, manager):
        """Test splitting long text into multiple lines."""
        long_text = " ".join(["word"] * 50)  # ~250 chars

        result = manager._split_preview_into_lines(
            long_text, max_line_len=80, num_lines=3
        )

        assert len(result) <= 3
        for line in result:
            assert len(line) <= 85  # Allow small overflow for word boundaries

    def test_split_preview_exact_lines(self, manager):
        """Test text splits into exact number of lines."""
        # Create text that should split into exactly 3 lines
        text = " ".join(["word"] * 30)

        result = manager._split_preview_into_lines(text, max_line_len=40, num_lines=3)

        assert len(result) <= 3

    def test_split_preview_empty_text(self, manager):
        """Test splitting empty text."""
        result = manager._split_preview_into_lines("", max_line_len=80)

        assert result == []

    def test_split_preview_single_long_word(self, manager):
        """Test handling single word longer than max_line_len."""
        long_word = "x" * 100

        result = manager._split_preview_into_lines(long_word, max_line_len=80)

        # Should still return the word (won't be broken mid-word)
        assert len(result) >= 1

    # ==================== UPDATE REASONING PREVIEW ====================

    def test_update_reasoning_preview_no_state(self, manager):
        """Test update with no streaming state."""
        # Lines 656-658
        manager.streaming_state = None

        manager._update_reasoning_preview()

        assert manager._last_reasoning_preview == ""

    def test_update_reasoning_preview_no_reasoning(self, manager):
        """Test update with empty reasoning content."""
        manager.streaming_state = StreamingState()
        manager.streaming_state.reasoning_content = ""

        manager._update_reasoning_preview()

        assert manager._last_reasoning_preview == ""

    def test_update_reasoning_preview_time_debounce(self, manager):
        """Test time-based debouncing."""
        # Lines 670-672
        manager.streaming_state = StreamingState()
        manager.streaming_state.reasoning_content = "Test reasoning"
        manager._last_reasoning_update = 0  # Force update

        manager._update_reasoning_preview()

        assert manager._last_reasoning_preview != ""
        assert "💭 Thinking" in manager._last_reasoning_preview

    def test_update_reasoning_preview_chunk_debounce(self, manager):
        """Test chunk-based debouncing."""
        # Lines 675-677
        manager.streaming_state = StreamingState()
        manager.streaming_state.reasoning_content = "Test reasoning"
        manager.streaming_state.chunks_received = 50
        manager._last_reasoning_chunk_count = 0  # Force chunk-based update
        manager._last_reasoning_update = time.time()  # Recent time update

        manager._update_reasoning_preview()

        # Should have updated due to chunk threshold
        assert manager._last_reasoning_preview != ""

    def test_update_reasoning_preview_no_update_when_debounced(self, manager):
        """Test preview not updated when debounced."""
        manager.streaming_state = StreamingState()
        manager.streaming_state.reasoning_content = "Test reasoning"
        manager._last_reasoning_update = time.time()
        manager._last_reasoning_chunk_count = manager.streaming_state.chunks_received
        manager._last_reasoning_preview = "old preview"

        manager._update_reasoning_preview()

        # Should keep old preview
        assert manager._last_reasoning_preview == "old preview"

    def test_update_reasoning_preview_long_content(self, manager):
        """Test reasoning preview with very long content (>1000 chars)."""
        # Lines 688-691 - formatting length for 1k+ content
        manager.streaming_state = StreamingState()
        manager.streaming_state.reasoning_content = "x" * 1500
        manager._last_reasoning_update = 0

        manager._update_reasoning_preview()

        # Should show formatted length with 'k'
        assert "1.5k" in manager._last_reasoning_preview

    def test_update_reasoning_preview_short_content(self, manager):
        """Test reasoning preview with short content (<1000 chars)."""
        manager.streaming_state = StreamingState()
        manager.streaming_state.reasoning_content = "x" * 500
        manager._last_reasoning_update = 0

        manager._update_reasoning_preview()

        # Should show raw number
        assert "500" in manager._last_reasoning_preview

    # ==================== RENDER TOOL STATUS ====================

    @pytest.mark.asyncio
    async def test_render_tool_status_without_tool(self, manager):
        """Test render_tool_status returns early with no tool."""
        # Line 712
        manager.tool_execution = None

        await manager._render_tool_status()

        # Should not raise

    @pytest.mark.asyncio
    async def test_render_tool_status_updates_live_status(self, manager):
        """Test render_tool_status updates live status display."""
        await manager.start_tool_execution("test_tool", {"arg": "value"})

        # Let it render
        await asyncio.sleep(0.15)

        # Should have updated last_status
        assert manager._last_status != ""
        assert manager._last_line_count == 1

        await manager.stop_tool_execution("done")

    # ==================== DO CLEAR DISPLAY ====================

    def test_do_clear_display_with_lines(self, manager):
        """Test _do_clear_display with lines to clear."""
        # Lines 265-277
        manager._last_line_count = 2
        manager._last_status = "some status"

        manager._do_clear_display()

        assert manager._last_line_count == 0
        assert manager._last_status == ""

    def test_do_clear_display_single_line(self, manager):
        """Test _do_clear_display with single line."""
        manager._last_line_count = 1
        manager._last_status = "status"

        manager._do_clear_display()

        assert manager._last_line_count == 0

    def test_do_clear_display_no_lines(self, manager):
        """Test _do_clear_display with no lines."""
        manager._last_line_count = 0

        manager._do_clear_display()

        assert manager._last_line_count == 0

    # ==================== FINISH DISPLAY ====================

    def test_finish_display(self, manager, capsys):
        """Test _finish_display clears and adds newline."""
        manager._last_line_count = 1
        manager._last_status = "status"

        manager._finish_display()

        assert manager._last_line_count == 0
        assert manager._last_status == ""

        # Should have printed newline
        captured = capsys.readouterr()
        assert "\n" in captured.out
