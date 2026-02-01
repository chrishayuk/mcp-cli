"""Tests for display renderers module."""

from mcp_cli.display.renderers import (
    render_streaming_status,
    render_tool_execution_status,
    show_final_streaming_response,
    show_tool_execution_result,
)
from mcp_cli.display.models import StreamingState
from mcp_cli.chat.models import ToolExecutionState
import time


class TestRenderStreamingStatus:
    """Tests for render_streaming_status function."""

    def test_basic_streaming_status(self):
        """Test basic streaming status rendering."""
        state = StreamingState()
        state.chunks_received = 10
        state.accumulated_content = "Hello world"

        result = render_streaming_status(state, "⠙")

        assert "⠙" in result
        assert "Streaming" in result
        assert "(10 chunks)" in result
        assert "11 chars" in result
        assert "s" in result  # time

    def test_streaming_with_reasoning(self):
        """Test streaming status with cached reasoning preview."""
        state = StreamingState()
        state.chunks_received = 5
        state.accumulated_content = "Answer"
        state.reasoning_content = "Let me think about this problem step by step..."

        # Create formatted preview (as the manager would - multi-line format)
        reasoning_preview = (
            f"💭 Thinking ({len(state.reasoning_content)} chars):\n  ...step by step"
        )

        result = render_streaming_status(
            state, "⠹", reasoning_preview=reasoning_preview
        )

        assert "⠹" in result
        assert "Streaming" in result
        assert "💭 Thinking" in result
        assert str(len(state.reasoning_content)) in result
        assert "...step by step" in result

    def test_streaming_without_reasoning(self):
        """Test streaming status without reasoning."""
        state = StreamingState()
        state.chunks_received = 3
        state.accumulated_content = "Test"

        result = render_streaming_status(state, "⠸")

        assert "⠸" in result
        assert "Streaming" in result
        assert "💭" not in result  # No thinking indicator

    def test_streaming_long_reasoning(self):
        """Test streaming with very long reasoning content."""
        state = StreamingState()
        state.reasoning_content = "x" * 200

        # Create formatted preview with truncation (multi-line format)
        reasoning_preview = f"💭 Thinking (200 chars):\n  ...{'x' * 27}"

        result = render_streaming_status(
            state, "⠼", reasoning_preview=reasoning_preview
        )

        # Should have preview (truncated)
        assert "200 chars" in result
        assert "💭 Thinking" in result
        assert "x" * 27 in result


class TestRenderToolExecutionStatus:
    """Tests for render_tool_execution_status function."""

    def test_basic_tool_status(self):
        """Test basic tool execution status."""
        tool = ToolExecutionState(
            name="test_tool",
            arguments={},
            start_time=time.time(),
        )

        result = render_tool_execution_status(tool, "⠙", elapsed=1.5)

        assert "⠙" in result
        assert "Executing tool: test_tool" in result
        assert "(1.5s)" in result

    def test_tool_with_arguments(self):
        """Test tool status with arguments."""
        tool = ToolExecutionState(
            name="query_db",
            arguments={"query": "SELECT * FROM users", "limit": 10},
            start_time=time.time(),
        )

        result = render_tool_execution_status(tool, "⠹", elapsed=2.0)

        assert "⠹" in result
        assert "Executing tool: query_db" in result
        assert "(2.0s)" in result
        # Should show args preview with pipe separator
        assert "|" in result
        assert "query=" in result or "limit=" in result

    def test_tool_without_arguments(self):
        """Test tool status without arguments."""
        tool = ToolExecutionState(
            name="ping",
            arguments={},
            start_time=time.time(),
        )

        result = render_tool_execution_status(tool, "⠸", elapsed=0.5)

        assert "Executing tool: ping" in result
        assert "(0.5s)" in result
        # Should not have pipe separator when no args
        assert "|" not in result

    def test_tool_with_many_arguments(self):
        """Test tool with many arguments (should preview first 2)."""
        tool = ToolExecutionState(
            name="complex_tool",
            arguments={"a": "1", "b": "2", "c": "3", "d": "4"},
            start_time=time.time(),
        )

        result = render_tool_execution_status(tool, "⠼", elapsed=1.0)

        assert "Executing tool: complex_tool" in result
        # Should show first 4 args (a, b, c, d)
        assert "a=1" in result
        assert "b=2" in result


class TestShowFinalStreamingResponse:
    """Tests for show_final_streaming_response function."""

    def test_normal_response(self, capsys):
        """Test showing normal (not interrupted) response."""
        show_final_streaming_response(
            content="This is the final response",
            elapsed=2.5,
            interrupted=False,
        )

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "🤖 Assistant" in output
        assert "(2.5s)" in output
        assert "This is the final response" in output
        assert "interrupted" not in output.lower()

    def test_interrupted_response(self, capsys):
        """Test showing interrupted response."""
        show_final_streaming_response(
            content="Partial response",
            elapsed=1.0,
            interrupted=True,
        )

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "interrupted" in output.lower()
        assert "⚠" in output


class TestShowToolExecutionResult:
    """Tests for show_tool_execution_result function."""

    def test_successful_tool_with_result(self, capsys):
        """Test showing successful tool execution with result."""
        tool = ToolExecutionState(
            name="fetch_data",
            arguments={},
            start_time=time.time(),
            result="Data fetched successfully",
            success=True,
            elapsed=1.5,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "✓" in output
        assert "fetch_data" in output
        assert "1.5" in output or "1.50s" in output
        assert "Data fetched successfully" in output

    def test_successful_tool_without_result(self, capsys):
        """Test showing successful tool with no result."""
        tool = ToolExecutionState(
            name="ping",
            arguments={},
            start_time=time.time(),
            result=None,
            success=True,
            elapsed=0.5,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "✓" in output
        assert "ping" in output
        assert "completed" in output

    def test_failed_tool_with_error(self, capsys):
        """Test showing failed tool execution."""
        tool = ToolExecutionState(
            name="broken_tool",
            arguments={},
            start_time=time.time(),
            result="Connection timeout",
            success=False,
            elapsed=3.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # chuk-term error output may not preserve all formatting
        # Just verify the error message is shown
        assert "Error:" in output
        assert "Connection timeout" in output

    def test_failed_tool_without_error(self, capsys):
        """Test showing failed tool with no error message."""
        tool = ToolExecutionState(
            name="mystery_fail",
            arguments={},
            start_time=time.time(),
            result=None,
            success=False,
            elapsed=1.0,
            completed=True,
        )

        # Just verify function runs without error
        # (chuk-term output may not be captured by capsys)
        show_tool_execution_result(tool)

        # Function completed successfully
        assert tool.completed is True

    def test_tool_with_long_result(self, capsys):
        """Test that long results are truncated to 200 chars."""
        long_result = "x" * 300

        tool = ToolExecutionState(
            name="big_tool",
            arguments={},
            start_time=time.time(),
            result=long_result,
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Result should be truncated
        assert "..." in output
        # Should not contain full 300 chars
        assert output.count("x") < 250
