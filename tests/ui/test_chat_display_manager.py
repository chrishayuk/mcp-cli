"""
Tests for the centralized chat display manager.

Tests the ChatDisplayManager class that consolidates ALL UI display logic
for chat mode into a single coherent system.
"""

import pytest
import time
import json
from unittest.mock import patch

from mcp_cli.ui.chat_display_manager import ChatDisplayManager
from mcp_cli.chat.models import ToolExecutionState


class TestChatDisplayManager:
    """Tests for ChatDisplayManager class."""

    @pytest.fixture
    def manager(self):
        """Create a ChatDisplayManager instance."""
        # Console parameter is optional and not used with chuk-term
        return ChatDisplayManager()

    def test_initialization(self, manager):
        """Test proper initialization."""
        # Display state
        assert not manager.is_streaming
        assert manager.streaming_content == ""
        assert manager.streaming_start_time == 0.0

        assert not manager.is_tool_executing
        assert manager.current_tool is None
        assert manager.tool_start_time == 0.0

        # Spinner
        assert len(manager.spinner_frames) > 0
        assert manager.spinner_index == 0

        # Live display tracking
        assert not manager.live_display_active
        assert manager.last_status_line == ""

    # ==================== STREAMING TESTS ====================

    def test_start_streaming(self, manager):
        """Test starting streaming response display."""
        manager.start_streaming()

        assert manager.is_streaming is True
        assert manager.streaming_content == ""
        assert isinstance(manager.streaming_start_time, float)
        assert manager.live_display_active is True

    def test_update_streaming(self, manager):
        """Test updating streaming content."""
        manager.is_streaming = True
        manager.live_display_active = True

        with patch("builtins.print") as mock_print:
            manager.update_streaming("Hello")
            manager.update_streaming(" world")

            assert manager.streaming_content == "Hello world"
            # Should have printed status updates
            assert mock_print.call_count > 0

    def test_update_streaming_when_not_streaming(self, manager):
        """Test update streaming does nothing when not streaming."""
        manager.is_streaming = False

        with patch.object(manager, "_refresh_display") as mock_refresh:
            manager.update_streaming("Hello")

            assert manager.streaming_content == ""
            mock_refresh.assert_not_called()

    def test_finish_streaming(self, manager):
        """Test finishing streaming response."""
        # Set up streaming state
        manager.is_streaming = True
        manager.streaming_content = "Final content"
        manager.streaming_start_time = time.time() - 2.0
        manager.live_display_active = True

        with (
            patch("chuk_term.ui.terminal.clear_line"),
            patch.object(manager, "_show_final_response") as mock_show,
        ):
            manager.finish_streaming()

            assert manager.is_streaming is False
            assert not manager.live_display_active

            # Should show final response
            mock_show.assert_called_once()
            args = mock_show.call_args[0]
            assert args[0] == "Final content"
            assert 1.9 < args[1] < 2.1  # Elapsed time approximately 2 seconds

    def test_finish_streaming_when_not_streaming(self, manager):
        """Test finish streaming when not streaming does nothing."""
        manager.is_streaming = False

        with patch.object(manager, "_stop_live_display") as mock_stop:
            manager.finish_streaming()
            mock_stop.assert_not_called()

    def test_finish_streaming_no_content(self, manager):
        """Test finish streaming with no content."""
        manager.is_streaming = True
        manager.streaming_content = ""
        manager.live_display_active = True

        with patch("chuk_term.ui.output.print"):
            manager.finish_streaming()

            assert not manager.is_streaming
            # No content means no display

    # ==================== TOOL EXECUTION TESTS ====================

    def test_start_tool_execution(self, manager):
        """Test starting tool execution display."""
        tool_name = "test_tool"
        arguments = {"param1": "value1", "param2": 42}

        manager.start_tool_execution(tool_name, arguments)

        assert manager.is_tool_executing is True
        assert manager.current_tool is not None
        assert manager.live_display_active is True

        current_tool = manager.current_tool
        assert current_tool.name == tool_name
        assert current_tool.arguments == arguments
        assert isinstance(current_tool.start_time, float)

    def test_finish_tool_execution_success(self, manager):
        """Test finishing tool execution successfully."""
        from chuk_term.ui import output

        # Set up tool execution state
        manager.is_tool_executing = True
        manager.current_tool = ToolExecutionState(
            name="test_tool",
            arguments={"param": "value"},
            start_time=time.time() - 1.5,
        )
        manager.live_display_active = True

        result = "Tool completed successfully"

        with (
            patch.object(output, "success") as mock_success,
            patch("mcp_cli.ui.chat_display_manager.clear_line"),
        ):
            manager.finish_tool_execution(result, success=True)

            assert manager.is_tool_executing is False
            assert manager.current_tool is None
            assert not manager.live_display_active

            # Should show success message
            mock_success.assert_called_once()
            assert "Completed" in mock_success.call_args[0][0]

    def test_finish_tool_execution_failure(self, manager):
        """Test finishing tool execution with failure."""
        from chuk_term.ui import output

        manager.is_tool_executing = True
        manager.current_tool = ToolExecutionState(
            name="failing_tool",
            arguments={},
            start_time=time.time() - 0.5,
        )
        manager.live_display_active = True

        error_result = "Tool failed with error"

        with (
            patch.object(output, "error") as mock_error,
            patch("mcp_cli.ui.chat_display_manager.clear_line"),
        ):
            manager.finish_tool_execution(error_result, success=False)

            assert manager.is_tool_executing is False
            assert manager.current_tool is None

            # Should show error message
            mock_error.assert_called_once()
            assert "Failed" in mock_error.call_args[0][0]

    def test_finish_tool_execution_when_not_executing(self, manager):
        """Test finish tool execution when not executing does nothing."""
        manager.is_tool_executing = False
        manager.current_tool = None

        with patch.object(manager, "_stop_live_display") as mock_stop:
            manager.finish_tool_execution("result")
            mock_stop.assert_not_called()

    # ==================== MESSAGE DISPLAY TESTS ====================

    def test_show_user_message(self, manager):
        """Test showing user message."""
        from chuk_term.ui import output

        message = "Hello, how can I help?"

        with patch.object(output, "print") as mock_print:
            manager.show_user_message(message)

            mock_print.assert_called_once()
            assert "User:" in mock_print.call_args[0][0]
            assert message in mock_print.call_args[0][0]

    def test_show_assistant_message(self, manager):
        """Test showing assistant message."""
        from chuk_term.ui import output

        content = "Here's my response"
        elapsed = 2.5

        with patch.object(output, "print") as mock_print:
            manager.show_assistant_message(content, elapsed)

            assert mock_print.call_count == 2  # Header and content
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Assistant" in c for c in calls)
            assert any(content in c for c in calls)

    # ==================== LIVE DISPLAY MANAGEMENT TESTS ====================

    def test_ensure_live_display(self, manager):
        """Test ensuring live display is active."""
        assert not manager.live_display_active

        manager._ensure_live_display()

        assert manager.live_display_active

    def test_ensure_live_display_when_already_active(self, manager):
        """Test ensure live display when already active."""
        manager.live_display_active = True

        manager._ensure_live_display()

        # Should still be active
        assert manager.live_display_active

    def test_stop_live_display(self, manager):
        """Test stopping live display."""

        manager.live_display_active = True
        manager.last_status_line = "test status"

        with patch("mcp_cli.ui.chat_display_manager.clear_line") as mock_clear:
            manager._stop_live_display()

        assert not manager.live_display_active
        assert manager.last_status_line == ""
        mock_clear.assert_called_once()

    def test_stop_live_display_when_none(self, manager):
        """Test stopping live display when none exists."""
        manager.live_display_active = False

        # Should not raise an error
        manager._stop_live_display()

    def test_refresh_display(self, manager):
        """Test refreshing live display."""
        manager.live_display_active = True
        manager.last_status_line = "old status"
        manager.is_streaming = True
        manager.streaming_content = "test"

        with patch("builtins.print") as mock_print:
            with patch("chuk_term.ui.terminal.clear_line"):
                manager._refresh_display()

        # Should have printed new status
        mock_print.assert_called()

    def test_refresh_display_when_not_active(self, manager):
        """Test refresh display when not active."""
        manager.live_display_active = False

        with patch("builtins.print") as mock_print:
            manager._refresh_display()

        # Should not print anything
        mock_print.assert_not_called()

    # ==================== LIVE STATUS CREATION TESTS ====================

    @patch("mcp_cli.ui.chat_display_manager.time.time")
    def test_create_live_status_streaming(self, mock_time, manager):
        """Test creating live status during streaming."""
        mock_time.return_value = 100.5

        manager.is_streaming = True
        manager.streaming_content = "Test streaming content"
        manager.streaming_start_time = 100.0

        status = manager._create_live_status()

        assert "Generating response" in status
        assert "22 chars" in status  # len("Test streaming content")
        assert "0.5s" in status

    @patch("mcp_cli.ui.chat_display_manager.time.time")
    def test_create_live_status_tool_executing(self, mock_time, manager):
        """Test creating live status during tool execution."""
        mock_time.return_value = 200.5

        manager.is_tool_executing = True
        manager.current_tool = ToolExecutionState(
            name="test_tool",
            arguments={"param": "value"},
            start_time=200.0,
        )

        status = manager._create_live_status()

        assert "Executing test_tool" in status
        assert "(0.5s)" in status

    def test_create_live_status_spinner_animation(self, manager):
        """Test spinner animation in live status."""
        manager.is_streaming = True
        manager.streaming_content = "test"
        manager.streaming_start_time = time.time()

        initial_index = manager.spinner_index

        # Create status multiple times
        manager._create_live_status()
        manager._create_live_status()

        # Spinner should animate
        assert manager.spinner_index != initial_index
        assert manager.spinner_index < len(manager.spinner_frames)

    def test_create_live_status_neither_streaming_nor_executing(self, manager):
        """Test create live status when neither streaming nor executing."""
        status = manager._create_live_status()
        assert status == ""

    # ==================== FINAL DISPLAY TESTS ====================

    def test_show_final_response(self, manager):
        """Test showing final response."""
        from chuk_term.ui import output

        content = "Final response content"
        elapsed = 2.5

        with patch.object(output, "print") as mock_print:
            manager._show_final_response(content, elapsed)

            assert mock_print.call_count == 2  # Header and content
            calls = [str(c) for c in mock_print.call_args_list]
            assert any("Assistant" in c for c in calls)
            assert any(content in c for c in calls)

    def test_show_final_tool_result_success(self, manager):
        """Test showing successful tool result."""
        from chuk_term.ui import output

        manager.current_tool = ToolExecutionState(
            name="successful_tool",
            arguments={"param": "value"},
            start_time=time.time(),
            elapsed=1.5,
            success=True,
            result='{"status": "ok"}',
        )

        with (
            patch.object(output, "success") as mock_success,
            patch.object(output, "code") as mock_code,
            patch.object(output, "print"),
        ):
            manager._show_final_tool_result()

            mock_success.assert_called_once()
            assert "Completed" in mock_success.call_args[0][0]

            # Should format JSON result
            mock_code.assert_called_once()

    def test_show_final_tool_result_failure(self, manager):
        """Test showing failed tool result."""
        from chuk_term.ui import output

        manager.current_tool = ToolExecutionState(
            name="failed_tool",
            arguments={"param": "value"},
            start_time=time.time(),
            elapsed=0.8,
            success=False,
            result="Error occurred during execution",
        )

        with (
            patch.object(output, "error") as mock_error,
            patch.object(output, "print"),
        ):
            manager._show_final_tool_result()

            mock_error.assert_called_once()
            assert "Failed" in mock_error.call_args[0][0]

    def test_show_final_tool_result_no_current_tool(self, manager):
        """Test showing tool result when no current tool."""
        from chuk_term.ui import output

        manager.current_tool = None

        with (
            patch.object(output, "success") as mock_success,
            patch.object(output, "error") as mock_error,
        ):
            manager._show_final_tool_result()

            # Should not call any output methods
            mock_success.assert_not_called()
            mock_error.assert_not_called()

    def test_show_final_tool_result_json_formatting(self, manager):
        """Test tool result with JSON formatting."""
        from chuk_term.ui import output

        json_result = '{"status": "success", "data": {"items": [1, 2, 3]}}'

        manager.current_tool = ToolExecutionState(
            name="json_tool",
            arguments={},
            start_time=time.time(),
            elapsed=1.0,
            success=True,
            result=json_result,
        )

        with (
            patch.object(output, "success"),
            patch.object(output, "code") as mock_code,
            patch.object(output, "print"),
        ):
            manager._show_final_tool_result()

            # Should format JSON nicely
            mock_code.assert_called_once()
            formatted = mock_code.call_args[0][0]
            # Check it's formatted JSON
            parsed = json.loads(formatted)
            assert parsed["status"] == "success"

    def test_show_final_tool_result_invalid_json(self, manager):
        """Test tool result with invalid JSON falls back to text."""
        from chuk_term.ui import output

        invalid_json = "Not valid JSON content"

        manager.current_tool = ToolExecutionState(
            name="text_tool",
            arguments={},
            start_time=time.time(),
            elapsed=1.0,
            success=True,
            result=invalid_json,
        )

        with (
            patch.object(output, "success"),
            patch.object(output, "print") as mock_print,
        ):
            manager._show_final_tool_result()

            # Should print as plain text, not code
            mock_print.assert_called()
            # Check the result text was printed
            calls = mock_print.call_args_list
            assert any("Not valid JSON content" in str(c) for c in calls)

    def test_show_final_tool_result_filtered_arguments(self, manager):
        """Test tool result with filtered arguments display."""
        from chuk_term.ui import output

        manager.current_tool = ToolExecutionState(
            name="test_tool",
            arguments={
                "valid_param": "value",
                "empty_param": "",
                "none_param": None,
                "whitespace_param": "   ",
                "another_valid": "another_value",
            },
            start_time=time.time(),
            elapsed=1.0,
            success=True,
            result="result",
        )

        with (
            patch.object(output, "success"),
            patch.object(output, "print") as mock_print,
        ):
            manager._show_final_tool_result()

            # Should only show valid_param and another_valid
            print_calls = str(mock_print.call_args_list)
            assert "valid_param" in print_calls
            assert "another_valid" in print_calls
            assert "empty_param" not in print_calls
            assert "none_param" not in print_calls

    # ==================== INVOCATION AND RESULT TESTS ====================

    def test_show_tool_invocation(self, manager):
        """Test showing tool invocation."""
        from chuk_term.ui import output

        tool_name = "test_tool"
        arguments = {"param": "value"}

        with patch.object(output, "tool_call") as mock_tool_call:
            manager._show_tool_invocation(tool_name, arguments)
            mock_tool_call.assert_called_once_with(tool_name, arguments)

    def test_show_tool_result_success(self, manager):
        """Test showing successful tool result."""
        from chuk_term.ui import output

        tool_info = {"name": "test_tool"}
        result = '{"status": "success"}'
        elapsed = 1.5

        with (
            patch.object(output, "success") as mock_success,
            patch.object(output, "code") as mock_code,
            patch.object(output, "print"),
        ):
            manager._show_tool_result(tool_info, result, elapsed, success=True)

            mock_success.assert_called_once()
            assert "Completed" in mock_success.call_args[0][0]
            mock_code.assert_called_once()  # JSON formatted

    def test_show_tool_result_failure(self, manager):
        """Test showing failed tool result."""
        from chuk_term.ui import output

        tool_info = {"name": "test_tool"}
        result = "Error message"
        elapsed = 0.5

        with (
            patch.object(output, "error") as mock_error,
            patch.object(output, "print"),
        ):
            manager._show_tool_result(tool_info, result, elapsed, success=False)

            mock_error.assert_called_once()
            assert "Failed" in mock_error.call_args[0][0]

    # ==================== INTEGRATION TESTS ====================

    def test_full_streaming_workflow(self, manager):
        """Test complete streaming workflow."""
        with (
            patch("builtins.print"),
            patch("chuk_term.ui.terminal.clear_line"),
            patch("chuk_term.ui.output.print"),
        ):
            # Start streaming
            manager.start_streaming()
            assert manager.is_streaming

            # Update content
            manager.update_streaming("Hello")
            manager.update_streaming(" world")
            assert manager.streaming_content == "Hello world"

            # Finish streaming
            manager.finish_streaming()
            assert not manager.is_streaming

    def test_full_tool_execution_workflow(self, manager):
        """Test complete tool execution workflow."""
        with (
            patch("builtins.print"),
            patch("chuk_term.ui.terminal.clear_line"),
            patch("chuk_term.ui.output.success"),
        ):
            # Start tool execution
            manager.start_tool_execution("test_tool", {"param": "value"})
            assert manager.is_tool_executing
            assert manager.current_tool is not None

            # Finish execution
            manager.finish_tool_execution("success result", success=True)
            assert not manager.is_tool_executing
            assert manager.current_tool is None
