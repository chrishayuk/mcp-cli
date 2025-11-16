"""Tests for cmd action module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from mcp_cli.commands.actions.cmd import (
    cmd_action_async,
    _execute_tool_direct,
    _handle_tool_calls,
)


class TestCmdActionAsync:
    """Tests for cmd_action_async main function."""

    @pytest.mark.asyncio
    async def test_no_context(self):
        """Test when context is not initialized."""
        with patch("mcp_cli.context.get_context", return_value=None):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await cmd_action_async()
                mock_output.error.assert_called_once()
                assert "Context not initialized" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_no_tool_manager(self):
        """Test when context exists but no tool manager."""
        mock_context = MagicMock()
        mock_context.tool_manager = None

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await cmd_action_async()
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_tool_mode(self):
        """Test executing in tool mode."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch(
                "mcp_cli.commands.actions.cmd._execute_tool_direct"
            ) as mock_exec:
                await cmd_action_async(tool="test_tool", tool_args='{"arg": "value"}')
                mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_mode(self):
        """Test executing in prompt mode."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch(
                "mcp_cli.commands.actions.cmd._execute_prompt_mode"
            ) as mock_exec:
                await cmd_action_async(prompt="Test prompt")
                mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_input_file_mode(self):
        """Test executing with input file."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch(
                "mcp_cli.commands.actions.cmd._execute_prompt_mode"
            ) as mock_exec:
                await cmd_action_async(input_file="input.txt")
                mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_mode_specified(self):
        """Test when no operation mode is specified."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await cmd_action_async()
                mock_output.error.assert_called()
                assert "No operation specified" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test exception handling in cmd_action_async."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch(
                "mcp_cli.commands.actions.cmd._execute_tool_direct",
                side_effect=RuntimeError("Test error"),
            ):
                with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                    with pytest.raises(RuntimeError, match="Test error"):
                        await cmd_action_async(tool="test_tool")
                    mock_output.error.assert_called()


class TestExecuteToolDirect:
    """Tests for _execute_tool_direct function."""

    @pytest.mark.asyncio
    async def test_no_tool_manager(self):
        """Test when tool manager is not available."""
        mock_context = MagicMock()
        mock_context.tool_manager = None

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_tool_direct("test_tool", None, None, False)
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_json_args(self):
        """Test with invalid JSON in tool arguments."""
        mock_context = MagicMock()
        mock_context.tool_manager = MagicMock()

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_tool_direct("test_tool", "invalid json", None, False)
                mock_output.error.assert_called()
                assert "Invalid JSON" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_tool_execution_success_raw(self):
        """Test successful tool execution in raw mode."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.error = None
        mock_result.result = {"data": "test"}
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print") as mock_print:
                    await _execute_tool_direct(
                        "test_tool", '{"arg": "value"}', None, True
                    )
                    mock_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_execution_success_formatted(self):
        """Test successful tool execution in formatted mode."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.error = None
        mock_result.result = {"data": "test"}
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print") as mock_print:
                    await _execute_tool_direct("test_tool", None, None, False)
                    mock_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_execution_with_string_result(self):
        """Test tool execution when result is a string."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.error = None
        mock_result.result = "string result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                with patch("builtins.print") as mock_print:
                    await _execute_tool_direct("test_tool", None, None, True)
                    mock_print.assert_called_with("string result")

    @pytest.mark.asyncio
    async def test_tool_execution_to_file(self, tmp_path):
        """Test tool execution with output to file."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.error = None
        mock_result.result = {"data": "test"}
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        output_file = tmp_path / "output.json"

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_tool_direct("test_tool", None, str(output_file), False)
                assert output_file.exists()
                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self):
        """Test when tool execution fails."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Tool failed"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _execute_tool_direct("test_tool", None, None, False)
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_tool_execution_exception(self):
        """Test exception handling in tool execution."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_tool_manager.execute_tool = AsyncMock(
            side_effect=RuntimeError("Test error")
        )

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                with pytest.raises(RuntimeError):
                    await _execute_tool_direct("test_tool", None, None, False)
                mock_output.error.assert_called()


class TestHandleToolCalls:
    """Tests for _handle_tool_calls function."""

    @pytest.mark.asyncio
    async def test_no_tool_manager(self):
        """Test when tool manager is not available."""
        mock_context = MagicMock()
        mock_context.tool_manager = None

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(None, [], [], "response", 10, False)
                assert result == "response"

    @pytest.mark.asyncio
    async def test_tool_call_dict_format(self):
        """Test handling tool calls in dict format."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "final response", "tool_calls": []}
        )

        tool_calls = [
            {
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
                "id": "call_1",
            }
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(
                    mock_client, [], tool_calls, "response", 10, False
                )
                assert result == "final response"

    @pytest.mark.asyncio
    async def test_tool_call_object_format(self):
        """Test handling tool calls in object format."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "final response", "tool_calls": []}
        )

        # Mock object format
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        mock_tool_call.id = "call_1"

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(
                    mock_client, [], [mock_tool_call], "response", 10, False
                )
                assert result == "final response"

    @pytest.mark.asyncio
    async def test_tool_call_with_dict_args(self):
        """Test tool call with arguments already as dict."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "final response", "tool_calls": []}
        )

        tool_calls = [
            {
                "function": {"name": "test_tool", "arguments": {"arg": "value"}},
                "id": "call_1",
            }
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(
                    mock_client, [], tool_calls, "response", 10, False
                )
                assert result == "final response"

    @pytest.mark.asyncio
    async def test_tool_call_failure(self):
        """Test when tool execution fails."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Tool failed"
        mock_result.result = None
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "final response", "tool_calls": []}
        )

        tool_calls = [
            {"function": {"name": "test_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output"):
                result = await _handle_tool_calls(
                    mock_client, [], tool_calls, "response", 10, False
                )
                assert result == "final response"

    @pytest.mark.asyncio
    async def test_tool_call_exception(self):
        """Test when tool execution raises exception."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_tool_manager.execute_tool = AsyncMock(
            side_effect=RuntimeError("Tool error")
        )
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "final response", "tool_calls": []}
        )

        tool_calls = [
            {"function": {"name": "test_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                result = await _handle_tool_calls(
                    mock_client, [], tool_calls, "response", 10, False
                )
                assert result == "final response"
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_max_turns_reached(self):
        """Test when max turns is reached."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        # Always return more tool calls to hit max turns
        mock_client.create_completion = AsyncMock(
            return_value={
                "response": "continuing",
                "tool_calls": [
                    {
                        "function": {"name": "test_tool", "arguments": "{}"},
                        "id": "call_1",
                    }
                ],
            }
        )

        tool_calls = [
            {"function": {"name": "test_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _handle_tool_calls(
                    mock_client, [], tool_calls, "response", 2, False
                )
                mock_output.warning.assert_called()
                assert "Max turns" in str(mock_output.warning.call_args)

    @pytest.mark.asyncio
    async def test_raw_mode(self):
        """Test in raw mode (no info output)."""
        mock_context = MagicMock()
        mock_tool_manager = AsyncMock()
        mock_context.tool_manager = mock_tool_manager
        mock_context.model = "gpt-4"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "tool result"
        mock_tool_manager.execute_tool = AsyncMock(return_value=mock_result)
        mock_tool_manager.get_tools_for_llm = AsyncMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "final response", "tool_calls": []}
        )

        tool_calls = [
            {"function": {"name": "test_tool", "arguments": "{}"}, "id": "call_1"}
        ]

        with patch("mcp_cli.context.get_context", return_value=mock_context):
            with patch("mcp_cli.commands.actions.cmd.output") as mock_output:
                await _handle_tool_calls(
                    mock_client, [], tool_calls, "response", 10, True
                )
                # In raw mode, info should not be called
                assert mock_output.info.call_count == 0
