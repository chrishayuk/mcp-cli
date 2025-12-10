"""Tests for the tool history command definition."""

import pytest
import json
from unittest.mock import MagicMock, patch

from mcp_cli.commands.tools.tool_history import ToolHistoryCommand
from mcp_cli.commands.base import CommandMode


@pytest.fixture
def tool_history_command():
    """Create a tool history command instance."""
    return ToolHistoryCommand()


@pytest.fixture
def mock_chat_context():
    """Create a mock chat context with tool history."""
    context = MagicMock()
    context.tool_history = [
        {
            "tool": "test_tool_1",
            "arguments": {"arg1": "value1"},
            "result": "Success",
            "success": True,
        },
        {
            "tool": "test_tool_2",
            "arguments": {"arg2": "value2", "arg3": "value3"},
            "result": "Another result",
            "success": True,
        },
        {
            "tool": "test_tool_3",
            "arguments": {"error": "test"},
            "result": "Failed",
            "success": False,
        },
    ]
    return context


def test_tool_history_command_properties(tool_history_command):
    """Test tool history command properties."""
    assert tool_history_command.name == "toolhistory"
    assert tool_history_command.aliases == ["th"]
    assert (
        tool_history_command.description == "View history of tool calls in this session"
    )
    assert tool_history_command.modes == CommandMode.CHAT
    assert len(tool_history_command.parameters) == 3
    assert "Inspect the history" in tool_history_command.help_text


@pytest.mark.asyncio
async def test_tool_history_without_context(tool_history_command):
    """Test tool history command without chat context."""
    result = await tool_history_command.execute()

    assert result.success is False
    assert result.error == "Tool history command requires chat context."


@pytest.mark.asyncio
async def test_tool_history_no_history_attribute(tool_history_command):
    """Test when chat context has no tool_history attribute."""
    mock_context = MagicMock(spec=[])  # No tool_history attribute

    result = await tool_history_command.execute(chat_context=mock_context)

    assert result.success is True
    assert result.output == "No tool history available."


@pytest.mark.asyncio
async def test_tool_history_empty_history(tool_history_command):
    """Test when tool history is empty."""
    mock_context = MagicMock()
    mock_context.tool_history = []

    result = await tool_history_command.execute(chat_context=mock_context)

    assert result.success is True
    assert result.output == "No tool calls have been made yet."


@pytest.mark.asyncio
async def test_tool_history_table_view(tool_history_command, mock_chat_context):
    """Test default table view of tool history."""
    with patch("mcp_cli.commands.tools.tool_history.output") as mock_output:
        with patch(
            "mcp_cli.commands.definitions.tool_history.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            result = await tool_history_command.execute(chat_context=mock_chat_context)

            assert result.success is True

            # Check table data
            table_data = mock_format_table.call_args[0][0]
            assert len(table_data) == 3
            assert table_data[0]["Tool"] == "test_tool_1"
            assert table_data[2]["Status"] == "✗"  # Failed tool

            mock_output.print_table.assert_called_once_with("formatted_table")


@pytest.mark.asyncio
async def test_tool_history_with_limit(tool_history_command, mock_chat_context):
    """Test tool history with limit parameter."""
    with patch("mcp_cli.commands.tools.tool_history.output") as mock_output:
        with patch(
            "mcp_cli.commands.definitions.tool_history.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            result = await tool_history_command.execute(
                chat_context=mock_chat_context, n=2
            )

            assert result.success is True

            # Should only show last 2 entries
            table_data = mock_format_table.call_args[0][0]
            assert len(table_data) == 2
            assert table_data[0]["Tool"] == "test_tool_2"

            mock_output.hint.assert_called_with("Showing last 2 of 3 total calls")


@pytest.mark.asyncio
async def test_tool_history_json_output(tool_history_command, mock_chat_context):
    """Test JSON output mode."""
    result = await tool_history_command.execute(
        chat_context=mock_chat_context, json=True
    )

    assert result.success is True

    # Parse JSON output
    output_data = json.loads(result.output)
    assert len(output_data) == 3
    assert output_data[0]["tool"] == "test_tool_1"


@pytest.mark.asyncio
async def test_tool_history_row_detail(tool_history_command, mock_chat_context):
    """Test detailed view of specific row."""
    with patch("mcp_cli.commands.tools.tool_history.output") as mock_output:
        result = await tool_history_command.execute(
            chat_context=mock_chat_context, row=2
        )

        assert result.success is True

        # Check panel was called with correct data
        panel_call = mock_output.panel.call_args
        assert "test_tool_2" in panel_call[0][0]
        assert "arg2" in panel_call[0][0]
        assert panel_call[1]["title"] == "Tool Call #2"


@pytest.mark.asyncio
async def test_tool_history_invalid_row(tool_history_command, mock_chat_context):
    """Test with invalid row number."""
    result = await tool_history_command.execute(chat_context=mock_chat_context, row=10)

    assert result.success is False
    assert "Invalid row number: 10" in result.error
    assert "Valid range: 1-3" in result.error


@pytest.mark.asyncio
async def test_tool_history_row_from_args_list(tool_history_command, mock_chat_context):
    """Test row number from args list."""
    with patch("mcp_cli.commands.tools.tool_history.output") as mock_output:
        result = await tool_history_command.execute(
            chat_context=mock_chat_context, args=["1"]
        )

        assert result.success is True
        assert "Tool Call #1" in mock_output.panel.call_args[1]["title"]


@pytest.mark.asyncio
async def test_tool_history_row_from_args_string(
    tool_history_command, mock_chat_context
):
    """Test row number from args string."""
    with patch("mcp_cli.commands.tools.tool_history.output") as mock_output:
        result = await tool_history_command.execute(
            chat_context=mock_chat_context, args="3"
        )

        assert result.success is True
        assert "Tool Call #3" in mock_output.panel.call_args[1]["title"]


@pytest.mark.asyncio
async def test_tool_history_invalid_args(tool_history_command, mock_chat_context):
    """Test with non-numeric args."""
    with patch("mcp_cli.commands.tools.tool_history.output") as mock_output:
        with patch(
            "mcp_cli.commands.definitions.tool_history.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            # Should fall back to table view
            result = await tool_history_command.execute(
                chat_context=mock_chat_context, args=["invalid"]
            )

            assert result.success is True
            mock_output.print_table.assert_called_once()


@pytest.mark.asyncio
async def test_tool_history_truncate_long_arguments(tool_history_command):
    """Test that long arguments are truncated in table view."""
    mock_context = MagicMock()
    mock_context.tool_history = [
        {
            "tool": "test_tool",
            "arguments": {"very_long_argument_name": "x" * 100},
            "result": "Success",
            "success": True,
        }
    ]

    with patch("mcp_cli.commands.tools.tool_history.output"):
        with patch(
            "mcp_cli.commands.definitions.tool_history.format_table"
        ) as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            result = await tool_history_command.execute(chat_context=mock_context)

            assert result.success is True

            # Check that arguments were truncated
            table_data = mock_format_table.call_args[0][0]
            assert len(table_data[0]["Arguments"]) == 50
            assert table_data[0]["Arguments"].endswith("...")


@pytest.mark.asyncio
async def test_tool_history_none_tool_history(tool_history_command):
    """Test when tool_history is None."""
    mock_context = MagicMock()
    mock_context.tool_history = None

    result = await tool_history_command.execute(chat_context=mock_context)

    assert result.success is True
    assert result.output == "No tool calls have been made yet."
