"""Tests for tools action."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mcp_cli.commands.actions.tools_confirm import tools_action_async, tools_action
from mcp_cli.tools.models import ToolInfo


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager."""
    tm = MagicMock()
    tm.get_unique_tools = AsyncMock()
    return tm


@pytest.fixture
def sample_tools():
    """Create sample tool data."""
    return [
        ToolInfo(
            name="test_tool",
            namespace="test_server",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        ),
        ToolInfo(
            name="another_tool",
            namespace="another_server",
            description="Another tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        ),
    ]


@pytest.mark.asyncio
async def test_tools_action_async_basic(mock_tool_manager, sample_tools):
    """Test basic tools action async."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with (
        patch("mcp_cli.commands.actions.tools_confirm.output") as mock_output,
        patch(
            "mcp_cli.commands.actions.tools_confirm.create_tools_table"
        ) as mock_create_table,
    ):
        mock_table = MagicMock()
        mock_create_table.return_value = mock_table

        result = await tools_action_async(mock_tool_manager)

        # Verify output calls
        mock_output.info.assert_called_once_with(
            "\nFetching tool catalogue from all servers…"
        )
        mock_output.print_table.assert_called_once_with(mock_table)
        mock_output.success.assert_called_once_with("Total tools available: 2")

        # Verify result structure
        assert len(result) == 2
        assert result[0]["name"] == "test_tool"
        assert result[0]["namespace"] == "test_server"
        assert result[1]["name"] == "another_tool"


@pytest.mark.asyncio
async def test_tools_action_async_no_tools(mock_tool_manager):
    """Test tools action when no tools available."""
    mock_tool_manager.get_unique_tools.return_value = []

    with patch("mcp_cli.commands.actions.tools_confirm.output") as mock_output:
        result = await tools_action_async(mock_tool_manager)

        mock_output.info.assert_called_once()
        mock_output.warning.assert_called_once_with(
            "No tools available from any server."
        )
        assert result == []


@pytest.mark.asyncio
async def test_tools_action_async_raw_mode(mock_tool_manager, sample_tools):
    """Test tools action with raw JSON output."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch("mcp_cli.commands.actions.tools_confirm.output") as mock_output:
        await tools_action_async(mock_tool_manager, show_raw=True)

        # Should call json output instead of table
        mock_output.json.assert_called_once()

        # Verify JSON structure was output
        call_args = mock_output.json.call_args[0][0]
        json_data = json.loads(call_args)
        assert len(json_data) == 2
        assert json_data[0]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_tools_action_async_show_details(mock_tool_manager, sample_tools):
    """Test tools action with show_details=True."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with (
        patch("mcp_cli.commands.actions.tools_confirm.output"),
        patch(
            "mcp_cli.commands.actions.tools_confirm.create_tools_table"
        ) as mock_create_table,
    ):
        mock_table = MagicMock()
        mock_create_table.return_value = mock_table

        await tools_action_async(mock_tool_manager, show_details=True)

        # Verify create_tools_table was called with show_details=True
        mock_create_table.assert_called_once_with(sample_tools, show_details=True)


def test_tools_action_sync_wrapper(mock_tool_manager):
    """Test the sync wrapper function."""
    with patch(
        "mcp_cli.commands.actions.tools_confirm.run_blocking"
    ) as mock_run_blocking:
        mock_run_blocking.return_value = []

        result = tools_action(mock_tool_manager, show_details=True, show_raw=False)

        # Verify run_blocking was called
        mock_run_blocking.assert_called_once()
        assert result == []
