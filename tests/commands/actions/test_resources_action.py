"""Tests for resources action."""

from unittest.mock import MagicMock, patch
import pytest

from mcp_cli.commands.actions.resources import (
    resources_action_async,
    resources_action,
    _human_size,
)


def test_human_size():
    """Test the _human_size helper function."""
    assert _human_size(None) == "-"
    assert _human_size(-1) == "-"
    assert _human_size(0) == "0 B"
    assert _human_size(500) == "500 B"
    assert _human_size(1024) == "1 KB"
    assert _human_size(1536) == "2 KB"  # 1.5 KB rounded up
    assert _human_size(1024 * 1024) == "1 MB"
    assert _human_size(1024 * 1024 * 1024) == "1 GB"
    assert _human_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


@pytest.fixture
def mock_context():
    """Create a mock context with tool manager."""
    context = MagicMock()
    tool_manager = MagicMock()
    context.tool_manager = tool_manager
    return context, tool_manager


@pytest.fixture
def sample_resources():
    """Create sample resource data."""
    return [
        {
            "server": "test_server",
            "uri": "file:///test.txt",
            "size": 1024,
            "mimeType": "text/plain",
        },
        {
            "server": "another_server",
            "uri": "file:///data.json",
            "size": 2048,
            "mimeType": "application/json",
        },
    ]


@pytest.mark.asyncio
async def test_resources_action_async_basic(mock_context, sample_resources):
    """Test basic resources action async."""
    context, tool_manager = mock_context
    tool_manager.list_resources.return_value = sample_resources

    with (
        patch("mcp_cli.commands.actions.resources.get_context", return_value=context),
        patch("mcp_cli.commands.actions.resources.output") as mock_output,
        patch("mcp_cli.commands.actions.resources.format_table") as mock_format_table,
    ):
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        result = await resources_action_async()

        # Verify table creation and output
        mock_format_table.assert_called_once()
        table_data = mock_format_table.call_args[0][0]
        assert len(table_data) == 2
        assert table_data[0]["Size"] == "1 KB"
        assert table_data[1]["Size"] == "2 KB"

        mock_output.print_table.assert_called_once_with(mock_table)

        # Verify result
        assert result == sample_resources


@pytest.mark.asyncio
async def test_resources_action_async_no_tool_manager():
    """Test resources action when no tool manager available."""
    context = MagicMock()
    context.tool_manager = None

    with (
        patch("mcp_cli.commands.actions.resources.get_context", return_value=context),
        patch("mcp_cli.commands.actions.resources.output") as mock_output,
    ):
        result = await resources_action_async()

        mock_output.error.assert_called_once_with("No tool manager available")
        assert result == []


@pytest.mark.asyncio
async def test_resources_action_async_no_resources(mock_context):
    """Test resources action when no resources available."""
    context, tool_manager = mock_context
    tool_manager.list_resources.return_value = []

    with (
        patch("mcp_cli.commands.actions.resources.get_context", return_value=context),
        patch("mcp_cli.commands.actions.resources.output") as mock_output,
    ):
        result = await resources_action_async()

        mock_output.info.assert_called_once_with("No resources recorded.")
        assert result == []


@pytest.mark.asyncio
async def test_resources_action_async_exception(mock_context):
    """Test resources action when exception occurs."""
    context, tool_manager = mock_context
    tool_manager.list_resources.side_effect = Exception("Test error")

    with (
        patch("mcp_cli.commands.actions.resources.get_context", return_value=context),
        patch("mcp_cli.commands.actions.resources.output") as mock_output,
    ):
        result = await resources_action_async()

        mock_output.error.assert_called_once_with("Test error")
        assert result == []


@pytest.mark.asyncio
async def test_resources_action_async_awaitable_result(mock_context, sample_resources):
    """Test resources action when list_resources returns an awaitable."""
    context, tool_manager = mock_context

    # Make list_resources return a coroutine
    async def async_list_resources():
        return sample_resources

    tool_manager.list_resources.return_value = async_list_resources()

    with (
        patch("mcp_cli.commands.actions.resources.get_context", return_value=context),
        patch("mcp_cli.commands.actions.resources.output"),
        patch("mcp_cli.commands.actions.resources.format_table") as mock_format_table,
    ):
        mock_format_table.return_value = MagicMock()

        result = await resources_action_async()

        assert result == sample_resources


def test_resources_action_sync_wrapper():
    """Test the sync wrapper function."""
    with patch("mcp_cli.commands.actions.resources.run_blocking") as mock_run_blocking:
        mock_run_blocking.return_value = []

        result = resources_action()

        mock_run_blocking.assert_called_once()
        assert result == []
