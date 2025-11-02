"""Tests for prompts action."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mcp_cli.commands.actions.prompts import (
    prompts_action_async,
    prompts_action,
    prompts_action_cmd,
)
from mcp_cli.commands.models import PromptInfoResponse


@pytest.fixture
def mock_context():
    """Create a mock context with tool manager."""
    context = MagicMock()
    tool_manager = MagicMock()
    context.tool_manager = tool_manager
    return context, tool_manager


@pytest.fixture
def sample_prompts():
    """Create sample prompt data."""
    return [
        {
            "server": "test_server",
            "name": "test_prompt",
            "description": "A test prompt",
        },
        {
            "server": "another_server",
            "name": "another_prompt",
            "description": "Another prompt",
        },
    ]


@pytest.mark.asyncio
async def test_prompts_action_async_basic(mock_context, sample_prompts):
    """Test basic prompts action async."""
    context, tool_manager = mock_context
    tool_manager.list_prompts.return_value = sample_prompts

    with (
        patch("mcp_cli.commands.actions.prompts.get_context", return_value=context),
        patch("mcp_cli.commands.actions.prompts.output") as mock_output,
        patch("mcp_cli.commands.actions.prompts.format_table") as mock_format_table,
    ):
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        result = await prompts_action_async()

        # Verify table creation and output
        mock_format_table.assert_called_once()
        mock_output.print_table.assert_called_once_with(mock_table)

        # Verify result - should be Pydantic models
        assert len(result) == len(sample_prompts)
        assert all(isinstance(p, PromptInfoResponse) for p in result)
        assert result[0].name == "test_prompt"
        assert result[0].server == "test_server"
        assert result[1].name == "another_prompt"
        assert result[1].server == "another_server"


@pytest.mark.asyncio
async def test_prompts_action_async_no_tool_manager():
    """Test prompts action when no tool manager available."""
    context = MagicMock()
    context.tool_manager = None

    with (
        patch("mcp_cli.commands.actions.prompts.get_context", return_value=context),
        patch("mcp_cli.commands.actions.prompts.output") as mock_output,
    ):
        result = await prompts_action_async()

        mock_output.error.assert_called_once_with("No tool manager available")
        assert result == []


@pytest.mark.asyncio
async def test_prompts_action_async_no_prompts(mock_context):
    """Test prompts action when no prompts available."""
    context, tool_manager = mock_context
    tool_manager.list_prompts.return_value = []

    with (
        patch("mcp_cli.commands.actions.prompts.get_context", return_value=context),
        patch("mcp_cli.commands.actions.prompts.output") as mock_output,
    ):
        result = await prompts_action_async()

        mock_output.info.assert_called_once_with("No prompts recorded.")
        assert result == []


@pytest.mark.asyncio
async def test_prompts_action_async_exception(mock_context):
    """Test prompts action when exception occurs."""
    context, tool_manager = mock_context
    tool_manager.list_prompts.side_effect = Exception("Test error")

    with (
        patch("mcp_cli.commands.actions.prompts.get_context", return_value=context),
        patch("mcp_cli.commands.actions.prompts.output") as mock_output,
    ):
        result = await prompts_action_async()

        mock_output.error.assert_called_once_with("Test error")
        assert result == []


@pytest.mark.asyncio
async def test_prompts_action_async_awaitable_result(mock_context, sample_prompts):
    """Test prompts action when list_prompts returns an awaitable."""
    context, tool_manager = mock_context

    # Make list_prompts return a coroutine
    async def async_list_prompts():
        return sample_prompts

    tool_manager.list_prompts.return_value = async_list_prompts()

    with (
        patch("mcp_cli.commands.actions.prompts.get_context", return_value=context),
        patch("mcp_cli.commands.actions.prompts.output"),
        patch("mcp_cli.commands.actions.prompts.format_table") as mock_format_table,
    ):
        mock_format_table.return_value = MagicMock()

        result = await prompts_action_async()

        # Verify result - should be Pydantic models
        assert len(result) == len(sample_prompts)
        assert all(isinstance(p, PromptInfoResponse) for p in result)


def test_prompts_action_sync_wrapper():
    """Test the sync wrapper function."""
    with patch("mcp_cli.commands.actions.prompts.run_blocking") as mock_run_blocking:
        mock_run_blocking.return_value = []

        result = prompts_action()

        mock_run_blocking.assert_called_once()
        assert result == []


@pytest.mark.asyncio
async def test_prompts_action_cmd_alias():
    """Test the cmd alias function."""
    with patch(
        "mcp_cli.commands.actions.prompts.prompts_action_async",
        new_callable=AsyncMock,
    ) as mock_prompts_action:
        mock_prompts_action.return_value = []

        result = await prompts_action_cmd()

        mock_prompts_action.assert_called_once()
        assert result == []
