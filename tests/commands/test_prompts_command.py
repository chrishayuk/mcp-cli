# commands/test_prompts_command.py
import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.prompts import prompts_action_async
from tests.conftest import setup_test_context


class DummyTMNoPrompts:
    async def list_prompts(self):
        return []


class DummyTMWithPromptsSync:
    def __init__(self, data):
        self._data = data

    async def list_prompts(self):
        return self._data


class DummyTMWithPromptsAsync:
    async def list_prompts(self):
        return [{"server": "s1", "name": "n1", "description": "d1"}]


@pytest.mark.asyncio
async def test_prompts_action_no_prompts():
    """Test when no prompts are available."""
    tm = DummyTMNoPrompts()
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.prompts.output") as mock_output:
        mock_output.info = MagicMock()

        result = await prompts_action_async()

        assert result == []
        # Should call info about no prompts
        mock_output.info.assert_called()
        info_call = str(mock_output.info.call_args)
        assert "No prompts recorded" in info_call


@pytest.mark.asyncio
async def test_prompts_action_with_prompts_sync():
    """Test with synchronous prompts data."""
    data = [{"server": "srv", "name": "nm", "description": "desc"}]
    tm = DummyTMWithPromptsSync(data)
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.prompts.output") as mock_output:
        mock_output.print_table = MagicMock()

        result = await prompts_action_async()

        assert result == data

        # Should print a table using print_table
        mock_output.print_table.assert_called_once()

        # The table was created with format_table and passed to print_table
        # We can't easily inspect the table structure without importing chuk_term internals,
        # but we can verify the method was called which means the table was created


@pytest.mark.asyncio
async def test_prompts_action_with_prompts_async():
    """Test with asynchronous prompts data."""
    tm = DummyTMWithPromptsAsync()
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.prompts.output") as mock_output:
        mock_output.print_table = MagicMock()

        result = await prompts_action_async()

        assert isinstance(result, list) and len(result) == 1

        # Should print a table using print_table
        mock_output.print_table.assert_called_once()

        # The table was created with format_table and passed to print_table
        # We can't easily inspect the table structure without importing chuk_term internals,
        # but we can verify the method was called which means the table was created
