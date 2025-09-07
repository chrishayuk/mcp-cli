# commands/test_resources_command.py
import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.resources import resources_action_async
from tests.conftest import setup_test_context


class DummyTMNoResources:
    async def list_resources(self):
        return []


class DummyTMWithResources:
    def __init__(self, data):
        self._data = data

    async def list_resources(self):
        return self._data


class DummyTMError:
    async def list_resources(self):
        raise RuntimeError("fail!")


@pytest.mark.asyncio
async def test_resources_action_error():
    """Test error handling in resources_action."""
    tm = DummyTMError()
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.resources.output") as mock_output:
        mock_output.error = MagicMock()

        result = await resources_action_async()

        assert result == []
        # Should call error with the exception message
        mock_output.error.assert_called()
        error_call = str(mock_output.error.call_args)
        assert "fail!" in error_call


@pytest.mark.asyncio
async def test_resources_action_no_resources():
    """Test when no resources are available."""
    tm = DummyTMNoResources()
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.resources.output") as mock_output:
        mock_output.info = MagicMock()

        result = await resources_action_async()

        assert result == []
        # Should call info about no resources
        mock_output.info.assert_called()
        info_call = str(mock_output.info.call_args)
        assert "No resources recorded" in info_call


@pytest.mark.asyncio
async def test_resources_action_with_resources():
    """Test when resources are available."""
    data = [
        {"server": "s1", "uri": "/path/1", "size": 500, "mimeType": "text/plain"},
        {
            "server": "s2",
            "uri": "/path/2",
            "size": 2048,
            "mimeType": "application/json",
        },
    ]
    tm = DummyTMWithResources(data)
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.resources.output") as mock_output:
        mock_output.print_table = MagicMock()

        result = await resources_action_async()

        assert result == data

        # Should print a table using print_table
        mock_output.print_table.assert_called_once()

        # The table was created with format_table and passed to print_table
        # We can't easily inspect the table structure without importing chuk_term internals,
        # but we can verify the method was called which means the table was created
