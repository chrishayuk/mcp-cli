"""Simplified tests for the tools command."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcp_cli.commands.definitions.tools import ToolsCommand
from mcp_cli.commands.base import UnifiedCommand


class TestToolsCommand:
    """Test the ToolsCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ToolsCommand instance."""
        return ToolsCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "tools"
        assert command.aliases == []
        assert "tools" in command.description.lower()

        # Check that it's a UnifiedCommand
        assert isinstance(command, UnifiedCommand)

        # Check parameters
        params = command.parameters
        param_names = [p.name for p in params]
        assert "filter" in param_names
        assert "raw" in param_names
        assert "details" in param_names

    @pytest.mark.asyncio
    async def test_execute_no_filter(self, command):
        """Test executing tools without a filter (list all)."""
        # Mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "Test tool"
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {}

        # Mock tool manager
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])

        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx

            result = await command.execute()

            assert result.success is True
            mock_tm.get_unique_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_server_filter(self, command):
        """Test executing with a server filter."""
        # Mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "echo_text"
        mock_tool1.namespace = "stdio"
        mock_tool1.description = "Echo text"
        mock_tool1.fully_qualified_name = "stdio.echo_text"
        mock_tool1.parameters = {}

        mock_tool2 = MagicMock()
        mock_tool2.name = "list_tables"
        mock_tool2.namespace = "stdio"
        mock_tool2.description = "List tables"
        mock_tool2.fully_qualified_name = "stdio.list_tables"
        mock_tool2.parameters = {}

        # Mock tool manager
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx

            # Filter by "echo" server
            result = await command.execute(args=["echo"])

            assert result.success is True
            mock_tm.get_unique_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_tool_filter(self, command):
        """Test executing with a tool name filter (show details)."""
        # Mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "Test tool with details"
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {
            "properties": {"param1": {"type": "string", "description": "Test param"}},
            "required": ["param1"],
        }

        # Mock tool manager
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])

        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx

            result = await command.execute(args=["test_tool"])

            assert result.success is True
            mock_tm.get_unique_tools.assert_called_once()
