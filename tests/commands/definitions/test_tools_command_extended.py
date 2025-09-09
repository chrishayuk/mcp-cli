"""Extended tests for the tools command."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from mcp_cli.commands.definitions.tools import ToolsCommand


class TestToolsCommand:
    """Test the ToolsCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ToolsCommand instance."""
        return ToolsCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "tools"
        assert command.description == "List and inspect MCP tools"
        assert len(command.parameters) > 0

    @pytest.mark.asyncio
    async def test_tools_list_all(self, command):
        """Test listing all tools."""
        # Mock context and tool manager
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "Test tool description"
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {}
        
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
    async def test_tools_filter_by_server(self, command):
        """Test filtering tools by server."""
        # Mock tools from different servers
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.namespace = "server1"
        mock_tool1.description = "Tool 1"
        mock_tool1.fully_qualified_name = "server1.tool1"
        mock_tool1.parameters = {}
        
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.namespace = "server2"
        mock_tool2.description = "Tool 2"
        mock_tool2.fully_qualified_name = "server2.tool2"
        mock_tool2.parameters = {}
        
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])
        
        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx
            
            result = await command.execute(args=["server1"])
            
            assert result.success is True
            mock_tm.get_unique_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_show_detail(self, command):
        """Test showing detailed tool information."""
        # Mock tool with parameters
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "Test tool with parameters"
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter",
                }
            },
            "required": ["param1"]
        }
        
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        
        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx
            
            result = await command.execute(args=["test_tool"])
            
            assert result.success is True
            mock_tm.get_unique_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_no_manager(self, command):
        """Test when no tool manager is available."""
        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None
            
            result = await command.execute()
            
            assert result.success is False
            assert "No tool manager available" in result.error

    @pytest.mark.asyncio
    async def test_tools_invalid_filter(self, command):
        """Test with invalid filter (no matching server or tool)."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "Test tool"
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {}
        
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        
        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx
            
            result = await command.execute(args=["nonexistent"])
            
            assert result.success is False
            assert "No tool or server found" in result.error

    @pytest.mark.asyncio
    async def test_tools_raw_output(self, command):
        """Test raw JSON output."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "Test tool"
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {}
        
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        
        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx
            
            result = await command.execute(raw=True)
            
            assert result.success is True
            assert result.data is not None

    @pytest.mark.asyncio
    async def test_tools_with_details_flag(self, command):
        """Test with --details flag for full descriptions."""
        # Mock tool with very long description
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.namespace = "test_server"
        mock_tool.description = "A" * 100  # Long description
        mock_tool.fully_qualified_name = "test_server.test_tool"
        mock_tool.parameters = {}
        
        mock_tm = MagicMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        
        with patch("mcp_cli.commands.definitions.tools.get_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.tool_manager = mock_tm
            mock_get_ctx.return_value = mock_ctx
            
            result = await command.execute(details=True)
            
            assert result.success is True
            mock_tm.get_unique_tools.assert_called_once()