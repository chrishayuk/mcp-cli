"""Additional tests to improve tools command coverage."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mcp_cli.commands.definitions.tools import ToolsCommand


class TestToolsCommandCoverage:
    """Additional coverage tests for ToolsCommand."""

    def setup_method(self):
        """Setup for each test."""
        self.tools_cmd = ToolsCommand()

    def test_help_text_property(self):
        """Test that help_text property returns the correct text."""
        help_text = self.tools_cmd.help_text
        assert "List and inspect MCP tools" in help_text
        assert "Usage:" in help_text
        assert "/tools" in help_text
        assert "Examples:" in help_text
        assert "--raw" in help_text
        assert "--details" in help_text

    @pytest.mark.asyncio
    async def test_execute_no_tools_available(self):
        """Test executing when no tools are available."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()
        mock_tm.get_unique_tools = AsyncMock(return_value=[])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await self.tools_cmd.execute()

        assert result.success is True
        assert "No tools available" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_args_as_string(self):
        """Test executing with args parameter as string instead of list."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.fully_qualified_name = "test.test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.parameters = {}
        mock_tool.namespace = "test"

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute(args="test_tool")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_exception(self):
        """Test that exceptions are caught and returned as errors."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()
        mock_tm.get_unique_tools = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await self.tools_cmd.execute()

        assert result.success is False
        assert "Failed to list tools" in result.error
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_show_tool_details_with_echo_tool(self):
        """Test showing tool details for an echo tool (fallback detection)."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock echo tool not in the mapping
        mock_tool = MagicMock()
        mock_tool.name = "echo_custom"
        mock_tool.fully_qualified_name = "test.echo_custom"
        mock_tool.description = "Custom echo tool"
        mock_tool.parameters = {}
        mock_tool.namespace = "stdio"

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute(filter="echo_custom")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_show_tool_details_with_sql_tool(self):
        """Test showing tool details for a SQL tool (fallback detection)."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock SQL tool not in the mapping
        mock_tool = MagicMock()
        mock_tool.name = "custom_query_tool"
        mock_tool.fully_qualified_name = "test.custom_query_tool"
        mock_tool.description = "Custom SQL query tool"
        mock_tool.parameters = {}
        mock_tool.namespace = "stdio"

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute(filter="custom_query_tool")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_show_tool_details_with_parameters_and_defaults(self):
        """Test showing tool details with parameters that have default values."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock tool with parameters and defaults
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.fully_qualified_name = "test.test_tool"
        mock_tool.description = "Test tool"
        mock_tool.namespace = "test"
        mock_tool.parameters = {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter",
                    "default": "default_value",
                },
                "param2": {"type": "number", "description": "Second parameter"},
            },
            "required": ["param2"],
        }

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute(filter="test_tool")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_show_tool_details_without_parameters(self):
        """Test showing tool details for a tool with no parameters."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock tool without parameters
        mock_tool = MagicMock()
        mock_tool.name = "simple_tool"
        mock_tool.fully_qualified_name = "test.simple_tool"
        mock_tool.description = "Simple tool without parameters"
        mock_tool.namespace = "test"
        mock_tool.parameters = None

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute(filter="simple_tool")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_show_tools_table_with_truncation(self):
        """Test showing tools table with long descriptions that get truncated."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock tool with a long description
        mock_tool = MagicMock()
        mock_tool.name = "long_desc_tool"
        mock_tool.fully_qualified_name = "test.long_desc_tool"
        mock_tool.description = "This is a very long description that should be truncated because it exceeds the character limit for the table display format"
        mock_tool.namespace = "test"
        mock_tool.parameters = {}

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_show_tools_table_with_details_no_truncation(self):
        """Test showing tools table with --details flag to avoid truncation."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock tool with a long description
        mock_tool = MagicMock()
        mock_tool.name = "long_desc_tool"
        mock_tool.fully_qualified_name = "test.long_desc_tool"
        mock_tool.description = "This is a very long description that should NOT be truncated when using the details flag"
        mock_tool.namespace = "test"
        mock_tool.parameters = {}

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute(details=True)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_server_name_detection_with_namespace(self):
        """Test server name detection when tool has a proper namespace."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a mock tool with namespace
        mock_tool = MagicMock()
        mock_tool.name = "custom_tool"
        mock_tool.fully_qualified_name = "myserver.custom_tool"
        mock_tool.description = "Tool with namespace"
        mock_tool.namespace = "myserver"
        mock_tool.parameters = {}

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_server_name_detection_with_echo_pattern(self):
        """Test server name detection for tools with 'echo' in the name."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a tool with echo in name but not in map
        mock_tool = MagicMock()
        mock_tool.name = "my_echo_service"
        mock_tool.fully_qualified_name = "test.my_echo_service"
        mock_tool.description = "Echo-like tool"
        mock_tool.namespace = "stdio"
        mock_tool.parameters = {}

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_server_name_detection_with_sql_pattern(self):
        """Test server name detection for tools with SQL keywords in the name."""
        mock_context = MagicMock()
        mock_tm = AsyncMock()

        # Create a tool with SQL keywords in name but not in map
        mock_tool = MagicMock()
        mock_tool.name = "database_query_runner"
        mock_tool.fully_qualified_name = "test.database_query_runner"
        mock_tool.description = "Database query tool"
        mock_tool.namespace = "stdio"
        mock_tool.parameters = {}

        mock_tm.get_unique_tools = AsyncMock(return_value=[mock_tool])
        mock_context.tool_manager = mock_tm

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            with patch("mcp_cli.commands.definitions.tools.output"):
                result = await self.tools_cmd.execute()

        assert result.success is True
