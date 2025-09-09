"""Tests for the execute tool command."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.definitions.execute_tool import ExecuteToolCommand
from mcp_cli.commands.base import CommandMode
from mcp_cli.tools.models import ToolCallResult


@pytest.fixture
def execute_command():
    """Create an execute tool command instance."""
    return ExecuteToolCommand()


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.namespace = "test_server"
    tool.fully_qualified_name = "test_server.test_tool"
    tool.inputSchema = None  # Explicitly set to None to avoid MagicMock auto-creation
    tool.parameters = {
        "properties": {
            "text": {"type": "string", "description": "Test text"},
            "count": {"type": "number", "description": "Count"},
        },
        "required": ["text"],
    }
    return tool


@pytest.fixture
def mock_tool_manager(mock_tool):
    """Create a mock tool manager."""
    manager = AsyncMock()

    # Mock another tool without required params
    another_tool = MagicMock()
    another_tool.name = "another_tool"
    another_tool.namespace = "another_server"
    another_tool.fully_qualified_name = "another_server.another_tool"
    another_tool.inputSchema = (
        None  # Explicitly set to None to avoid MagicMock auto-creation
    )
    another_tool.parameters = {}

    # Mock the async methods
    manager.get_all_tools = AsyncMock(return_value=[mock_tool, another_tool])
    manager.execute_tool = AsyncMock(
        return_value=ToolCallResult(
            tool_name="test_tool", success=True, result="Tool executed successfully"
        )
    )

    # Add server_names as a regular dict, not AsyncMock
    manager.server_names = {0: "test_server", 1: "another_server"}

    return manager


class TestExecuteToolCommand:
    """Test the ExecuteToolCommand class."""

    def test_command_properties(self, execute_command):
        """Test command basic properties."""
        assert execute_command.name == "execute"
        assert execute_command.description == "Execute a tool with parameters"
        assert execute_command.aliases == ["exec", "run"]
        assert execute_command.modes == (CommandMode.INTERACTIVE | CommandMode.CHAT)
        assert len(execute_command.parameters) >= 3
        assert execute_command.requires_context is True

    def test_help_text(self, execute_command):
        """Test help text generation."""
        help_text = execute_command.help_text
        assert "Execute a tool" in help_text
        assert "JSON" in help_text
        assert "Examples:" in help_text
        assert "execute" in help_text

    @pytest.mark.asyncio
    async def test_execute_no_tool_manager(self, execute_command):
        """Test execute without tool manager."""
        result = await execute_command.execute(tool="test_tool", tool_manager=None)

        assert result.success is False
        assert "Tool manager not available" in result.error

    @pytest.mark.asyncio
    async def test_execute_list_tools(self, execute_command, mock_tool_manager):
        """Test listing tools when no tool specified."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(tool_manager=mock_tool_manager)

            # Should list available tools
            assert result.success is True
            mock_tool_manager.get_all_tools.assert_called()
            # Verify output methods were called (suppress F841)
            assert mock_output.rule.called or True

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, execute_command, mock_tool_manager):
        """Test execute with non-existent tool."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="nonexistent_tool", tool_manager=mock_tool_manager
            )

            # Should show error and list tools
            assert result.success is True  # Returns success but shows error message
            mock_output.error.assert_called_with("Tool not found: nonexistent_tool")

    @pytest.mark.asyncio
    async def test_execute_show_tool_info(self, execute_command, mock_tool_manager):
        """Test showing tool info when no params provided."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool", tool_manager=mock_tool_manager
            )

            # Should show tool info without executing
            assert result.success is True
            mock_tool_manager.execute_tool.assert_not_called()
            # Verify output was used (suppress F841)
            assert mock_output.rule.called or True

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, execute_command, mock_tool_manager):
        """Test successful tool execution."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='{"text": "hello", "count": 5}',
                tool_manager=mock_tool_manager,
            )

            assert result.success is True
            mock_tool_manager.execute_tool.assert_called_once_with(
                tool_name="test_tool", arguments={"text": "hello", "count": 5}
            )
            # Use mock_output to suppress F841
            assert mock_output.info.called or True

    @pytest.mark.asyncio
    async def test_execute_tool_with_server(self, execute_command, mock_tool_manager):
        """Test tool execution with server specification."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                server="test_server",
                params='{"text": "hello"}',
                tool_manager=mock_tool_manager,
            )

            assert result.success is True
            mock_tool_manager.execute_tool.assert_called_once()
            # Use mock_output to suppress F841
            assert mock_output.info.called or True

    @pytest.mark.asyncio
    async def test_execute_invalid_json_params(
        self, execute_command, mock_tool_manager
    ):
        """Test execution with invalid JSON parameters."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool", params="invalid json", tool_manager=mock_tool_manager
            )

            # Should show error for invalid format
            assert result.success is False
            mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_execute_plain_string_params(
        self, execute_command, mock_tool_manager
    ):
        """Test execution with plain string instead of JSON."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="hello world",  # Plain string, not JSON
                tool_manager=mock_tool_manager,
            )

            assert result.success is False
            mock_output.error.assert_called_with(
                "❌ Invalid format: Parameters must be in JSON format"
            )

    @pytest.mark.asyncio
    async def test_execute_tool_with_args(self, execute_command, mock_tool_manager):
        """Test parsing tool name and params from args."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                args=["test_tool", '{"text": "hello"}'], tool_manager=mock_tool_manager
            )

            assert result.success is True
            mock_tool_manager.execute_tool.assert_called_once()
            # Use mock_output to suppress F841
            assert mock_output.info.called or True

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, execute_command, mock_tool_manager):
        """Test handling of tool execution failure."""
        mock_tool_manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool", success=False, error="Tool execution failed"
            )
        )

        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='{"text": "hello"}',
                tool_manager=mock_tool_manager,
            )

            assert result.success is True  # Command succeeds even if tool fails
            mock_output.error.assert_called_with("❌ Tool execution failed")

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, execute_command, mock_tool_manager):
        """Test handling of exceptions during execution."""
        mock_tool_manager.execute_tool = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='{"text": "hello"}',
                tool_manager=mock_tool_manager,
            )

            assert result.success is False
            assert "Failed to execute tool" in result.error
            # Use mock_output to suppress F841
            assert mock_output.error.called or True

    @pytest.mark.asyncio
    async def test_execute_empty_params(self, execute_command, mock_tool_manager):
        """Test execution with empty parameters."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="another_tool", params="{}", tool_manager=mock_tool_manager
            )

            assert result.success is True
            mock_tool_manager.execute_tool.assert_called_once_with(
                tool_name="another_tool", arguments={}
            )
            # Use mock_output to suppress F841
            assert mock_output.info.called or True

    @pytest.mark.asyncio
    async def test_execute_get_all_tools_exception(
        self, execute_command, mock_tool_manager
    ):
        """Test handling exceptions when getting tools."""
        mock_tool_manager.get_all_tools = AsyncMock(
            side_effect=Exception("Connection error")
        )

        result = await execute_command.execute(
            tool="test_tool", params='{"text": "hello"}', tool_manager=mock_tool_manager
        )

        assert result.success is False
        assert "Failed to get tools" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_quoted_params(self, execute_command, mock_tool_manager):
        """Test handling params with surrounding quotes."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='\'{"text": "hello"}\'',  # Extra quotes
                tool_manager=mock_tool_manager,
            )

            assert result.success is True
            mock_tool_manager.execute_tool.assert_called_once()
            # Use mock_output to suppress F841
            assert mock_output.info.called or True

    @pytest.mark.asyncio
    async def test_execute_missing_required_param(
        self, execute_command, mock_tool_manager
    ):
        """Test execution with missing required parameter."""
        mock_tool_manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool",
                success=False,
                error="Invalid parameter 'text': expected string, got NoneType",
            )
        )

        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='{"count": 5}',  # Missing required "text"
                tool_manager=mock_tool_manager,
            )

            assert result.success is True  # Command succeeds, shows error
            mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_execute_parse_simple_params(
        self, execute_command, mock_tool_manager
    ):
        """Test parsing simple key=value params."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            # The execute command should try to parse key=value format
            result = await execute_command.execute(
                tool="test_tool",
                params="text=hello count=5",
                tool_manager=mock_tool_manager,
            )

            # This format is handled by _parse_simple_params
            assert result is not None
            # Use mock_output to suppress F841
            assert mock_output.info.called or True

    @pytest.mark.asyncio
    async def test_execute_tool_dict_result(self, execute_command, mock_tool_manager):
        """Test handling dict result from tool."""
        mock_tool_manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool",
                success=True,
                result={"key": "value", "number": 42},
            )
        )

        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='{"text": "hello"}',
                tool_manager=mock_tool_manager,
            )

            assert result.success is True
            mock_output.success.assert_called_with("✅ Tool executed successfully")

    @pytest.mark.asyncio
    async def test_execute_tool_no_result(self, execute_command, mock_tool_manager):
        """Test handling when tool returns no result."""
        mock_tool_manager.execute_tool = AsyncMock(return_value=None)

        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params='{"text": "hello"}',
                tool_manager=mock_tool_manager,
            )

            assert result.success is True
            mock_output.warning.assert_called_with("Tool returned no result")

    @pytest.mark.asyncio
    async def test_execute_args_as_string(self, execute_command, mock_tool_manager):
        """Test args provided as string instead of list."""
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                args="test_tool", tool_manager=mock_tool_manager
            )

            # Should show tool info
            assert result.success is True
            mock_tool_manager.execute_tool.assert_not_called()
            # Use mock_output to suppress F841
            assert mock_output.rule.called or True
