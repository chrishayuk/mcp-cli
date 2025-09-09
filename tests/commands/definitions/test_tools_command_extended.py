"""Extended tests for the tools command group and subcommands."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from mcp_cli.commands.definitions.tools import (
    ToolsCommand,
    ToolsListCommand,
    ToolsCallCommand,
    ToolsConfirmCommand,
)


class TestToolsListCommand:
    """Test the ToolsListCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ToolsListCommand instance."""
        return ToolsListCommand()

    def test_list_command_properties(self, command):
        """Test list command properties."""
        assert command.name == "list"
        assert command.aliases == ["ls", "show"]
        assert "List all available MCP tools" == command.description
        assert len(command.parameters) > 0

    @pytest.mark.asyncio
    async def test_list_command_execute(self, command):
        """Test executing list command."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {
                "tools": [{"name": "test_tool", "description": "Test"}]
            }

            result = await command.execute()

            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_command_with_raw_output(self, command):
        """Test list command with raw output."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {
                "tools": [{"name": "test_tool", "description": "Test"}]
            }

            result = await command.execute(raw=True)

            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_command_with_details(self, command):
        """Test list command with details flag."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "Test",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }

            result = await command.execute(details=True)

            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_command_error_handling(self, command):
        """Test list command error handling."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.side_effect = Exception("Connection failed")

            result = await command.execute()

            assert result.success is False
            assert "Failed to list tools" in result.error


class TestToolsCallCommand:
    """Test the ToolsCallCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ToolsCallCommand instance."""
        return ToolsCallCommand()

    @pytest.fixture
    def mock_context(self):
        """Create a mock context."""
        context = MagicMock()
        context.tool_manager = MagicMock()
        return context

    def test_call_command_properties(self, command):
        """Test call command properties."""
        assert command.name == "call"
        assert command.aliases == ["run", "execute"]
        assert "Call a specific" in command.description

    @pytest.mark.asyncio
    async def test_call_command_without_tool_manager(self, command):
        """Test call command without tool manager."""
        with patch("mcp_cli.commands.definitions.tools.get_context", return_value=None):
            result = await command.execute()

            assert result.success is False
            assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_call_command_without_tool_name(self, command, mock_context):
        """Test call command without tool name."""
        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await command.execute()

            assert result.success is False
            assert "Tool name is required" in result.error

    @pytest.mark.asyncio
    async def test_call_command_success(self, command, mock_context):
        """Test successful tool call."""
        mock_result = {"result": "test result"}
        mock_context.tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await command.execute(tool_name="test_tool", args={"key": "value"})

            assert result.success is True
            assert "Tool 'test_tool' executed successfully" in result.output
            assert "test result" in result.output
            mock_context.tool_manager.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_command_with_json_args(self, command, mock_context):
        """Test tool call with JSON string arguments."""
        mock_result = {"result": "test result"}
        mock_context.tool_manager.execute_tool = AsyncMock(return_value=mock_result)

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await command.execute(
                tool_name="test_tool", args='{"key": "value"}'
            )

            assert result.success is True
            assert "Tool 'test_tool' executed successfully" in result.output
            mock_context.tool_manager.execute_tool.assert_called_with(
                tool_name="test_tool", arguments={"key": "value"}
            )

    @pytest.mark.asyncio
    async def test_call_command_tool_error(self, command, mock_context):
        """Test tool call with error result."""
        # When execution raises an exception
        mock_context.tool_manager.execute_tool = AsyncMock(
            side_effect=Exception("Tool execution failed")
        )

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await command.execute(tool_name="test_tool")

            assert result.success is False
            assert "Failed to execute tool 'test_tool'" in result.error
            assert "Tool execution failed" in result.error

    @pytest.mark.asyncio
    async def test_call_command_execution_exception(self, command, mock_context):
        """Test tool call with execution exception."""
        mock_context.tool_manager.execute_tool = AsyncMock(
            side_effect=Exception("Execution failed")
        )

        with patch(
            "mcp_cli.commands.definitions.tools.get_context", return_value=mock_context
        ):
            result = await command.execute(tool_name="test_tool")

            assert result.success is False
            assert "Failed to execute tool" in result.error


class TestToolsConfirmCommand:
    """Test the ToolsConfirmCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ToolsConfirmCommand instance."""
        return ToolsConfirmCommand()

    @pytest.fixture
    def mock_pref_manager(self):
        """Create a mock preference manager."""
        manager = MagicMock()
        manager.get_tool_confirmation_mode.return_value = "smart"
        return manager

    def test_confirm_command_properties(self, command):
        """Test confirm command properties."""
        assert command.name == "confirm"
        assert command.aliases == ["confirmation"]
        assert "tool" in command.description.lower()

    @pytest.mark.asyncio
    async def test_confirm_command_show_status(self, command, mock_pref_manager):
        """Test showing current confirmation status."""
        with patch(
            "mcp_cli.utils.preferences.get_preference_manager",
            return_value=mock_pref_manager,
        ):
            result = await command.execute()

            assert result.success is True
            assert "Current tool confirmation mode: smart" in result.output

    @pytest.mark.asyncio
    async def test_confirm_command_set_always(self, command, mock_pref_manager):
        """Test setting confirmation to always."""
        with patch(
            "mcp_cli.utils.preferences.get_preference_manager",
            return_value=mock_pref_manager,
        ):
            result = await command.execute(mode="always")

            assert result.success is True
            assert "Tool confirmation mode set to: always" in result.output
            mock_pref_manager.set_tool_confirmation_mode.assert_called_once_with(
                "always"
            )

    @pytest.mark.asyncio
    async def test_confirm_command_set_never(self, command, mock_pref_manager):
        """Test setting confirmation to never."""
        with patch(
            "mcp_cli.utils.preferences.get_preference_manager",
            return_value=mock_pref_manager,
        ):
            result = await command.execute(mode="never")

            assert result.success is True
            assert "Tool confirmation mode set to: never" in result.output
            mock_pref_manager.set_tool_confirmation_mode.assert_called_once_with(
                "never"
            )

    @pytest.mark.asyncio
    async def test_confirm_command_set_smart(self, command, mock_pref_manager):
        """Test setting confirmation to smart."""
        with patch(
            "mcp_cli.utils.preferences.get_preference_manager",
            return_value=mock_pref_manager,
        ):
            result = await command.execute(mode="smart")

            assert result.success is True
            assert "Tool confirmation mode set to: smart" in result.output
            mock_pref_manager.set_tool_confirmation_mode.assert_called_once_with(
                "smart"
            )

    @pytest.mark.asyncio
    async def test_confirm_command_invalid_mode(self, command, mock_pref_manager):
        """Test with invalid mode."""
        with patch(
            "mcp_cli.utils.preferences.get_preference_manager",
            return_value=mock_pref_manager,
        ):
            result = await command.execute(mode="invalid")

            assert result.success is False
            assert "Invalid mode" in result.error
            assert "always" in result.error
            assert "never" in result.error
            assert "smart" in result.error


class TestToolsCommandIntegration:
    """Test the ToolsCommand group integration."""

    @pytest.fixture
    def command(self):
        """Create a ToolsCommand instance."""
        return ToolsCommand()

    def test_command_help_text(self, command):
        """Test command help text."""
        help_text = command.help_text
        assert "Manage and interact with MCP tools" in help_text
        assert "list" in help_text
        assert "call" in help_text
        assert "confirm" in help_text

    @pytest.mark.asyncio
    async def test_default_to_list(self, command):
        """Test that tools command defaults to list subcommand."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {"tools": []}

            # No subcommand provided
            result = await command.execute()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_args(self, command):
        """Test executing with args passed through."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {"tools": []}

            result = await command.execute(subcommand="list", raw=True)

            assert result.success is True
