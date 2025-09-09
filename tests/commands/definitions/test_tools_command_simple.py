"""Simplified tests for the tools command group."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.definitions.tools import ToolsCommand
from mcp_cli.commands.base import CommandGroup


class TestToolsCommandGroup:
    """Test the ToolsCommand group implementation."""

    @pytest.fixture
    def command(self):
        """Create a ToolsCommand instance."""
        return ToolsCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "tools"
        assert command.aliases == []
        assert "tools" in command.description.lower()

        # Check that it's a command group with subcommands
        assert isinstance(command, CommandGroup)
        assert "list" in command.subcommands
        assert "call" in command.subcommands
        assert "confirm" in command.subcommands

    @pytest.mark.asyncio
    async def test_execute_no_subcommand(self, command):
        """Test executing tools without a subcommand."""
        # When no subcommand is provided, it defaults to 'list' for tools command
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {
                "tools": [{"name": "test_tool", "description": "Test"}]
            }

            result = await command.execute()

            # Should default to list subcommand for tools
            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test executing the list subcommand through the group."""
        with patch("mcp_cli.commands.actions.tools.tools_action_async") as mock_action:
            mock_action.return_value = {
                "tools": [{"name": "test_tool", "description": "Test"}]
            }

            # Execute through the group
            result = await command.execute(subcommand="list")

            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_subcommand(self, command):
        """Test executing with an invalid subcommand."""
        result = await command.execute(subcommand="invalid")

        assert result.success is False
        assert "Unknown tools subcommand" in result.error
