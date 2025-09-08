"""Tests for the help command."""

import pytest
from unittest.mock import Mock, patch
from mcp_cli.commands.definitions.help import HelpCommand
from mcp_cli.commands.base import CommandMode


class TestHelpCommand:
    """Test the HelpCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a HelpCommand instance."""
        return HelpCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "help"
        assert command.aliases == ["h", "?"]
        assert "Show help information" in command.description
        assert command.modes == CommandMode.ALL

    @pytest.mark.asyncio
    async def test_execute_general_help(self, command):
        """Test showing general help."""
        with patch(
            "mcp_cli.commands.definitions.help.UnifiedCommandRegistry"
        ) as mock_registry_class:
            # Create a mock registry instance
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            # Mock the registry to return some commands
            mock_cmd1 = Mock()
            mock_cmd1.name = "test1"
            mock_cmd1.description = "Test command 1"
            mock_cmd1.hidden = False
            mock_cmd1.aliases = []

            mock_cmd2 = Mock()
            mock_cmd2.name = "test2"
            mock_cmd2.description = "Test command 2"
            mock_cmd2.hidden = False
            mock_cmd2.aliases = []

            mock_registry.list_commands.return_value = [mock_cmd1, mock_cmd2]

            # Patch the output functions to avoid actual printing
            with patch("mcp_cli.commands.definitions.help.output"):
                with patch("mcp_cli.commands.definitions.help.format_table"):
                    result = await command.execute()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_specific_command_help(self, command):
        """Test showing help for a specific command."""
        with patch(
            "mcp_cli.commands.definitions.help.UnifiedCommandRegistry"
        ) as mock_registry_class:
            # Create a mock registry instance
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            # Mock a specific command
            mock_cmd = Mock()
            mock_cmd.name = "test"
            mock_cmd.description = "Test command"
            mock_cmd.help_text = "Detailed help for test command"
            mock_cmd.parameters = []
            mock_cmd.aliases = []

            mock_registry.get.return_value = mock_cmd

            # Patch the output to avoid actual printing
            with patch("mcp_cli.commands.definitions.help.output"):
                result = await command.execute(command="test")

            assert result.success is True
            mock_registry.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self, command):
        """Test showing help for an unknown command."""
        with patch(
            "mcp_cli.commands.definitions.help.UnifiedCommandRegistry"
        ) as mock_registry_class:
            # Create a mock registry instance
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            mock_registry.get.return_value = None

            result = await command.execute(command="unknown")

            assert result.success is False
            assert "Unknown command" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_mode_filter(self, command):
        """Test showing help filtered by mode."""
        with patch(
            "mcp_cli.commands.definitions.help.UnifiedCommandRegistry"
        ) as mock_registry_class:
            # Create a mock registry instance
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            # Mock commands with different modes
            mock_cmd1 = Mock()
            mock_cmd1.name = "chat_only"
            mock_cmd1.description = "Chat only command"
            mock_cmd1.modes = CommandMode.CHAT
            mock_cmd1.hidden = False
            mock_cmd1.aliases = []

            mock_cmd2 = Mock()
            mock_cmd2.name = "all_modes"
            mock_cmd2.description = "All modes command"
            mock_cmd2.modes = CommandMode.ALL
            mock_cmd2.hidden = False
            mock_cmd2.aliases = []

            mock_registry.list_commands.return_value = [mock_cmd1, mock_cmd2]

            # Patch the output functions to avoid actual printing
            with patch("mcp_cli.commands.definitions.help.output"):
                with patch("mcp_cli.commands.definitions.help.format_table"):
                    # Execute with chat mode filter
                    result = await command.execute(mode=CommandMode.CHAT)

            assert result.success is True
            mock_registry.list_commands.assert_called_with(mode=CommandMode.CHAT)
