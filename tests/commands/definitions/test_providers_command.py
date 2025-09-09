"""Tests for the providers command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.definitions.providers import ProviderCommand
from mcp_cli.commands.base import CommandGroup


class TestProviderCommand:
    """Test the ProviderCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ProviderCommand instance."""
        return ProviderCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "providers"
        assert command.aliases == []  # No aliases in the implementation
        assert "List available LLM providers" in command.description

        # Check that it's a command group with subcommands
        assert isinstance(command, CommandGroup)
        assert "list" in command.subcommands
        assert "set" in command.subcommands
        assert "show" in command.subcommands

    @pytest.mark.asyncio
    async def test_execute_no_subcommand(self, command):
        """Test executing providers without a subcommand."""
        # When no subcommand is provided, it should use the default (list)
        with patch(
            "mcp_cli.commands.actions.providers.provider_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "providers": [{"name": "openai", "status": "configured"}]
            }

            result = await command.execute()

            # Should default to list subcommand
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test executing the list subcommand."""
        # Get the list subcommand
        list_cmd = command.subcommands.get("list")
        assert list_cmd is not None

        with patch(
            "mcp_cli.commands.actions.providers.provider_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "providers": [{"name": "ollama", "status": "active"}]
            }

            result = await list_cmd.execute()

            assert result.success is True
            mock_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_subcommand(self, command):
        """Test executing with an invalid subcommand."""
        # The ProviderCommand treats unknown subcommands as provider names
        # So we need to test with args that would be an invalid provider
        with patch(
            "mcp_cli.commands.actions.providers.provider_action_async"
        ) as mock_action:
            # Simulate the action failing for an invalid provider
            mock_action.side_effect = Exception("Provider not found: invalid")

            result = await command.execute(args=["invalid"])

            assert result.success is False
            assert "Failed to switch provider" in result.error
