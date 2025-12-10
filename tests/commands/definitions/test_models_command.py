"""Tests for the models command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.providers.models import ModelCommand
from mcp_cli.commands.base import CommandGroup


class TestModelCommand:
    """Test the ModelCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ModelCommand instance."""
        return ModelCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "models"
        assert command.aliases == ["model"]
        assert "Manage LLM models" in command.description

        # Check that it's a command group with subcommands
        assert isinstance(command, CommandGroup)
        assert "list" in command.subcommands
        assert "set" in command.subcommands
        assert "show" in command.subcommands

    @pytest.mark.asyncio
    async def test_execute_no_subcommand(self, command):
        """Test executing models without a subcommand."""
        # When no subcommand is provided, it should show current model status
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"
            mock_ctx.llm_manager.get_current_provider.return_value = "openai"

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            # Should show current model status
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test executing the list subcommand."""
        # Get the list subcommand
        list_cmd = command.subcommands.get("list")
        assert list_cmd is not None

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"

            with patch("chuk_term.ui.output"):
                result = await list_cmd.execute()

            assert result.success is True
            mock_ctx.llm_manager.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_subcommand(self, command):
        """Test executing with an invalid subcommand."""
        # The ModelCommand treats unknown subcommands as model names
        # So we need to test with args that would be an invalid model
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            # Simulate the model switch failing for an invalid model
            mock_ctx.llm_manager.set_model.side_effect = Exception("Model not found: invalid")

            result = await command.execute(args=["invalid"])

            assert result.success is False
            assert "Failed to switch model" in result.error
