"""Tests for the providers command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.providers.providers import ProviderCommand
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
        # When no subcommand is provided, it should list providers
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_providers.return_value = ["openai", "anthropic"]
            mock_ctx.llm_manager.get_current_provider.return_value = "openai"
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute()

                    # Should default to list
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test executing the list subcommand."""
        # Get the list subcommand
        list_cmd = command.subcommands.get("list")
        assert list_cmd is not None

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_providers.return_value = ["ollama"]
            mock_ctx.llm_manager.get_current_provider.return_value = "ollama"
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await list_cmd.execute()

                    assert result.success is True
                    mock_ctx.llm_manager.list_providers.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_subcommand(self, command):
        """Test executing with an invalid subcommand."""
        # The ProviderCommand treats unknown subcommands as provider names
        # So we need to test with args that would be an invalid provider
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            # Simulate the action failing for an invalid provider
            mock_ctx.llm_manager.set_provider.side_effect = Exception(
                "Provider not found: invalid"
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["invalid"])

                assert result.success is False
                assert "Failed to switch provider" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_args_error(self, command):
        """Test error handling when listing with no args fails."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_providers.side_effect = Exception(
                "Connection failed"
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=[])

                assert result.success is False
                assert "Failed to list providers" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_subcommand(self, command):
        """Test executing the set subcommand."""
        set_cmd = command.subcommands.get("set")
        assert set_cmd is not None

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_provider.return_value = None
            with patch("chuk_term.ui.output"):
                result = await set_cmd.execute(args=["ollama"])

                assert result.success is True
                mock_ctx.llm_manager.set_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_set_error(self, command):
        """Test error handling in set subcommand."""
        set_cmd = command.subcommands.get("set")

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_provider.side_effect = Exception(
                "Invalid provider"
            )
            with patch("chuk_term.ui.output"):
                result = await set_cmd.execute(args=["invalid"])

                assert result.success is False
                assert "Failed to set provider" in result.error

    @pytest.mark.asyncio
    async def test_execute_show_subcommand(self, command):
        """Test executing the show subcommand."""
        show_cmd = command.subcommands.get("show")
        assert show_cmd is not None

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.get_current_provider.return_value = "openai"
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"
            with patch("chuk_term.ui.output"):
                result = await show_cmd.execute()

                assert result.success is True
                mock_ctx.llm_manager.get_current_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_show_error(self, command):
        """Test error handling in show subcommand."""
        show_cmd = command.subcommands.get("show")

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.get_current_provider.side_effect = Exception(
                "Failed to get info"
            )
            with patch("chuk_term.ui.output"):
                result = await show_cmd.execute()

                assert result.success is False
                assert "Failed to get provider info" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_known_subcommand(self, command):
        """Test that known subcommands are routed to parent."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.list_providers.return_value = ["openai"]
            mock_ctx.llm_manager.get_current_provider.return_value = "openai"
            mock_ctx.llm_manager.get_current_model.return_value = "gpt-4"
            mock_ctx.llm_manager.set_provider.return_value = None
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    # Test various known subcommand aliases
                    for subcmd in [
                        "list",
                        "ls",
                        "set",
                        "use",
                        "switch",
                        "show",
                        "current",
                        "status",
                    ]:
                        result = await command.execute(args=[subcmd])
                        # Should be handled by subcommand routing
                        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_provider_name_directly(self, command):
        """Test passing provider name directly (not a subcommand)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.llm_manager.set_provider.return_value = None
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["ollama"])

                assert result.success is True
                # Should treat "ollama" as a provider name to switch to
                mock_ctx.llm_manager.set_provider.assert_called_once_with("ollama")
