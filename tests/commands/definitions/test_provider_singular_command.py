"""Tests for the provider singular command definition."""

import pytest
from unittest.mock import patch

from mcp_cli.commands.providers.provider_singular import ProviderSingularCommand


@pytest.fixture
def provider_command():
    """Create a provider singular command instance."""
    return ProviderSingularCommand()


def test_provider_command_properties(provider_command):
    """Test provider command properties."""
    assert provider_command.name == "provider"
    assert provider_command.aliases == []
    assert (
        provider_command.description
        == "Show current provider status or switch providers"
    )
    assert "Show current LLM provider" in provider_command.help_text
    assert "/provider ollama" in provider_command.help_text


@pytest.mark.asyncio
async def test_provider_show_status_no_args(provider_command):
    """Test showing current provider status with no arguments."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.get_active_provider.return_value = "openai"
        mock_ctx.model_manager.get_active_model.return_value = "gpt-4"
        with patch("chuk_term.ui.output"):
            result = await provider_command.execute(args=[])

            assert result.success is True
            mock_ctx.model_manager.get_active_provider.assert_called_once()


@pytest.mark.asyncio
async def test_provider_show_status_error(provider_command):
    """Test error handling when showing provider status fails."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.get_active_provider.side_effect = Exception(
            "Connection failed"
        )
        with patch("chuk_term.ui.output"):
            result = await provider_command.execute(args=[])

            assert result.success is False
            assert "Failed to show provider status: Connection failed" in result.error


@pytest.mark.asyncio
async def test_provider_switch_to_provider(provider_command):
    """Test switching to a different provider."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.switch_provider.return_value = None
        with patch("chuk_term.ui.output"):
            result = await provider_command.execute(args=["ollama"])

            assert result.success is True
            mock_ctx.model_manager.switch_provider.assert_called_once_with("ollama")


@pytest.mark.asyncio
async def test_provider_switch_error(provider_command):
    """Test error handling when switching provider fails."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.switch_provider.side_effect = Exception(
            "Invalid provider"
        )
        with patch("chuk_term.ui.output"):
            result = await provider_command.execute(args=["invalid"])

            assert result.success is False
            assert "Failed to switch provider: Invalid provider" in result.error


@pytest.mark.asyncio
async def test_provider_list_subcommand(provider_command):
    """Test handling of list subcommand."""
    with patch("mcp_cli.context.get_context"):
        result = await provider_command.execute(args=["list"])

        assert result.success is False
        assert "Use /providers list" in result.error


@pytest.mark.asyncio
async def test_provider_ls_alias(provider_command):
    """Test handling of ls alias for list."""
    with patch("mcp_cli.context.get_context"):
        result = await provider_command.execute(args=["ls"])

        assert result.success is False
        assert "Use /providers ls" in result.error


@pytest.mark.asyncio
async def test_provider_set_subcommand(provider_command):
    """Test handling of set subcommand."""
    with patch("mcp_cli.context.get_context"):
        result = await provider_command.execute(
            args=["set", "openai", "api_key", "test-key"]
        )

        assert result.success is False
        assert "Use /providers set" in result.error


@pytest.mark.asyncio
async def test_provider_subcommand_error(provider_command):
    """Test error handling for subcommand failures."""
    with patch("mcp_cli.context.get_context"):
        result = await provider_command.execute(args=["set", "invalid"])

        assert result.success is False
        assert "Use /providers set" in result.error


@pytest.mark.asyncio
async def test_provider_with_string_args(provider_command):
    """Test handling of string arguments instead of list."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.switch_provider.return_value = None
        with patch("chuk_term.ui.output"):
            # Test with string argument
            result = await provider_command.execute(args="ollama")

            assert result.success is True
            mock_ctx.model_manager.switch_provider.assert_called_once_with("ollama")


@pytest.mark.asyncio
async def test_provider_with_multiple_args(provider_command):
    """Test handling of multiple arguments."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.switch_provider.return_value = None
        with patch("chuk_term.ui.output"):
            result = await provider_command.execute(args=["openai", "gpt-4"])

            assert result.success is True
            mock_ctx.model_manager.switch_provider.assert_called_once_with("openai")


@pytest.mark.asyncio
async def test_provider_with_no_kwargs(provider_command):
    """Test handling when no kwargs provided."""
    with patch("mcp_cli.context.get_context") as mock_get_ctx:
        mock_ctx = mock_get_ctx.return_value
        mock_ctx.model_manager.get_active_provider.return_value = "openai"
        mock_ctx.model_manager.get_active_model.return_value = "gpt-4"
        with patch("chuk_term.ui.output"):
            # No args key in kwargs
            result = await provider_command.execute()

            assert result.success is True
            mock_ctx.model_manager.get_active_provider.assert_called_once()
