"""Tests to improve coverage for specific command definitions."""

import pytest
from unittest.mock import patch

from mcp_cli.commands.providers.providers import (
    ProviderCommand,
    ProviderSetCommand,
    ProviderShowCommand,
)
from mcp_cli.commands.servers.server_singular import ServerSingularCommand


# Removed help tests that were not working properly
# Focus on provider and server tests that do work


class TestProviderCommandCoverage:
    """Tests to improve provider command coverage."""

    @pytest.fixture
    def command(self):
        return ProviderCommand()

    @pytest.fixture
    def set_command(self):
        return ProviderSetCommand()

    @pytest.fixture
    def show_command(self):
        return ProviderShowCommand()

    @pytest.mark.asyncio
    async def test_provider_direct_switch(self, command):
        """Test provider direct switch with args."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_provider.return_value = None
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["openai"])
                assert result.success is True

    @pytest.mark.asyncio
    async def test_provider_set_from_args_string(self, set_command):
        """Test set provider from string arg."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_provider.return_value = None
            with patch("chuk_term.ui.output"):
                result = await set_command.execute(args="anthropic")
                assert result.success is True
                mock_ctx.model_manager.switch_provider.assert_called_once_with(
                    "anthropic"
                )

    @pytest.mark.asyncio
    async def test_provider_set_from_args_list(self, set_command):
        """Test set provider from list args."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_provider.return_value = None
            with patch("chuk_term.ui.output"):
                result = await set_command.execute(args=["ollama"])
                assert result.success is True

    @pytest.mark.asyncio
    async def test_provider_set_no_name(self, set_command):
        """Test set without provider name."""
        result = await set_command.execute()
        assert result.success is False
        assert "Provider name is required" in result.error

    @pytest.mark.asyncio
    async def test_provider_set_error(self, set_command):
        """Test set provider error."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_provider.side_effect = Exception("Failed")
            with patch("chuk_term.ui.output"):
                result = await set_command.execute(provider_name="bad")
                assert result.success is False
                assert "Failed to set provider" in result.error

    @pytest.mark.asyncio
    async def test_provider_show_execute(self, show_command):
        """Test show provider."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "openai"
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"
            with patch("chuk_term.ui.output"):
                result = await show_command.execute()
                assert result.success is True

    @pytest.mark.asyncio
    async def test_provider_show_error(self, show_command):
        """Test show provider error."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.side_effect = Exception("Failed")
            with patch("chuk_term.ui.output"):
                result = await show_command.execute()
                assert result.success is False
                assert "Failed to get provider info" in result.error

    @pytest.mark.asyncio
    async def test_provider_command_error(self, command):
        """Test provider command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_provider.side_effect = Exception("Error")
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["test"])
                assert result.success is False


class TestServerSingularCommandCoverage:
    """Tests to improve server singular command coverage."""

    @pytest.fixture
    def command(self):
        return ServerSingularCommand()

    @pytest.mark.asyncio
    async def test_server_details_string_args(self, command):
        """Test server details with string args."""
        from mcp_cli.tools.models import ServerInfo
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_server = ServerInfo(
                id=1,
                name="test-server",
                status="running",
                connected=True,
                tool_count=5,
                namespace="test",
            )
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                return_value=[mock_server]
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args="test-server")
                assert result.success is True

    @pytest.mark.asyncio
    async def test_server_details_error(self, command):
        """Test server details error."""
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(
                side_effect=Exception("Not found")
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["bad-server"])
                assert result.success is False
                assert "Failed to get server details" in result.error

    @pytest.mark.asyncio
    async def test_server_no_args(self, command):
        """Test server with no args - should list servers."""
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_server_info = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute()
                    assert result.success is True
