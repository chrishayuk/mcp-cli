"""Tests for the clear command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.core.clear import ClearCommand


class TestClearCommand:
    """Test the ClearCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ClearCommand instance."""
        return ClearCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "clear"
        assert command.aliases == []  # No aliases in current implementation
        assert "Clear the terminal screen" in command.description
        assert command.requires_context is False  # Clear doesn't need context

    @pytest.mark.asyncio
    async def test_execute(self, command):
        """Test executing the clear command."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                # Mock get_context from the context module
                with patch("mcp_cli.context.get_context") as mock_context:
                    mock_context.return_value = None

                    result = await command.execute()

                    # Verify clear_screen was called
                    mock_clear.assert_called_once()
                    # Banner should not be called when no context
                    mock_banner.assert_not_called()

                    # Check result
                    assert result.success is True
                    assert result.should_clear is False  # We handle clearing internally

    @pytest.mark.asyncio
    async def test_execute_with_context(self, command):
        """Test executing the clear command with context available."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            with patch("chuk_term.ui.display_chat_banner") as mock_banner:
                # Mock get_context from the context module
                with patch("mcp_cli.context.get_context") as mock_context:
                    mock_ctx = type("Context", (), {})()
                    mock_ctx.model_manager = type("ModelManager", (), {})()
                    mock_ctx.model_manager.get_active_provider = lambda: "ollama"
                    mock_ctx.model_manager.get_active_model = lambda: "gpt-oss"
                    mock_ctx.tool_manager = type("ToolManager", (), {})()
                    mock_ctx.tool_manager.list_tools = lambda: [
                        "tool1",
                        "tool2",
                        "tool3",
                    ]
                    mock_context.return_value = mock_ctx

                    result = await command.execute(verbose=True)

                    # Verify clear_screen was called
                    mock_clear.assert_called_once()
                    # Verify banner was called with correct info
                    mock_banner.assert_called_once_with(
                        provider="ollama",
                        model="gpt-oss",
                        additional_info={"Tools": "3"},
                    )

                    # Check result
                    assert result.success is True
                    assert result.should_clear is False  # We handle clearing internally

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during clear."""
        with patch("chuk_term.ui.clear_screen") as mock_clear:
            mock_clear.side_effect = Exception("Clear failed")

            # The clear command doesn't have explicit error handling,
            # so an exception will propagate
            with pytest.raises(Exception, match="Clear failed"):
                await command.execute()
