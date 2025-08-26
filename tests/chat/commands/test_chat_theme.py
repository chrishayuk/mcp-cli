"""Tests for chat mode theme command."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.chat.commands.theme import handle_theme_command


class TestChatThemeCommand:
    """Test chat mode theme command."""

    @pytest.fixture
    def mock_context(self):
        """Create mock chat context."""
        context = MagicMock()
        return context

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_show_current_theme(
        self, mock_get_manager, mock_set_theme, mock_output, mock_ask, mock_context
    ):
        """Test showing current theme with no arguments triggers interactive selection."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "dark"
        mock_get_manager.return_value = mock_manager
        # Mock selection returns theme number for monokai
        mock_ask.return_value = "6"  # monokai is at index 6

        await handle_theme_command(mock_context, [])

        # Should trigger interactive selection
        mock_ask.assert_called_once()
        # Should set the selected theme
        mock_set_theme.assert_called_once_with("monokai")
        mock_manager.set_theme.assert_called_once_with("monokai")

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_set_valid_theme(
        self, mock_get_manager, mock_set_theme, mock_output, mock_context
    ):
        """Test setting a valid theme."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["monokai"])

        # Should set theme in UI and preferences
        mock_set_theme.assert_called_once_with("monokai")
        mock_manager.set_theme.assert_called_once_with("monokai")

        # Should display success message (could be in preview)
        assert mock_output.success.called
        success_calls = str(mock_output.success.call_args_list)
        assert "monokai" in success_calls or "Theme switched" in success_calls

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_set_invalid_theme(
        self, mock_get_manager, mock_set_theme, mock_output, mock_context
    ):
        """Test setting an invalid theme."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        await handle_theme_command(mock_context, ["invalid_theme"])

        # Should not set theme
        mock_set_theme.assert_not_called()
        mock_manager.set_theme.assert_not_called()

        # Should display error message
        mock_output.error.assert_called()
        error_args = str(mock_output.error.call_args)
        assert "Invalid theme" in error_args
        assert "invalid_theme" in error_args

    @pytest.mark.asyncio
    @patch("mcp_cli.chat.commands.theme.ask")
    @patch("mcp_cli.chat.commands.theme.output")
    @patch("mcp_cli.chat.commands.theme.set_theme")
    @patch("mcp_cli.chat.commands.theme.get_preference_manager")
    async def test_select_theme_interactive(
        self, mock_get_manager, mock_set_theme, mock_output, mock_ask, mock_context
    ):
        """Test interactive theme selection."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"
        mock_get_manager.return_value = mock_manager

        # Mock the selection to return theme number
        mock_ask.return_value = "6"  # monokai is at index 6

        await handle_theme_command(mock_context, ["select"])

        # Should prompt for selection
        mock_ask.assert_called_once()
        prompt = mock_ask.call_args[0][0]
        assert "Enter theme number (1-8) or name:" in prompt
        assert mock_ask.call_args.kwargs.get("default") == "1"  # default is at index 1

        # Should set selected theme (monokai)
        mock_set_theme.assert_called_once_with("monokai")
        mock_manager.set_theme.assert_called_once_with("monokai")

        # Should display success
        assert mock_output.success.called
