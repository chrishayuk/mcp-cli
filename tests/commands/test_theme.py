"""Tests for theme command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.commands.theme import theme_command


class TestThemeCommand:
    """Test theme command functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / ".mcp-cli"

    @patch("mcp_cli.commands.theme.get_preference_manager")
    @patch("mcp_cli.commands.theme.set_theme")
    @patch("mcp_cli.commands.theme.output")
    def test_list_themes(self, mock_output, mock_set_theme, mock_get_manager):
        """Test listing available themes."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "dark"
        mock_get_manager.return_value = mock_manager

        theme_command(list_themes=True)

        # Should print header and all themes
        mock_output.rule.assert_called_with("Available Themes")
        calls = mock_output.print.call_args_list
        assert len(calls) > 0
        # Check that themes are listed
        all_output = str(calls)
        assert "dark" in all_output
        assert "monokai" in all_output
        assert "dracula" in all_output

    @patch("mcp_cli.commands.theme.get_preference_manager")
    @patch("mcp_cli.commands.theme.set_theme")
    @patch("mcp_cli.commands.theme.output")
    def test_show_current_theme(self, mock_output, mock_set_theme, mock_get_manager):
        """Test showing current theme when no args provided."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "monokai"
        mock_get_manager.return_value = mock_manager

        theme_command()

        # Should show current theme (when no args, it lists themes with current marked)
        mock_output.rule.assert_called_with("Available Themes")
        # Check both print and info calls since we use info for current theme
        print_calls = mock_output.print.call_args_list
        info_calls = mock_output.info.call_args_list
        all_output = str(print_calls) + str(info_calls)
        assert "monokai" in all_output
        assert "(current)" in all_output

    @patch("mcp_cli.commands.theme._show_theme_preview")
    @patch("mcp_cli.commands.theme.get_preference_manager")
    @patch("mcp_cli.commands.theme.set_theme")
    @patch("mcp_cli.commands.theme.output")
    def test_set_theme_valid(
        self, mock_output, mock_set_theme, mock_get_manager, mock_preview
    ):
        """Test setting a valid theme."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        theme_command(theme_name="dracula")

        # Should set theme in both UI and preferences
        mock_set_theme.assert_called_once_with("dracula")
        mock_manager.set_theme.assert_called_once_with("dracula")

        # Should confirm change
        mock_output.success.assert_called_with("Theme switched to: dracula")

    @patch("mcp_cli.commands.theme.get_preference_manager")
    @patch("mcp_cli.commands.theme.set_theme")
    @patch("mcp_cli.commands.theme.output")
    def test_set_theme_invalid(self, mock_output, mock_set_theme, mock_get_manager):
        """Test setting an invalid theme."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        theme_command(theme_name="invalid_theme")

        # Should not set theme in UI
        mock_set_theme.assert_not_called()
        mock_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
        error_call = str(mock_output.error.call_args)
        assert "Invalid theme" in error_call
        assert "invalid_theme" in error_call

    @patch("mcp_cli.commands.theme._show_theme_preview")
    @patch("mcp_cli.commands.theme.get_preference_manager")
    @patch("mcp_cli.commands.theme.set_theme")
    @patch("mcp_cli.commands.theme.ask")
    @patch("mcp_cli.commands.theme.output")
    def test_select_theme_interactive(
        self, mock_output, mock_ask, mock_set_theme, mock_get_manager, mock_preview
    ):
        """Test interactive theme selection."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"  # Must return a string
        mock_get_manager.return_value = mock_manager
        mock_ask.return_value = "8"  # Select solarized (index 8 in 1-based list)

        theme_command(select=True)

        # Should show selection prompt
        mock_ask.assert_called_once()
        args, kwargs = mock_ask.call_args
        assert "Enter theme number (1-8) or name:" in args[0]
        assert kwargs.get("default") == "1"  # default theme is at index 1

        # Should set selected theme
        mock_set_theme.assert_called_once_with("solarized")
        mock_manager.set_theme.assert_called_once_with("solarized")

        # Should confirm change - check success was called
        assert mock_output.success.called

    @patch("mcp_cli.commands.theme.get_preference_manager")
    @patch("mcp_cli.commands.theme.set_theme")
    @patch("mcp_cli.commands.theme.ask")
    @patch("mcp_cli.commands.theme.output")
    def test_select_theme_cancelled(
        self, mock_output, mock_ask, mock_set_theme, mock_get_manager
    ):
        """Test invalid number in interactive theme selection."""
        mock_manager = MagicMock()
        mock_manager.get_theme.return_value = "default"  # Must return a string
        mock_get_manager.return_value = mock_manager
        mock_ask.return_value = "99"  # Invalid number

        theme_command(select=True)

        # Should not set any theme
        mock_set_theme.assert_not_called()
        mock_manager.set_theme.assert_not_called()

        # Should show error
        mock_output.error.assert_called()
