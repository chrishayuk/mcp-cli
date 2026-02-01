"""Tests for the theme commands."""

import pytest
from unittest.mock import patch, Mock
from mcp_cli.commands.theme.theme_singular import ThemeSingularCommand
from mcp_cli.commands.theme.themes_plural import ThemesPluralCommand


class TestThemeSingularCommand:
    """Test the ThemeSingularCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ThemeSingularCommand instance."""
        return ThemeSingularCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "theme"
        assert command.aliases == []
        assert "current theme" in command.description.lower()
        assert command.requires_context is True  # Uses default

    @pytest.mark.asyncio
    async def test_execute_show_current(self, command):
        """Test showing current theme."""
        # Just test that the command executes without error
        # The actual theme display is handled by chuk_term which is tested separately
        with patch("chuk_term.ui.theme.get_theme") as mock_get_theme:
            with patch("mcp_cli.utils.preferences.get_preference_manager") as mock_pref:
                mock_theme = Mock()
                mock_theme.description = "Dark theme"
                mock_get_theme.return_value = mock_theme
                mock_pref_mgr = Mock()
                mock_pref_mgr.get_theme.return_value = "dark"
                mock_pref.return_value = mock_pref_mgr
                with patch("chuk_term.ui.output"):
                    result = await command.execute()

                    # We just care that it executes successfully
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_theme(self, command):
        """Test setting a theme."""
        with patch("chuk_term.ui.theme.set_theme") as mock_set:
            with patch("mcp_cli.utils.preferences.get_preference_manager") as mock_pref:
                mock_pref_mgr = Mock()
                mock_pref.return_value = mock_pref_mgr
                with patch("chuk_term.ui.output"):
                    result = await command.execute(args=["dark"])

                    mock_set.assert_called_once_with("dark")
                    mock_pref_mgr.set_theme.assert_called_once_with("dark")
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_invalid_theme(self, command):
        """Test setting an invalid theme."""
        with patch("chuk_term.ui.output"):
            result = await command.execute(args=["invalid"])

            # The command will catch the exception and return an error
            assert result.success is False


class TestThemesPluralCommand:
    """Test the ThemesPluralCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ThemesPluralCommand instance."""
        return ThemesPluralCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "themes"
        assert command.aliases == []
        assert "List all available themes" in command.description
        assert command.requires_context is True  # Uses default

    @pytest.mark.asyncio
    async def test_execute_list_themes(self, command):
        """Test listing available themes."""
        with patch("mcp_cli.utils.preferences.get_preference_manager") as mock_pref:
            mock_pref_mgr = Mock()
            mock_pref_mgr.get_theme.return_value = "dark"
            mock_pref.return_value = mock_pref_mgr
            with patch("chuk_term.ui.output"):
                result = await command.execute()

                assert result.success is True
