"""Tests for the themes plural command."""

import pytest
from unittest.mock import patch, Mock
from mcp_cli.commands.theme.themes_plural import ThemesPluralCommand


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
        assert command.description == "List all available themes"
        assert "List all available UI themes" in command.help_text

    @pytest.mark.asyncio
    async def test_execute_success(self, command):
        """Test successful execution."""
        with patch("mcp_cli.utils.preferences.get_preference_manager") as mock_pref:
            mock_pref_mgr = Mock()
            mock_pref_mgr.get_theme.return_value = "dark"
            mock_pref.return_value = mock_pref_mgr
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute()

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self, command):
        """Test execution with kwargs (should be ignored)."""
        with patch("mcp_cli.utils.preferences.get_preference_manager") as mock_pref:
            mock_pref_mgr = Mock()
            mock_pref_mgr.get_theme.return_value = "dark"
            mock_pref.return_value = mock_pref_mgr
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute(some_arg="value")

                    # Should still call successfully
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        with patch("mcp_cli.utils.preferences.get_preference_manager") as mock_pref:
            mock_pref.side_effect = Exception("Theme list failed")
            with patch("chuk_term.ui.output"):
                result = await command.execute()

                assert result.success is False
                assert "Failed to list themes" in result.error
                assert "Theme list failed" in result.error

    def test_help_text_content(self, command):
        """Test that help text contains expected information."""
        help_text = command.help_text

        # Check for key usage patterns
        assert "/themes" in help_text
        assert "Usage:" in help_text
        assert "Examples:" in help_text
