"""Extended tests for the theme command definition."""

import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.definitions.theme import ThemeCommand
from mcp_cli.commands.base import CommandMode


@pytest.fixture
def theme_command():
    """Create a theme command instance."""
    return ThemeCommand()


@pytest.fixture
def mock_pref_manager():
    """Create a mock preference manager."""
    manager = MagicMock()
    manager.get_theme.return_value = "default"
    manager.set_theme.return_value = None
    return manager


def test_theme_command_properties(theme_command):
    """Test theme command properties."""
    assert theme_command.name == "theme"
    assert theme_command.aliases == ["themes"]
    assert theme_command.description == "Change the UI theme"
    assert theme_command.modes == CommandMode.CHAT | CommandMode.INTERACTIVE
    assert len(theme_command.parameters) == 1
    assert theme_command.parameters[0].name == "theme_name"
    assert not theme_command.parameters[0].required
    assert "Available themes" in theme_command.help_text


@pytest.mark.asyncio
async def test_theme_set_valid_theme(theme_command, mock_pref_manager):
    """Test setting a valid theme."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.definitions.theme.set_theme") as mock_set_theme:
            result = await theme_command.execute(theme_name="dark")

            assert result.success is True
            assert result.output == "Theme changed to: dark"
            mock_set_theme.assert_called_once_with("dark")
            mock_pref_manager.set_theme.assert_called_once_with("dark")


@pytest.mark.asyncio
async def test_theme_set_invalid_theme(theme_command, mock_pref_manager):
    """Test setting an invalid theme."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        result = await theme_command.execute(theme_name="invalid_theme")

        assert result.success is False
        assert "Invalid theme: invalid_theme" in result.error
        assert "Available themes:" in result.error


@pytest.mark.asyncio
async def test_theme_from_args_list(theme_command, mock_pref_manager):
    """Test setting theme from args list."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.definitions.theme.set_theme") as mock_set_theme:
            result = await theme_command.execute(args=["monokai"])

            assert result.success is True
            assert result.output == "Theme changed to: monokai"
            mock_set_theme.assert_called_once_with("monokai")


@pytest.mark.asyncio
async def test_theme_from_args_string(theme_command, mock_pref_manager):
    """Test setting theme from args string."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.definitions.theme.set_theme") as mock_set_theme:
            result = await theme_command.execute(args="dracula")

            assert result.success is True
            assert result.output == "Theme changed to: dracula"
            mock_set_theme.assert_called_once_with("dracula")


@pytest.mark.asyncio
async def test_theme_interactive_selection(theme_command, mock_pref_manager):
    """Test interactive theme selection."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch(
            "mcp_cli.commands.actions.theme._interactive_theme_selection"
        ) as mock_interactive:
            mock_interactive.return_value = None

            result = await theme_command.execute()

            assert result.success is True
            mock_interactive.assert_called_once_with(mock_pref_manager)


@pytest.mark.asyncio
async def test_theme_interactive_selection_error(theme_command, mock_pref_manager):
    """Test fallback when interactive selection fails."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch(
            "mcp_cli.commands.actions.theme._interactive_theme_selection"
        ) as mock_interactive:
            mock_interactive.side_effect = Exception("Interactive failed")

            result = await theme_command.execute()

            assert result.success is True
            assert "Current theme: default" in result.output
            assert "Available themes:" in result.output


@pytest.mark.asyncio
async def test_theme_all_valid_themes(theme_command, mock_pref_manager):
    """Test setting all valid themes."""
    valid_themes = [
        "default",
        "dark",
        "light",
        "minimal",
        "terminal",
        "monokai",
        "dracula",
        "solarized",
    ]

    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch("mcp_cli.commands.definitions.theme.set_theme") as mock_set_theme:
            for theme in valid_themes:
                result = await theme_command.execute(theme_name=theme)

                assert result.success is True
                assert result.output == f"Theme changed to: {theme}"
                mock_set_theme.assert_called_with(theme)


@pytest.mark.asyncio
async def test_theme_empty_args_list(theme_command, mock_pref_manager):
    """Test with empty args list."""
    with patch(
        "mcp_cli.utils.preferences.get_preference_manager",
        return_value=mock_pref_manager,
    ):
        with patch(
            "mcp_cli.commands.actions.theme._interactive_theme_selection"
        ) as mock_interactive:
            mock_interactive.return_value = None

            result = await theme_command.execute(args=[])

            assert result.success is True
            mock_interactive.assert_called_once()


@pytest.mark.asyncio
async def test_theme_parameter_choices(theme_command):
    """Test that parameter choices match available themes."""
    param = theme_command.parameters[0]
    assert param.choices == [
        "default",
        "dark",
        "light",
        "minimal",
        "terminal",
        "monokai",
        "dracula",
        "solarized",
    ]
