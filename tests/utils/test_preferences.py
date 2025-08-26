"""Tests for preference management system."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.utils.preferences import (
    MCPPreferences,
    PreferenceManager,
    Theme,
    get_preference_manager,
)


class TestThemeEnum:
    """Test Theme enum functionality."""

    def test_theme_values(self):
        """Test that all expected theme values exist."""
        expected_themes = [
            "default",
            "dark",
            "light",
            "minimal",
            "terminal",
            "monokai",
            "dracula",
            "solarized",
        ]
        actual_themes = [theme.value for theme in Theme]
        assert set(expected_themes) == set(actual_themes)

    def test_theme_from_string(self):
        """Test creating theme from string value."""
        assert Theme("dark") == Theme.DARK
        assert Theme("monokai") == Theme.MONOKAI

    def test_invalid_theme(self):
        """Test that invalid theme raises ValueError."""
        with pytest.raises(ValueError):
            Theme("invalid_theme")


class TestMCPPreferences:
    """Test MCPPreferences dataclass."""

    def test_default_preferences(self):
        """Test default preference values."""
        prefs = MCPPreferences()
        assert prefs.ui.theme == Theme.DEFAULT.value
        assert prefs.provider.active_provider is None
        assert prefs.provider.active_model is None
        assert prefs.ui.confirm_tools is True

    def test_to_dict(self):
        """Test converting preferences to dictionary."""
        from mcp_cli.utils.preferences import UIPreferences, ProviderPreferences
        prefs = MCPPreferences(
            ui=UIPreferences(theme="dark"),
            provider=ProviderPreferences(active_provider="ollama", active_model="gpt-oss")
        )
        result = prefs.to_dict()
        assert result["ui"]["theme"] == "dark"
        assert result["provider"]["active_provider"] == "ollama"
        assert result["provider"]["active_model"] == "gpt-oss"
        assert result["ui"]["confirm_tools"] is True

    def test_from_dict(self):
        """Test creating preferences from dictionary."""
        data = {
            "ui": {
                "theme": "monokai",
                "confirm_tools": False,
            },
            "provider": {
                "active_provider": "openai",
                "active_model": "gpt-4",
            }
        }
        prefs = MCPPreferences.from_dict(data)
        assert prefs.ui.theme == "monokai"
        assert prefs.provider.active_provider == "openai"
        assert prefs.provider.active_model == "gpt-4"
        assert prefs.ui.confirm_tools is False

    def test_from_dict_partial(self):
        """Test creating preferences from partial dictionary."""
        data = {"ui": {"theme": "dracula"}}
        prefs = MCPPreferences.from_dict(data)
        assert prefs.ui.theme == "dracula"
        assert prefs.provider.active_provider is None
        assert prefs.provider.active_model is None
        assert prefs.ui.confirm_tools is True


class TestPreferenceManager:
    """Test PreferenceManager functionality."""

    def test_initialization(self):
        """Test preference manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            assert manager.config_dir == config_dir
            assert manager.config_dir.exists()
            assert manager.preferences_file == config_dir / "preferences.json"

    def test_load_nonexistent_preferences(self):
        """Test loading preferences when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            # Preferences are loaded in __init__
            assert manager.preferences.ui.theme == Theme.DEFAULT.value
            assert manager.preferences.provider.active_provider is None

    def test_save_and_load_preferences(self):
        """Test saving and loading preferences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from mcp_cli.utils.preferences import UIPreferences, ProviderPreferences
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Modify and save preferences
            manager.preferences.ui.theme = "dark"
            manager.preferences.ui.confirm_tools = False
            manager.preferences.provider.active_provider = "anthropic"
            manager.preferences.provider.active_model = "claude-3"
            manager.save_preferences()

            # Create new manager to load from file
            new_manager = PreferenceManager(config_dir=config_dir)
            assert new_manager.preferences.ui.theme == "dark"
            assert new_manager.preferences.provider.active_provider == "anthropic"
            assert new_manager.preferences.provider.active_model == "claude-3"
            assert new_manager.preferences.ui.confirm_tools is False

    def test_get_and_set_theme(self):
        """Test getting and setting theme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Default theme
            assert manager.get_theme() == "default"

            # Set theme
            manager.set_theme("monokai")
            assert manager.get_theme() == "monokai"

            # Verify persistence
            new_manager = PreferenceManager(config_dir=config_dir)
            assert new_manager.get_theme() == "monokai"

    def test_set_invalid_theme(self):
        """Test setting invalid theme raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            with pytest.raises(ValueError, match="Invalid theme"):
                manager.set_theme("invalid_theme")

    def test_get_and_set_provider(self):
        """Test getting and setting default provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # No default provider initially
            assert manager.get_active_provider() is None

            # Set provider
            manager.set_active_provider("openai")
            assert manager.get_active_provider() == "openai"

    def test_get_and_set_model(self):
        """Test getting and setting default model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # No default model initially
            assert manager.get_active_model() is None

            # Set model
            manager.set_active_model("gpt-4-turbo")
            assert manager.get_active_model() == "gpt-4-turbo"

    def test_get_and_set_tool_confirmation(self):
        """Test getting and setting tool confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Default is True
            assert manager.get_confirm_tools() is True

            # Set to False
            manager.set_confirm_tools(False)
            assert manager.get_confirm_tools() is False

    def test_get_history_file(self):
        """Test getting history file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)
            history_file = manager.get_history_file()
            assert history_file == config_dir / "chat_history"

    def test_corrupted_preferences_file(self):
        """Test handling corrupted preferences file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Write corrupted JSON
            config_dir.mkdir(parents=True, exist_ok=True)
            prefs_file = config_dir / "preferences.json"
            prefs_file.write_text("{corrupted json")

            # Should return default preferences (backup original)
            prefs = manager.load_preferences()
            assert prefs.ui.theme == Theme.DEFAULT.value
            # Check that backup was created
            backup_file = config_dir / "preferences.json.backup"
            assert backup_file.exists()

    def test_partial_preferences_file(self):
        """Test loading partial preferences file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Write partial preferences (with correct nested structure)
            config_dir.mkdir(parents=True, exist_ok=True)
            prefs_file = config_dir / "preferences.json"
            prefs_file.write_text('{"ui": {"theme": "dracula"}}')

            # Should merge with defaults
            # Create new manager to load from file
            new_manager = PreferenceManager(config_dir=config_dir)
            assert new_manager.preferences.ui.theme == "dracula"
            assert new_manager.preferences.provider.active_provider is None
            assert new_manager.preferences.ui.confirm_tools is True


class TestSingletonManager:
    """Test singleton behavior of preference manager."""

    @patch("mcp_cli.utils.preferences.PreferenceManager")
    def test_get_preference_manager_singleton(self, mock_manager_class):
        """Test that get_preference_manager returns singleton."""
        mock_instance = MagicMock()
        mock_manager_class.return_value = mock_instance

        # Clear any existing singleton
        import mcp_cli.utils.preferences

        mcp_cli.utils.preferences._preference_manager = None

        # First call creates instance
        manager1 = get_preference_manager()
        assert manager1 == mock_instance
        mock_manager_class.assert_called_once()

        # Second call returns same instance
        manager2 = get_preference_manager()
        assert manager2 == mock_instance
        assert mock_manager_class.call_count == 1  # Still only called once

    def test_preference_manager_real_singleton(self):
        """Test real singleton behavior."""
        # Clear any existing singleton
        import mcp_cli.utils.preferences

        mcp_cli.utils.preferences._preference_manager = None

        manager1 = get_preference_manager()
        manager2 = get_preference_manager()
        assert manager1 is manager2