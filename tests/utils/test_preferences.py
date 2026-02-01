"""Tests for preference management system."""

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
            provider=ProviderPreferences(
                active_provider="ollama", active_model="gpt-oss"
            ),
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
            },
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
            PreferenceManager(config_dir=config_dir)  # Create initial manager

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


class TestPreferenceManagerExtended:
    """Test server enable/disable preferences."""

    def test_server_disabled_by_default(self):
        """Test that servers are enabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Servers should be enabled by default
            assert not manager.is_server_disabled("test_server")
            assert not manager.is_server_disabled("another_server")

    def test_disable_server(self):
        """Test disabling a server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Disable a server
            manager.disable_server("test_server")

            # Check it's disabled
            assert manager.is_server_disabled("test_server")
            assert not manager.is_server_disabled("another_server")

            # Check persistence
            manager2 = PreferenceManager(config_dir=config_dir)
            assert manager2.is_server_disabled("test_server")

    def test_enable_server(self):
        """Test enabling a previously disabled server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Disable then enable
            manager.disable_server("test_server")
            assert manager.is_server_disabled("test_server")

            manager.enable_server("test_server")
            assert not manager.is_server_disabled("test_server")

            # Check persistence
            manager2 = PreferenceManager(config_dir=config_dir)
            assert not manager2.is_server_disabled("test_server")

    def test_set_server_disabled(self):
        """Test set_server_disabled method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Test disabling
            manager.set_server_disabled("server1", True)
            assert manager.is_server_disabled("server1")

            # Test enabling
            manager.set_server_disabled("server1", False)
            assert not manager.is_server_disabled("server1")

    def test_get_disabled_servers(self):
        """Test getting all disabled servers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Initially empty
            assert manager.get_disabled_servers() == {}

            # Disable multiple servers
            manager.disable_server("server1")
            manager.disable_server("server2")

            disabled = manager.get_disabled_servers()
            assert "server1" in disabled
            assert "server2" in disabled
            assert disabled["server1"] is True
            assert disabled["server2"] is True

    def test_clear_disabled_servers(self):
        """Test clearing all disabled server preferences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Disable some servers
            manager.disable_server("server1")
            manager.disable_server("server2")
            assert len(manager.get_disabled_servers()) == 2

            # Clear all
            manager.clear_disabled_servers()
            assert manager.get_disabled_servers() == {}
            assert not manager.is_server_disabled("server1")
            assert not manager.is_server_disabled("server2")

    def test_server_preferences_persist(self):
        """Test that server preferences persist across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager1 = PreferenceManager(config_dir=config_dir)

            # Set various server states
            manager1.disable_server("disabled_server")
            manager1.enable_server("enabled_server")
            manager1.disable_server("another_disabled")

            # Create new manager instance
            manager2 = PreferenceManager(config_dir=config_dir)

            # Check states persist
            assert manager2.is_server_disabled("disabled_server")
            assert not manager2.is_server_disabled("enabled_server")
            assert manager2.is_server_disabled("another_disabled")

            # Check disabled list
            disabled = manager2.get_disabled_servers()
            assert "disabled_server" in disabled
            assert "another_disabled" in disabled
            assert "enabled_server" not in disabled

    def test_get_and_set_verbose(self):
        """Test getting and setting verbose mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Default is True
            assert manager.get_verbose() is True

            # Set to False
            manager.set_verbose(False)
            assert manager.get_verbose() is False

            # Verify persistence
            new_manager = PreferenceManager(config_dir=config_dir)
            assert new_manager.get_verbose() is False

    def test_tool_confirmation_mode(self):
        """Test tool confirmation mode settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Default mode
            assert manager.get_tool_confirmation_mode() == "smart"

            # Set to always
            manager.set_tool_confirmation_mode("always")
            assert manager.get_tool_confirmation_mode() == "always"

            # Set to never
            manager.set_tool_confirmation_mode("never")
            assert manager.get_tool_confirmation_mode() == "never"

            # Invalid mode
            with pytest.raises(ValueError, match="Invalid confirmation mode"):
                manager.set_tool_confirmation_mode("invalid")

    def test_per_tool_confirmation(self):
        """Test per-tool confirmation settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # No tool-specific setting initially
            assert manager.get_tool_confirmation("test_tool") is None

            # Set tool-specific confirmation
            manager.set_tool_confirmation("test_tool", "always")
            assert manager.get_tool_confirmation("test_tool") == "always"

            # Set another tool
            manager.set_tool_confirmation("danger_tool", "never")
            assert manager.get_tool_confirmation("danger_tool") == "never"

            # Get all confirmations
            all_confirmations = manager.get_all_tool_confirmations()
            assert all_confirmations["test_tool"] == "always"
            assert all_confirmations["danger_tool"] == "never"

            # Remove tool confirmation (set to None)
            manager.set_tool_confirmation("test_tool", None)
            assert manager.get_tool_confirmation("test_tool") is None

            # Invalid setting
            with pytest.raises(ValueError, match="Invalid tool confirmation setting"):
                manager.set_tool_confirmation("tool", "invalid")

    def test_clear_tool_confirmations(self):
        """Test clearing all tool confirmations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Add some confirmations
            manager.set_tool_confirmation("tool1", "always")
            manager.set_tool_confirmation("tool2", "never")
            manager.set_tool_confirmation("tool3", "ask")

            # Clear all
            manager.clear_tool_confirmations()

            # Verify all cleared
            assert manager.get_all_tool_confirmations() == {}
            assert manager.get_tool_confirmation("tool1") is None
            assert manager.get_tool_confirmation("tool2") is None
            assert manager.get_tool_confirmation("tool3") is None

    def test_tool_risk_level(self):
        """Test tool risk level determination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Set up categories with patterns
            manager.preferences.ui.tool_confirmation.categories = {
                "read_*": "safe",
                "*_write": "high",
                "list_*": "safe",
                "*_delete": "high",
            }

            # Test pattern matching
            assert manager.get_tool_risk_level("read_file") == "safe"
            assert manager.get_tool_risk_level("file_write") == "high"
            assert manager.get_tool_risk_level("list_items") == "safe"
            assert manager.get_tool_risk_level("item_delete") == "high"

            # Default for unmatched
            assert manager.get_tool_risk_level("unknown_tool") == "moderate"

    def test_should_confirm_tool(self):
        """Test tool confirmation logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Test with per-tool settings
            manager.set_tool_confirmation("always_tool", "always")
            manager.set_tool_confirmation("never_tool", "never")
            manager.set_tool_confirmation("ask_tool", "ask")

            assert manager.should_confirm_tool("always_tool") is True
            assert manager.should_confirm_tool("never_tool") is False
            assert manager.should_confirm_tool("ask_tool") is True

            # Test with global mode = always
            manager.set_tool_confirmation_mode("always")
            assert manager.should_confirm_tool("any_tool") is True

            # Test with global mode = never (but per-tool overrides)
            manager.set_tool_confirmation_mode("never")
            assert (
                manager.should_confirm_tool("always_tool") is True
            )  # per-tool override
            assert manager.should_confirm_tool("any_tool") is False

            # Test smart mode with risk levels
            manager.set_tool_confirmation_mode("smart")
            manager.preferences.ui.tool_confirmation.categories = {
                "safe_*": "safe",
                "*_danger": "high",
            }
            manager.preferences.ui.tool_confirmation.risk_thresholds = {
                "safe": False,
                "moderate": True,
                "high": True,
            }

            assert manager.should_confirm_tool("safe_tool") is False
            assert manager.should_confirm_tool("tool_danger") is True
            assert (
                manager.should_confirm_tool("normal_tool") is True
            )  # moderate default

    def test_tool_patterns(self):
        """Test tool pattern management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Add patterns
            manager.add_tool_pattern("write_*", "always")
            manager.add_tool_pattern("read_*", "never")

            # Verify patterns added
            patterns = manager.preferences.ui.tool_confirmation.patterns
            assert len(patterns) == 2
            # Patterns are now ToolPatternRule Pydantic models
            assert any(
                p.pattern == "write_*" and p.action == "always" for p in patterns
            )
            assert any(p.pattern == "read_*" and p.action == "never" for p in patterns)

            # Remove pattern
            assert manager.remove_tool_pattern("write_*") is True
            patterns = manager.preferences.ui.tool_confirmation.patterns
            assert len(patterns) == 1
            assert any(p.pattern == "read_*" and p.action == "never" for p in patterns)

            # Remove non-existent pattern
            assert manager.remove_tool_pattern("nonexistent_*") is False

    def test_risk_thresholds(self):
        """Test risk threshold settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Set risk thresholds
            manager.set_risk_threshold("safe", False)
            manager.set_risk_threshold("moderate", True)
            manager.set_risk_threshold("high", True)

            # Verify thresholds
            thresholds = manager.preferences.ui.tool_confirmation.risk_thresholds
            assert thresholds["safe"] is False
            assert thresholds["moderate"] is True
            assert thresholds["high"] is True

            # Invalid risk level
            with pytest.raises(ValueError, match="Invalid risk level"):
                manager.set_risk_threshold("invalid", True)

    def test_last_servers(self):
        """Test last servers management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # No last servers initially
            assert manager.get_last_servers() is None

            # Set last servers
            manager.set_last_servers("server1,server2")
            assert manager.get_last_servers() == "server1,server2"

            # Verify persistence
            new_manager = PreferenceManager(config_dir=config_dir)
            assert new_manager.get_last_servers() == "server1,server2"

    def test_config_file(self):
        """Test config file path management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # No config file initially
            assert manager.get_config_file() is None

            # Set config file
            manager.set_config_file("/path/to/config.json")
            assert manager.get_config_file() == "/path/to/config.json"

            # Verify persistence
            new_manager = PreferenceManager(config_dir=config_dir)
            assert new_manager.get_config_file() == "/path/to/config.json"

    def test_reset_preferences(self):
        """Test resetting preferences to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Set various preferences
            manager.set_theme("dark")
            manager.set_active_provider("openai")
            manager.set_active_model("gpt-4")
            manager.set_verbose(False)
            manager.set_last_servers("test")

            # Reset all preferences
            manager.reset_preferences()

            # Verify all reset to defaults
            assert manager.get_theme() == "default"
            assert manager.get_active_provider() is None
            assert manager.get_active_model() is None
            assert manager.get_verbose() is True
            assert manager.get_last_servers() is None

    def test_history_and_logs(self):
        """Test history file and logs directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Get history file
            history_file = manager.get_history_file()
            assert history_file == config_dir / "chat_history"

            # Get logs directory (should create if not exists)
            logs_dir = manager.get_logs_dir()
            assert logs_dir == config_dir / "logs"
            assert logs_dir.exists()

    def test_clear_all_disabled_servers(self):
        """Test clearing all disabled servers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Disable some servers
            manager.disable_server("server1")
            manager.disable_server("server2")
            manager.disable_server("server3")

            # Verify disabled
            disabled = manager.get_disabled_servers()
            assert len(disabled) == 3

            # Clear all disabled
            manager.clear_disabled_servers()

            # Verify cleared
            disabled = manager.get_disabled_servers()
            assert len(disabled) == 0

    def test_runtime_server_check(self):
        """Test checking if server is runtime server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Add runtime server
            manager.add_runtime_server("runtime1", {"transport": "stdio"})

            # Check runtime server
            assert manager.is_runtime_server("runtime1") is True
            assert manager.is_runtime_server("not_runtime") is False

    def test_custom_provider_management(self):
        """Test custom provider management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Add custom provider with defaults
            manager.add_custom_provider(
                name="custom_ai",
                api_base="https://api.custom.ai",
                default_model="custom-gpt-4",
            )

            # Verify provider added
            providers = manager.get_custom_providers()
            assert "custom_ai" in providers
            provider = providers["custom_ai"]
            assert provider["api_base"] == "https://api.custom.ai"
            assert provider["default_model"] == "custom-gpt-4"
            assert "gpt-4" in provider["models"]  # Default models

            # Add provider with custom models and env var
            manager.add_custom_provider(
                name="another_ai",
                api_base="https://api.another.ai",
                default_model="model-1",
                models=["model-1", "model-2", "model-3"],
                env_var_name="ANOTHER_AI_KEY",
            )

            # Verify second provider
            providers = manager.get_custom_providers()
            assert "another_ai" in providers
            provider = providers["another_ai"]
            assert provider["models"] == ["model-1", "model-2", "model-3"]
            assert provider["env_var_name"] == "ANOTHER_AI_KEY"

            # Get specific provider
            provider = manager.get_custom_provider("custom_ai")
            assert provider is not None
            assert provider["name"] == "custom_ai"

            # Get non-existent provider
            assert manager.get_custom_provider("nonexistent") is None

            # Update existing provider
            manager.update_custom_provider(
                name="custom_ai", default_model="custom-gpt-3.5"
            )
            provider = manager.get_custom_provider("custom_ai")
            assert provider["default_model"] == "custom-gpt-3.5"

            # Remove provider
            assert manager.remove_custom_provider("custom_ai") is True
            assert manager.get_custom_provider("custom_ai") is None

            # Remove non-existent provider
            assert manager.remove_custom_provider("nonexistent") is False

            # Test multiple providers exist
            manager.add_custom_provider("test1", "https://test1.ai", "model1")
            manager.add_custom_provider("test2", "https://test2.ai", "model2")
            providers = manager.get_custom_providers()
            # Should have test1, test2, and another_ai
            assert len(providers) >= 2

            # Remove all added providers for cleanup
            manager.remove_custom_provider("test1")
            manager.remove_custom_provider("test2")
            manager.remove_custom_provider("another_ai")

    def test_additional_coverage(self):
        """Test additional methods for complete coverage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Test get_runtime_server (covers line 562)
            manager.add_runtime_server("test_srv", {"command": "test"})
            server = manager.get_runtime_server("test_srv")
            assert server is not None
            assert server["command"] == "test"

            # Non-existent server
            assert manager.get_runtime_server("nonexistent") is None

            # Test is_custom_provider (covers line 627)
            manager.add_custom_provider("custom", "http://api.test", "model1")
            assert manager.is_custom_provider("custom") is True
            assert manager.is_custom_provider("nonexistent") is False

            # Test update_custom_provider for non-existent provider (covers line 650)
            result = manager.update_custom_provider(
                "nonexistent", api_base="http://new.api"
            )
            assert result is False

            # Test update with only api_base (covers line 655)
            result = manager.update_custom_provider(
                "custom", api_base="http://updated.api"
            )
            assert result is True
            provider = manager.get_custom_provider("custom")
            assert provider["api_base"] == "http://updated.api"

            # Test update with only models (covers line 659)
            result = manager.update_custom_provider(
                "custom", models=["new-model-1", "new-model-2"]
            )
            assert result is True
            provider = manager.get_custom_provider("custom")
            assert provider["models"] == ["new-model-1", "new-model-2"]

            # Test get_custom_provider_api_key (covers lines 675-687)
            import os

            # Test with provider that doesn't exist
            assert manager.get_custom_provider_api_key("nonexistent") is None

            # Test with default env var name pattern
            os.environ["CUSTOM_API_KEY"] = "test-key-default"
            api_key = manager.get_custom_provider_api_key("custom")
            assert api_key == "test-key-default"

            # Test with custom env_var_name
            manager.update_custom_provider("custom", env_var_name="MY_CUSTOM_KEY")
            os.environ["MY_CUSTOM_KEY"] = "test-key-custom"
            api_key = manager.get_custom_provider_api_key("custom")
            assert api_key == "test-key-custom"

            # Clean up env vars
            del os.environ["CUSTOM_API_KEY"]
            del os.environ["MY_CUSTOM_KEY"]

    def test_custom_provider_from_dict(self):
        """Test CustomProvider.from_dict method (covers lines 120-133)."""
        from mcp_cli.utils.preferences import CustomProvider

        # Test with minimal data
        data = {
            "name": "test_provider",
            "api_base": "http://test.api",
        }
        provider = CustomProvider.from_dict(data)
        assert provider.name == "test_provider"
        assert provider.api_base == "http://test.api"
        assert provider.default_model == "gpt-4"  # Default
        assert provider.models == ["gpt-4", "gpt-3.5-turbo"]  # Default
        assert provider.env_var_name is None

        # Test get_env_var_name with no env_var_name set (covers lines 130-133)
        env_var = provider.get_env_var_name()
        assert env_var == "TEST_PROVIDER_API_KEY"

        # Test with full data
        data_full = {
            "name": "another-provider",
            "api_base": "http://another.api",
            "default_model": "custom-model",
            "models": ["model1", "model2"],
            "env_var_name": "CUSTOM_ENV_VAR",
        }
        provider_full = CustomProvider.from_dict(data_full)
        assert provider_full.default_model == "custom-model"
        assert provider_full.models == ["model1", "model2"]
        assert provider_full.env_var_name == "CUSTOM_ENV_VAR"

        # Test get_env_var_name with custom env_var_name
        env_var = provider_full.get_env_var_name()
        assert env_var == "CUSTOM_ENV_VAR"

    def test_update_custom_provider_env_var_coverage(self):
        """Test updating custom provider with env_var_name to cover line 661."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Add a provider
            manager.add_custom_provider(
                name="test_provider",
                api_base="https://api.test.ai",
                default_model="test-model",
            )

            # Update with env_var_name (covers line 661)
            result = manager.update_custom_provider(
                name="test_provider",
                env_var_name="TEST_PROVIDER_KEY",
            )
            assert result is True

            provider = manager.get_custom_provider("test_provider")
            assert provider["env_var_name"] == "TEST_PROVIDER_KEY"

    def test_get_runtime_servers_copy(self):
        """Test that get_runtime_servers returns a copy to cover line 558."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Add runtime server
            manager.add_runtime_server(
                "test_server", {"command": "test", "args": ["arg1"]}
            )

            # Get runtime servers (covers line 558)
            servers = manager.get_runtime_servers()
            assert "test_server" in servers

            # Verify it's a shallow copy - can add/remove servers without affecting original
            servers["new_server"] = {"command": "new"}

            # Original should not have the new server
            original_servers = manager.get_runtime_servers()
            assert "new_server" not in original_servers

            # But the method does return shallow copy, so nested changes will affect original
            # This is the actual behavior of the code

    def test_should_confirm_tool_default_true(self):
        """Test that should_confirm_tool returns True by default to cover line 391."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Directly set an invalid mode in preferences to bypass validation
            # This simulates corrupted/manually edited preferences
            manager.preferences.ui.tool_confirmation.mode = "invalid_mode"

            # This should hit the default return True at line 391
            # because the mode is not recognized
            result = manager.should_confirm_tool("some_tool")
            assert result is True

    def test_load_preferences_with_errors(self):
        """Test loading preferences with file errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            prefs_file = config_dir / "preferences.json"

            # Create directory and invalid JSON file
            config_dir.mkdir(parents=True, exist_ok=True)
            prefs_file.write_text("invalid json {]")

            # Should handle invalid JSON gracefully
            manager = PreferenceManager(config_dir=config_dir)
            assert manager.preferences.ui.theme == "default"  # Defaults loaded

            # Create valid JSON with invalid theme
            prefs_file.write_text('{"ui": {"theme": "invalid_theme"}}')
            manager = PreferenceManager(config_dir=config_dir)
            # Should handle invalid theme value
            assert manager.preferences.ui.theme in ["default", "invalid_theme"]

    def test_legacy_confirm_tools_compatibility(self):
        """Test legacy confirm_tools compatibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Test get_confirm_tools with different modes
            manager.set_tool_confirmation_mode("always")
            assert manager.get_confirm_tools() is True

            manager.set_tool_confirmation_mode("never")
            assert manager.get_confirm_tools() is False

            manager.set_tool_confirmation_mode("smart")
            assert manager.get_confirm_tools() is True

            # Test set_confirm_tools updates both old and new settings
            manager.set_confirm_tools(False)
            assert manager.get_tool_confirmation_mode() == "never"
            assert manager.preferences.ui.confirm_tools is False

            manager.set_confirm_tools(True)
            assert manager.get_tool_confirmation_mode() == "smart"
            assert manager.preferences.ui.confirm_tools is True

    def test_remove_runtime_server_not_found(self):
        """Test removing non-existent runtime server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".mcp-cli"
            manager = PreferenceManager(config_dir=config_dir)

            # Remove non-existent server returns False
            assert manager.remove_runtime_server("nonexistent") is False

            # Add and then remove server
            manager.add_runtime_server("test", {"transport": "stdio"})
            assert manager.remove_runtime_server("test") is True
            assert manager.remove_runtime_server("test") is False  # Already removed


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
