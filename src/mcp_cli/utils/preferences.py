"""
Centralized preference management for MCP CLI.

This module handles all user preferences including themes, provider settings,
model preferences, and other configuration options in a centralized way.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class Theme(str, Enum):
    """Available UI themes from chuk-term."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    TERMINAL = "terminal"
    MONOKAI = "monokai"
    DRACULA = "dracula"
    SOLARIZED = "solarized"


@dataclass
class UIPreferences:
    """UI-related preferences."""

    theme: str = "default"
    verbose: bool = True
    confirm_tools: bool = True
    show_reasoning: bool = True


@dataclass
class ProviderPreferences:
    """Provider and model preferences."""

    active_provider: Optional[str] = None
    active_model: Optional[str] = None
    provider_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerPreferences:
    """Server-related preferences."""

    disabled_servers: Dict[str, bool] = field(default_factory=dict)
    server_settings: Dict[str, Any] = field(default_factory=dict)
    runtime_servers: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # User-added servers


@dataclass
class MCPPreferences:
    """Complete MCP CLI preferences."""

    ui: UIPreferences = field(default_factory=UIPreferences)
    provider: ProviderPreferences = field(default_factory=ProviderPreferences)
    servers: ServerPreferences = field(default_factory=ServerPreferences)
    last_servers: Optional[str] = None
    config_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary."""
        return {
            "ui": asdict(self.ui),
            "provider": asdict(self.provider),
            "servers": asdict(self.servers),
            "last_servers": self.last_servers,
            "config_file": self.config_file,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPPreferences":
        """Create preferences from dictionary."""
        ui_data = data.get("ui", {})
        provider_data = data.get("provider", {})
        servers_data = data.get("servers", {})

        return cls(
            ui=UIPreferences(**ui_data) if ui_data else UIPreferences(),
            provider=ProviderPreferences(**provider_data)
            if provider_data
            else ProviderPreferences(),
            servers=ServerPreferences(**servers_data)
            if servers_data
            else ServerPreferences(),
            last_servers=data.get("last_servers"),
            config_file=data.get("config_file"),
        )


class PreferenceManager:
    """Manages MCP CLI preferences with file persistence."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize preference manager.

        Args:
            config_dir: Optional custom config directory, defaults to ~/.mcp-cli
        """
        self.config_dir = config_dir or Path.home() / ".mcp-cli"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.preferences_file = self.config_dir / "preferences.json"
        self.preferences = self.load_preferences()

    def load_preferences(self) -> MCPPreferences:
        """Load preferences from file or create defaults."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, "r") as f:
                    data = json.load(f)
                    return MCPPreferences.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                # If preferences are corrupted, backup and create new
                backup_file = self.preferences_file.with_suffix(".json.backup")
                self.preferences_file.rename(backup_file)
                return MCPPreferences()
        return MCPPreferences()

    def save_preferences(self) -> None:
        """Save preferences to file."""
        with open(self.preferences_file, "w") as f:
            json.dump(self.preferences.to_dict(), f, indent=2)

    def get_theme(self) -> str:
        """Get current theme."""
        return self.preferences.ui.theme

    def set_theme(self, theme: str) -> None:
        """Set and persist theme.

        Args:
            theme: Theme name to set

        Raises:
            ValueError: If theme is not valid
        """
        # Validate theme
        valid_themes = [t.value for t in Theme]
        if theme not in valid_themes:
            raise ValueError(
                f"Invalid theme: {theme}. Valid themes are: {', '.join(valid_themes)}"
            )

        self.preferences.ui.theme = theme
        self.save_preferences()

    def get_verbose(self) -> bool:
        """Get verbose setting."""
        return self.preferences.ui.verbose

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode."""
        self.preferences.ui.verbose = verbose
        self.save_preferences()

    def get_confirm_tools(self) -> bool:
        """Get tool confirmation setting."""
        return self.preferences.ui.confirm_tools

    def set_confirm_tools(self, confirm: bool) -> None:
        """Set tool confirmation mode."""
        self.preferences.ui.confirm_tools = confirm
        self.save_preferences()

    def get_active_provider(self) -> Optional[str]:
        """Get active provider."""
        return self.preferences.provider.active_provider

    def set_active_provider(self, provider: str) -> None:
        """Set active provider."""
        self.preferences.provider.active_provider = provider
        self.save_preferences()

    def get_active_model(self) -> Optional[str]:
        """Get active model."""
        return self.preferences.provider.active_model

    def set_active_model(self, model: str) -> None:
        """Set active model."""
        self.preferences.provider.active_model = model
        self.save_preferences()

    def get_last_servers(self) -> Optional[str]:
        """Get last used servers."""
        return self.preferences.last_servers

    def set_last_servers(self, servers: str) -> None:
        """Set last used servers."""
        self.preferences.last_servers = servers
        self.save_preferences()

    def get_config_file(self) -> Optional[str]:
        """Get default config file path."""
        return self.preferences.config_file

    def set_config_file(self, config_file: str) -> None:
        """Set default config file path."""
        self.preferences.config_file = config_file
        self.save_preferences()

    def reset_preferences(self) -> None:
        """Reset all preferences to defaults."""
        self.preferences = MCPPreferences()
        self.save_preferences()

    def get_history_file(self) -> Path:
        """Get path to chat history file."""
        return self.config_dir / "chat_history"

    def get_logs_dir(self) -> Path:
        """Get path to logs directory."""
        logs_dir = self.config_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def is_server_disabled(self, server_name: str) -> bool:
        """Check if a server is disabled in preferences.

        Args:
            server_name: Name of the server to check

        Returns:
            True if server is disabled, False otherwise
        """
        return self.preferences.servers.disabled_servers.get(server_name, False)

    def set_server_disabled(self, server_name: str, disabled: bool = True) -> None:
        """Set server disabled state in preferences.

        Args:
            server_name: Name of the server
            disabled: Whether server should be disabled
        """
        if disabled:
            self.preferences.servers.disabled_servers[server_name] = True
        else:
            # Remove from disabled list if enabling
            self.preferences.servers.disabled_servers.pop(server_name, None)
        self.save_preferences()

    def enable_server(self, server_name: str) -> None:
        """Enable a server in preferences."""
        self.set_server_disabled(server_name, False)

    def disable_server(self, server_name: str) -> None:
        """Disable a server in preferences."""
        self.set_server_disabled(server_name, True)

    def get_disabled_servers(self) -> Dict[str, bool]:
        """Get all disabled servers."""
        return self.preferences.servers.disabled_servers.copy()

    def clear_disabled_servers(self) -> None:
        """Clear all disabled server preferences."""
        self.preferences.servers.disabled_servers.clear()
        self.save_preferences()

    def add_runtime_server(self, name: str, config: Dict[str, Any]) -> None:
        """Add a runtime server to preferences.

        Args:
            name: Server name
            config: Server configuration (command, args, env, transport, url, etc.)
        """
        self.preferences.servers.runtime_servers[name] = config
        self.save_preferences()

    def remove_runtime_server(self, name: str) -> bool:
        """Remove a runtime server from preferences.

        Args:
            name: Server name to remove

        Returns:
            True if server was removed, False if not found
        """
        if name in self.preferences.servers.runtime_servers:
            del self.preferences.servers.runtime_servers[name]
            self.save_preferences()
            return True
        return False

    def get_runtime_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get all runtime servers."""
        return self.preferences.servers.runtime_servers.copy()

    def get_runtime_server(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific runtime server configuration."""
        return self.preferences.servers.runtime_servers.get(name)

    def is_runtime_server(self, name: str) -> bool:
        """Check if a server is a runtime server."""
        return name in self.preferences.servers.runtime_servers


# Global singleton instance
_preference_manager: Optional[PreferenceManager] = None


def get_preference_manager() -> PreferenceManager:
    """Get or create the global preference manager instance."""
    global _preference_manager
    if _preference_manager is None:
        _preference_manager = PreferenceManager()
    return _preference_manager
