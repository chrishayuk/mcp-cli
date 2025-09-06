"""
Centralized preference management for MCP CLI.

This module handles all user preferences including themes, provider settings,
model preferences, and other configuration options in a centralized way.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List
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


class ConfirmationMode(str, Enum):
    """Global tool confirmation modes."""

    ALWAYS = "always"  # Always ask for confirmation
    NEVER = "never"  # Never ask for confirmation
    SMART = "smart"  # Smart mode based on risk level


class ToolRiskLevel(str, Enum):
    """Risk levels for tools."""

    SAFE = "safe"  # Read-only operations
    MODERATE = "moderate"  # Local modifications
    HIGH = "high"  # System-wide or destructive operations


@dataclass
class ToolConfirmationPreferences:
    """Tool confirmation preferences."""

    mode: str = "smart"  # Global confirmation mode
    per_tool: Dict[str, str] = field(
        default_factory=dict
    )  # Per-tool overrides (always/never/ask)
    patterns: List[Dict[str, str]] = field(default_factory=list)  # Pattern-based rules
    risk_thresholds: Dict[str, bool] = field(
        default_factory=lambda: {
            "safe": False,  # Don't confirm safe tools
            "moderate": True,  # Confirm moderate risk tools
            "high": True,  # Always confirm high risk tools
        }
    )
    categories: Dict[str, str] = field(
        default_factory=lambda: {
            # Default risk categories for common tool patterns
            "read_*": "safe",
            "list_*": "safe",
            "get_*": "safe",
            "describe_*": "safe",
            "write_*": "moderate",
            "create_*": "moderate",
            "update_*": "moderate",
            "delete_*": "high",
            "remove_*": "high",
            "execute_*": "high",
            "run_*": "high",
        }
    )


@dataclass
class UIPreferences:
    """UI-related preferences."""

    theme: str = "default"
    verbose: bool = True
    confirm_tools: bool = True
    show_reasoning: bool = True
    tool_confirmation: ToolConfirmationPreferences = field(
        default_factory=ToolConfirmationPreferences
    )


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

        # Handle tool_confirmation nested data
        if ui_data and "tool_confirmation" in ui_data:
            tool_conf_data = ui_data["tool_confirmation"]
            if isinstance(tool_conf_data, dict):
                ui_data["tool_confirmation"] = ToolConfirmationPreferences(
                    **tool_conf_data
                )

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
        """Get tool confirmation setting (legacy compatibility)."""
        return self.preferences.ui.tool_confirmation.mode != "never"

    def set_confirm_tools(self, confirm: bool) -> None:
        """Set tool confirmation mode (legacy compatibility)."""
        self.preferences.ui.confirm_tools = confirm
        # Also update new confirmation mode
        self.preferences.ui.tool_confirmation.mode = "smart" if confirm else "never"
        self.save_preferences()

    def get_tool_confirmation_mode(self) -> str:
        """Get the global tool confirmation mode."""
        return self.preferences.ui.tool_confirmation.mode

    def set_tool_confirmation_mode(self, mode: str) -> None:
        """Set the global tool confirmation mode.

        Args:
            mode: One of 'always', 'never', or 'smart'
        """
        if mode not in [m.value for m in ConfirmationMode]:
            raise ValueError(f"Invalid confirmation mode: {mode}")
        self.preferences.ui.tool_confirmation.mode = mode
        # Update legacy flag
        self.preferences.ui.confirm_tools = mode != "never"
        self.save_preferences()

    def get_tool_confirmation(self, tool_name: str) -> Optional[str]:
        """Get confirmation setting for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            'always', 'never', 'ask', or None (use global default)
        """
        return self.preferences.ui.tool_confirmation.per_tool.get(tool_name)

    def set_tool_confirmation(self, tool_name: str, setting: Optional[str]) -> None:
        """Set confirmation for a specific tool.

        Args:
            tool_name: Name of the tool
            setting: 'always', 'never', 'ask', or None (remove override)
        """
        if setting is None:
            self.preferences.ui.tool_confirmation.per_tool.pop(tool_name, None)
        else:
            if setting not in ["always", "never", "ask"]:
                raise ValueError(f"Invalid tool confirmation setting: {setting}")
            self.preferences.ui.tool_confirmation.per_tool[tool_name] = setting
        self.save_preferences()

    def get_all_tool_confirmations(self) -> Dict[str, str]:
        """Get all per-tool confirmation settings."""
        return self.preferences.ui.tool_confirmation.per_tool.copy()

    def clear_tool_confirmations(self) -> None:
        """Clear all per-tool confirmation settings."""
        self.preferences.ui.tool_confirmation.per_tool.clear()
        self.save_preferences()

    def get_tool_risk_level(self, tool_name: str) -> str:
        """Determine the risk level of a tool based on patterns.

        Args:
            tool_name: Name of the tool

        Returns:
            Risk level: 'safe', 'moderate', or 'high'
        """
        # Check if tool matches any category pattern
        for pattern, risk in self.preferences.ui.tool_confirmation.categories.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if tool_name.startswith(prefix):
                    return risk
            elif pattern.startswith("*"):
                suffix = pattern[1:]
                if tool_name.endswith(suffix):
                    return risk

        # Default to moderate risk
        return "moderate"

    def should_confirm_tool(self, tool_name: str) -> bool:
        """Determine if a tool should be confirmed based on preferences.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool should be confirmed, False otherwise
        """
        # Check per-tool override first
        tool_setting = self.get_tool_confirmation(tool_name)
        if tool_setting == "always":
            return True
        elif tool_setting == "never":
            return False
        elif tool_setting == "ask":
            # Use global mode
            pass

        # Check global mode
        mode = self.get_tool_confirmation_mode()
        if mode == "always":
            return True
        elif mode == "never":
            return False
        elif mode == "smart":
            # Use risk-based decision
            risk_level = self.get_tool_risk_level(tool_name)
            return self.preferences.ui.tool_confirmation.risk_thresholds.get(
                risk_level, True
            )

        # Default to confirming
        return True

    def add_tool_pattern(self, pattern: str, action: str) -> None:
        """Add a pattern-based rule for tool confirmations.

        Args:
            pattern: Glob pattern for tool names
            action: 'always', 'never', or risk level
        """
        self.preferences.ui.tool_confirmation.patterns.append(
            {"pattern": pattern, "action": action}
        )
        self.save_preferences()

    def remove_tool_pattern(self, pattern: str) -> bool:
        """Remove a pattern-based rule.

        Args:
            pattern: The pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        patterns = self.preferences.ui.tool_confirmation.patterns
        original_len = len(patterns)
        self.preferences.ui.tool_confirmation.patterns = [
            p for p in patterns if p.get("pattern") != pattern
        ]
        if len(self.preferences.ui.tool_confirmation.patterns) < original_len:
            self.save_preferences()
            return True
        return False

    def set_risk_threshold(self, risk_level: str, should_confirm: bool) -> None:
        """Set whether to confirm tools at a specific risk level.

        Args:
            risk_level: 'safe', 'moderate', or 'high'
            should_confirm: Whether to confirm tools at this risk level
        """
        if risk_level not in ["safe", "moderate", "high"]:
            raise ValueError(f"Invalid risk level: {risk_level}")
        self.preferences.ui.tool_confirmation.risk_thresholds[risk_level] = (
            should_confirm
        )
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


__all__ = [
    "PreferenceManager",
    "get_preference_manager",
    "MCPPreferences",
    "UIPreferences",
    "ProviderPreferences",
    "ServerPreferences",
    "ToolConfirmationPreferences",
    "Theme",
    "ConfirmationMode",
    "ToolRiskLevel",
]


def get_preference_manager() -> PreferenceManager:
    """Get or create the global preference manager instance."""
    global _preference_manager
    if _preference_manager is None:
        _preference_manager = PreferenceManager()
    return _preference_manager
