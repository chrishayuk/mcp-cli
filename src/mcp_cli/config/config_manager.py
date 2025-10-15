"""
Centralized configuration manager for MCP CLI.

This module provides a centralized way to manage configuration
instead of loading JSON files all over the place.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp_cli.auth.oauth_config import OAuthConfig
from mcp_cli.tools.models import ServerInfo


@dataclass
class ServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # For HTTP/SSE servers
    oauth: Optional[OAuthConfig] = None  # OAuth configuration
    disabled: bool = False

    @property
    def transport(self) -> str:
        """Determine transport type from config."""
        if self.url:
            return "http"
        elif self.command:
            return "stdio"
        else:
            return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        config: Dict[str, Any] = {}
        if self.command:
            config["command"] = self.command
        if self.args:
            config["args"] = self.args
        if self.env:
            config["env"] = self.env
        if self.url:
            config["url"] = self.url
        if self.oauth:
            config["oauth"] = self.oauth.to_dict()
        if self.disabled:
            config["disabled"] = self.disabled
        return config

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> ServerConfig:
        """Create from dictionary format."""
        # Get env from config
        env = data.get("env", {}).copy()

        # Ensure PATH is inherited from current environment if not explicitly set
        if "PATH" not in env:
            env["PATH"] = os.environ.get("PATH", "")

        # Parse OAuth config if present
        oauth = None
        if "oauth" in data:
            oauth = OAuthConfig.from_dict(data["oauth"])

        return cls(
            name=name,
            command=data.get("command"),
            args=data.get("args", []),
            env=env,
            url=data.get("url"),
            oauth=oauth,
            disabled=data.get("disabled", False),
        )

    def to_server_info(self, server_id: int = 0) -> ServerInfo:
        """Convert to ServerInfo model."""
        return ServerInfo(
            id=server_id,
            name=self.name,
            status="configured",
            tool_count=0,
            namespace=self.name,
            enabled=not self.disabled,
            connected=False,
            transport=self.transport,
            capabilities={},
            command=self.command,
            args=self.args,
            env=self.env,
        )


@dataclass
class MCPConfig:
    """Complete MCP configuration."""

    servers: Dict[str, ServerConfig] = field(default_factory=dict)
    default_provider: str = "openai"
    default_model: str = "gpt-4"
    theme: str = "default"
    verbose: bool = True
    confirm_tools: bool = True

    # Token storage configuration
    token_store_backend: str = (
        "auto"  # auto, keychain, windows, secretservice, vault, encrypted
    )
    token_store_password: Optional[str] = None
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    vault_mount_point: str = "secret"
    vault_path_prefix: str = "mcp-cli/oauth"
    vault_namespace: Optional[str] = None

    @classmethod
    def load_from_file(cls, config_path: Path) -> MCPConfig:
        """Load configuration from JSON file."""
        config = cls()

        if not config_path.exists():
            return config

        try:
            with open(config_path, "r") as f:
                data = json.load(f)

            # Load servers
            if "mcpServers" in data:
                for name, server_data in data["mcpServers"].items():
                    config.servers[name] = ServerConfig.from_dict(name, server_data)

            # Load other settings
            config.default_provider = data.get("defaultProvider", "openai")
            config.default_model = data.get("defaultModel", "gpt-4")
            config.theme = data.get("theme", "default")
            config.verbose = data.get("verbose", True)
            config.confirm_tools = data.get("confirmTools", True)

            # Load token storage configuration
            token_storage = data.get("tokenStorage", {})
            config.token_store_backend = token_storage.get("backend", "auto")
            config.token_store_password = token_storage.get("password")
            config.vault_url = token_storage.get("vaultUrl")
            config.vault_token = token_storage.get("vaultToken")
            config.vault_mount_point = token_storage.get("vaultMountPoint", "secret")
            config.vault_path_prefix = token_storage.get(
                "vaultPathPrefix", "mcp-cli/oauth"
            )
            config.vault_namespace = token_storage.get("vaultNamespace")

        except Exception as e:
            # Log error but return empty config
            print(f"Error loading config: {e}")

        return config

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            "mcpServers": {
                name: server.to_dict() for name, server in self.servers.items()
            },
            "defaultProvider": self.default_provider,
            "defaultModel": self.default_model,
            "theme": self.theme,
            "verbose": self.verbose,
            "confirmTools": self.confirm_tools,
        }

        # Add token storage configuration if non-default
        token_storage: Dict[str, Any] = {}
        if self.token_store_backend != "auto":
            token_storage["backend"] = self.token_store_backend
        if self.token_store_password:
            token_storage["password"] = self.token_store_password
        if self.vault_url:
            token_storage["vaultUrl"] = self.vault_url
        if self.vault_token:
            token_storage["vaultToken"] = self.vault_token
        if self.vault_mount_point != "secret":
            token_storage["vaultMountPoint"] = self.vault_mount_point
        if self.vault_path_prefix != "mcp-cli/oauth":
            token_storage["vaultPathPrefix"] = self.vault_path_prefix
        if self.vault_namespace:
            token_storage["vaultNamespace"] = self.vault_namespace

        if token_storage:
            data["tokenStorage"] = token_storage

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_server(self, name: str) -> Optional[ServerConfig]:
        """Get a server configuration by name."""
        return self.servers.get(name)

    def add_server(self, server: ServerConfig) -> None:
        """Add or update a server configuration."""
        self.servers[server.name] = server

    def remove_server(self, name: str) -> bool:
        """Remove a server configuration."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False

    def list_servers(self) -> List[ServerConfig]:
        """Get list of all server configurations."""
        return list(self.servers.values())

    def list_enabled_servers(self) -> List[ServerConfig]:
        """Get list of enabled server configurations."""
        return [s for s in self.servers.values() if not s.disabled]


class ConfigManager:
    """
    Manager for application configuration.

    This provides a singleton-like pattern for managing configuration.
    """

    _instance: Optional[ConfigManager] = None
    _config: Optional[MCPConfig] = None
    _config_path: Optional[Path] = None

    def __new__(cls) -> ConfigManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, config_path: Optional[Path] = None) -> MCPConfig:
        """
        Initialize or get the configuration.
        """
        if self._config is None:
            self._config_path = config_path or Path("server_config.json")
            self._config = MCPConfig.load_from_file(self._config_path)
        return self._config

    def get_config(self) -> MCPConfig:
        """
        Get the current configuration.

        Raises:
            RuntimeError: If config hasn't been initialized
        """
        if self._config is None:
            raise RuntimeError("Config not initialized. Call initialize() first.")
        return self._config

    def save(self) -> None:
        """Save the current configuration to file."""
        if self._config and self._config_path:
            self._config.save_to_file(self._config_path)

    def reload(self) -> MCPConfig:
        """Reload configuration from file."""
        if self._config_path:
            self._config = MCPConfig.load_from_file(self._config_path)
            return self._config
        raise RuntimeError("No config path set")

    def reset(self) -> None:
        """Reset the configuration (useful for testing)."""
        self._config = None
        self._config_path = None


def get_config() -> MCPConfig:
    """
    Convenience function to get the current configuration.

    Returns:
        The current MCPConfig

    Raises:
        RuntimeError: If config hasn't been initialized
    """
    manager = ConfigManager()
    return manager.get_config()


def initialize_config(config_path: Optional[Path] = None) -> MCPConfig:
    """
    Convenience function to initialize the configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        The initialized MCPConfig
    """
    manager = ConfigManager()
    return manager.initialize(config_path)
