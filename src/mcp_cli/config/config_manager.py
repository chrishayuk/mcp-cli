"""
Centralized configuration manager for MCP CLI.

This module provides a centralized way to manage configuration
instead of loading JSON files all over the place.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from mcp_cli.auth import OAuthConfig
from mcp_cli.tools.models import ServerInfo

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None  # For HTTP/SSE servers
    headers: Optional[Dict[str, str]] = None  # HTTP headers (e.g., Authorization)
    oauth: Optional[OAuthConfig] = None  # OAuth configuration
    disabled: bool = False

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def transport(self) -> str:
        """Determine transport type from config."""
        if self.url:
            return "http"
        elif self.command:
            return "stdio"
        else:
            return "unknown"

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ServerConfig":
        """Create from dictionary format with environment variable handling."""
        # Get env from config
        env = data.get("env", {}).copy()

        # Ensure PATH is inherited from current environment if not explicitly set
        if "PATH" not in env:
            env["PATH"] = os.environ.get("PATH", "")

        # Parse OAuth config if present
        oauth = None
        if "oauth" in data:
            oauth = OAuthConfig.model_validate(data["oauth"])

        return cls(
            name=name,
            command=data.get("command"),
            args=data.get("args", []),
            env=env,
            url=data.get("url"),
            headers=data.get("headers"),
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


class MCPConfig(BaseModel):
    """Complete MCP configuration."""

    servers: Dict[str, ServerConfig] = Field(default_factory=dict)
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

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @classmethod
    def load_from_file(cls, config_path: Path) -> MCPConfig:
        """Load configuration from JSON file."""
        config = cls()

        if not config_path.exists():
            return config

        try:
            # Handle both regular files and package resources
            data_str = config_path.read_text()
            data = json.loads(data_str)

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
                name: server.model_dump(exclude_none=True, exclude_defaults=True)
                for name, server in self.servers.items()
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

        Priority order:
        1. Explicit config_path if provided
        2. server_config.json in current directory (overrides package default)
        3. server_config.json bundled in package (fallback)
        """
        if self._config is None:
            if config_path:
                # Explicit path provided
                self._config_path = Path(config_path)
            else:
                # Check current directory first
                cwd_config = Path("server_config.json")
                if cwd_config.exists():
                    self._config_path = cwd_config
                else:
                    # Fall back to package bundled config
                    import importlib.resources as resources

                    try:
                        # Python 3.9+
                        if hasattr(resources, "files"):
                            package_files = resources.files("mcp_cli")
                            config_file = package_files / "server_config.json"
                            if config_file.is_file():
                                self._config_path = Path(str(config_file))
                            else:
                                # Package config doesn't exist, use cwd path anyway
                                self._config_path = cwd_config
                        else:
                            # Python 3.8 fallback
                            with resources.path("mcp_cli", "server_config.json") as p:
                                if p.exists():
                                    self._config_path = p
                                else:
                                    self._config_path = cwd_config
                    except (ImportError, FileNotFoundError, AttributeError, TypeError):
                        # If package config doesn't exist or can't be accessed, use cwd
                        self._config_path = cwd_config

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


def detect_server_types(
    cfg: MCPConfig, servers: List[str]
) -> Tuple[List[dict], List[str]]:
    """
    Detect which servers are HTTP vs STDIO based on configuration.

    Args:
        cfg: MCPConfig instance
        servers: List of server names to detect

    Returns:
        Tuple of (http_servers_list, stdio_servers_list)
    """
    http_servers = []
    stdio_servers = []

    if not cfg or not cfg.servers:
        # No config, assume all are STDIO
        return [], servers

    for server in servers:
        server_config = cfg.servers.get(server)

        if not server_config:
            logger.warning(f"Server '{server}' not found in configuration")
            stdio_servers.append(server)
            continue

        if server_config.url:
            # HTTP server
            http_servers.append({"name": server, "url": server_config.url})
            logger.debug(f"Detected HTTP server: {server} -> {server_config.url}")
        elif server_config.command:
            # STDIO server
            stdio_servers.append(server)
            logger.debug(f"Detected STDIO server: {server}")
        else:
            logger.warning(
                f"Server '{server}' has unclear configuration, assuming STDIO"
            )
            stdio_servers.append(server)

    return http_servers, stdio_servers


def validate_server_config(
    cfg: MCPConfig, servers: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate server configuration and return status and errors.

    Args:
        cfg: MCPConfig instance
        servers: List of server names to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not cfg or not cfg.servers:
        errors.append("No servers found in configuration")
        return False, errors

    for server in servers:
        if server not in cfg.servers:
            errors.append(f"Server '{server}' not found in configuration")
            continue

        server_config = cfg.servers[server]

        # Check for valid configuration
        has_url = server_config.url is not None
        has_command = server_config.command is not None

        if not has_url and not has_command:
            errors.append(f"Server '{server}' missing both 'url' and 'command' fields")
        elif has_url and has_command:
            errors.append(
                f"Server '{server}' has both 'url' and 'command' fields (should have only one)"
            )
        elif has_url:
            # Validate URL format
            url = server_config.url
            if url and not url.startswith(("http://", "https://")):
                errors.append(
                    f"Server '{server}' URL must start with http:// or https://"
                )
        elif has_command:
            # Validate command format
            command = server_config.command
            if not isinstance(command, str) or not command.strip():
                errors.append(f"Server '{server}' command must be a non-empty string")

    return len(errors) == 0, errors
