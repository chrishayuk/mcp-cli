# mcp_cli/tools/config_loader.py
"""MCP configuration loading and OAuth integration.

Handles parsing MCP config files, detecting server types, and OAuth token management.
Async-native with proper type safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from mcp_cli.auth import OAuthHandler, TokenManager, TokenStoreBackend
from mcp_cli.config.server_models import HTTPServerConfig, STDIOServerConfig
from mcp_cli.constants import NAMESPACE
from mcp_cli.tools.models import TransportType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants - no magic strings!
# ──────────────────────────────────────────────────────────────────────────────
TOKEN_PLACEHOLDER_PREFIX = "{{token:"
TOKEN_PLACEHOLDER_SUFFIX = "}}"
CONFIG_KEY_MCP_SERVERS = "mcpServers"


class ConfigLoader:
    """Loads and processes MCP configuration files with OAuth support."""

    def __init__(self, config_file: str, servers: list[str]) -> None:
        """Initialize config loader.

        Args:
            config_file: Path to MCP config JSON file
            servers: List of server names to load
        """
        self.config_file = config_file
        self.servers = servers
        self._config_cache: dict[str, Any] | None = None

        # Token manager for OAuth
        self._token_manager = TokenManager(
            backend=TokenStoreBackend.AUTO,
            namespace=NAMESPACE,
            service_name="mcp-cli",
        )

        # Detected servers by transport type
        self.http_servers: list[HTTPServerConfig] = []
        self.sse_servers: list[HTTPServerConfig] = []
        self.stdio_servers: list[STDIOServerConfig] = []

    def load(self) -> dict[str, Any]:
        """Load and parse MCP config file with token resolution (sync).

        For async contexts, prefer load_async() to avoid blocking the event loop.

        Returns:
            Parsed config dict, or empty dict on error
        """
        if self._config_cache:
            return self._config_cache

        try:
            with open(self.config_file) as f:
                config = cast(dict[str, Any], json.load(f))

            # Resolve {{token:provider}} placeholders
            self._resolve_token_placeholders(config)

            self._config_cache = config
            return config

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    async def load_async(self) -> dict[str, Any]:
        """Load and parse MCP config file with token resolution (async).

        Uses asyncio.to_thread() to avoid blocking the event loop during file I/O.

        Returns:
            Parsed config dict, or empty dict on error
        """
        if self._config_cache:
            return self._config_cache

        try:
            # Use asyncio.to_thread for non-blocking file I/O
            def _read_file() -> str:
                with open(self.config_file) as f:
                    return f.read()

            content = await asyncio.to_thread(_read_file)
            config = cast(dict[str, Any], json.loads(content))

            # Resolve {{token:provider}} placeholders
            self._resolve_token_placeholders(config)

            self._config_cache = config
            return config

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _resolve_token_placeholders(self, config: dict[str, Any]) -> None:
        """Replace {{token:provider}} with actual OAuth tokens."""
        prefix_len = len(TOKEN_PLACEHOLDER_PREFIX)
        suffix_len = len(TOKEN_PLACEHOLDER_SUFFIX)

        def process_value(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(TOKEN_PLACEHOLDER_PREFIX):
                # Extract provider name using constants
                provider = value[prefix_len:-suffix_len]
                try:
                    tokens = self._token_manager.load_tokens(provider)
                    if tokens and tokens.access_token:
                        return f"Bearer {tokens.access_token}"
                except Exception as e:
                    logger.warning(f"Failed to get token for {provider}: {e}")
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        # Process entire config using constant
        if CONFIG_KEY_MCP_SERVERS in config:
            config[CONFIG_KEY_MCP_SERVERS] = process_value(
                config[CONFIG_KEY_MCP_SERVERS]
            )

    def detect_server_types(self, config: dict[str, Any]) -> None:
        """Detect HTTP/SSE/STDIO servers from config and populate server lists."""
        mcp_servers = config.get(CONFIG_KEY_MCP_SERVERS, {})

        # Clear existing lists
        self.http_servers.clear()
        self.sse_servers.clear()
        self.stdio_servers.clear()

        for server_name in self.servers:
            if server_name not in mcp_servers:
                logger.warning(f"Server '{server_name}' not found in config")
                continue

            server_cfg = mcp_servers[server_name]

            if "url" in server_cfg:
                transport_str = server_cfg.get("transport", "").lower()
                http_config = HTTPServerConfig(
                    name=server_name,
                    url=server_cfg["url"],
                    headers=server_cfg.get("headers", {}),
                    disabled=server_cfg.get("disabled", False),
                )

                # Use TransportType enum for comparison - no magic strings!
                if TransportType.SSE.value in transport_str:
                    self.sse_servers.append(http_config)
                else:
                    self.http_servers.append(http_config)
            else:
                # STDIO server
                stdio_config = STDIOServerConfig(
                    name=server_name,
                    command=server_cfg.get("command", ""),
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env", {}),
                    disabled=server_cfg.get("disabled", False),
                )
                self.stdio_servers.append(stdio_config)

    def create_oauth_refresh_callback(
        self,
        http_servers: list[HTTPServerConfig],
        sse_servers: list[HTTPServerConfig],
    ):
        """Create OAuth token refresh callback for StreamManager.

        Args:
            http_servers: List of HTTP server configs
            sse_servers: List of SSE server configs

        Returns:
            Async callback function for token refresh
        """

        async def refresh_oauth_token(
            server_url: str | None = None,
        ) -> dict[str, str] | None:
            """
            Refresh OAuth token for a server and return updated headers.

            Args:
                server_url: URL of the server that needs token refresh

            Returns:
                Dictionary with updated Authorization header, or None if refresh failed
            """
            logger.info(f"OAuth token refresh triggered for URL: {server_url}")

            if not server_url:
                logger.warning("Cannot refresh OAuth token: server URL not provided")
                return None

            # Map URL back to server name
            base_url = server_url.replace("/mcp", "")
            server_name = None

            for server_list in [http_servers, sse_servers]:
                for server_config in server_list:
                    config_url = server_config.url.replace("/mcp", "")
                    if config_url == base_url or server_config.url == server_url:
                        server_name = server_config.name
                        break
                if server_name:
                    break

            if not server_name:
                logger.error(f"Cannot map URL {server_url} to a known server")
                return None

            logger.debug(f"Mapped URL {server_url} to server: {server_name}")

            try:
                # Get token manager
                token_mgr = TokenManager(
                    backend=TokenStoreBackend.AUTO,
                    namespace=NAMESPACE,
                    service_name="mcp-cli",
                )

                # Get existing token data
                token_data = token_mgr.get_token(server_name)

                if not token_data:
                    logger.warning(f"No token found for server: {server_name}")
                    return None

                # Check if we have a refresh token
                refresh_token = token_data.get("refresh_token")

                if not refresh_token:
                    logger.warning(
                        f"No refresh_token available for server: {server_name}, "
                        "re-authentication required"
                    )
                    return None

                # Attempt to refresh the token
                oauth_handler = OAuthHandler(base_url)

                logger.debug(f"Attempting to refresh OAuth token for {server_name}...")
                new_tokens = await oauth_handler.refresh_access_token(refresh_token)

                if not new_tokens or "access_token" not in new_tokens:
                    logger.error(f"Token refresh failed for {server_name}")
                    return None

                # Store the new tokens
                token_mgr.store_token(
                    name=server_name, token_data=new_tokens, token_type="oauth"
                )

                logger.info(f"OAuth token refreshed successfully for {server_name}")

                return {"Authorization": f"Bearer {new_tokens['access_token']}"}

            except Exception as e:
                logger.error(
                    f"OAuth token refresh failed for {server_name}: {e}", exc_info=True
                )
                return None

        return refresh_oauth_token
