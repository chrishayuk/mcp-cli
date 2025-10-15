"""OAuth handler for MCP server connections."""

import logging
from typing import Dict, Optional

from mcp_cli.auth.mcp_oauth import MCPOAuthClient
from mcp_cli.auth.oauth_config import OAuthConfig, OAuthTokens
from mcp_cli.auth.oauth_flow import OAuthFlow
from mcp_cli.auth.token_manager import TokenManager
from mcp_cli.config.config_manager import ServerConfig

logger = logging.getLogger(__name__)


class OAuthHandler:
    """Handles OAuth authentication for MCP servers."""

    def __init__(self, token_manager: Optional[TokenManager] = None):
        """
        Initialize OAuth handler.

        Args:
            token_manager: Token manager instance (creates default if not provided)
        """
        self.token_manager = token_manager or TokenManager()
        self._active_tokens: Dict[str, OAuthTokens] = {}

    async def ensure_authenticated_mcp(
        self, server_name: str, server_url: str, scopes: Optional[list[str]] = None
    ) -> OAuthTokens:
        """
        Ensure remote MCP server has valid authentication using MCP OAuth spec.

        This uses:
        - OAuth Authorization Server Metadata discovery (RFC 8414)
        - Dynamic Client Registration (RFC 7591)
        - Authorization Code Flow with PKCE

        Args:
            server_name: Name of the MCP server
            server_url: Base URL of the MCP server
            scopes: Optional OAuth scopes

        Returns:
            Valid OAuth tokens
        """
        # Check memory cache first
        if server_name in self._active_tokens:
            tokens = self._active_tokens[server_name]
            if not tokens.is_expired():
                return tokens

        # Check disk storage
        stored_tokens = self.token_manager.load_tokens(server_name)
        if stored_tokens and not stored_tokens.is_expired():
            self._active_tokens[server_name] = stored_tokens
            return stored_tokens

        # Create MCP OAuth client
        mcp_client = MCPOAuthClient(server_url)

        # Load stored client registration if exists
        stored_registration = self.token_manager.load_client_registration(server_name)
        if stored_registration:
            mcp_client._client_registration = stored_registration
            logger.info(f"Using stored client registration for {server_name}")

        # Try refresh if we have a refresh token
        if stored_tokens and stored_tokens.refresh_token:
            try:
                # Need to discover metadata first for refresh
                await mcp_client.discover_authorization_server()
                if not mcp_client._client_registration:
                    mcp_client._client_registration = stored_registration

                tokens = await mcp_client.refresh_token(stored_tokens.refresh_token)
                self.token_manager.save_tokens(server_name, tokens)
                self._active_tokens[server_name] = tokens
                logger.info(f"Refreshed tokens for {server_name}")
                return tokens
            except Exception as e:
                logger.warning(f"Token refresh failed for {server_name}: {e}")
                # Fall through to full auth flow

        # Perform full MCP OAuth flow
        print(f"\n🔐 Authentication required for {server_name}")
        print("=" * 60)
        tokens = await mcp_client.authorize(scopes)

        # Save both tokens and client registration
        self.token_manager.save_tokens(server_name, tokens)
        if mcp_client._client_registration:
            self.token_manager.save_client_registration(
                server_name, mcp_client._client_registration
            )

        self._active_tokens[server_name] = tokens
        logger.info(f"Completed MCP OAuth flow for {server_name}")

        return tokens

    async def ensure_authenticated(
        self, server_name: str, oauth_config: OAuthConfig
    ) -> OAuthTokens:
        """
        Ensure server has valid authentication tokens (legacy OAuth).

        This method:
        1. Checks for cached tokens in memory
        2. Checks for stored tokens on disk
        3. Refreshes expired tokens if refresh token available
        4. Performs full OAuth flow if no valid tokens exist

        Args:
            server_name: Name of the MCP server
            oauth_config: OAuth configuration for the server

        Returns:
            Valid OAuth tokens

        Raises:
            Exception: If authentication fails
        """
        # Check memory cache first
        if server_name in self._active_tokens:
            tokens = self._active_tokens[server_name]
            if not tokens.is_expired():
                return tokens

        # Check disk storage
        stored_tokens = self.token_manager.load_tokens(server_name)
        if stored_tokens and not stored_tokens.is_expired():
            self._active_tokens[server_name] = stored_tokens
            return stored_tokens

        # Try refresh if we have a refresh token
        if stored_tokens and stored_tokens.refresh_token:
            try:
                tokens = await self._refresh_tokens(
                    server_name, oauth_config, stored_tokens.refresh_token
                )
                return tokens
            except Exception as e:
                logger.warning(f"Token refresh failed for {server_name}: {e}")
                # Fall through to full auth flow

        # Perform full OAuth flow
        tokens = await self._perform_oauth_flow(server_name, oauth_config)
        return tokens

    async def _refresh_tokens(
        self, server_name: str, oauth_config: OAuthConfig, refresh_token: str
    ) -> OAuthTokens:
        """Refresh access token."""
        flow = OAuthFlow(oauth_config)
        tokens = await flow.refresh_token(refresh_token)

        # Save and cache new tokens
        self.token_manager.save_tokens(server_name, tokens)
        self._active_tokens[server_name] = tokens

        logger.info(f"Refreshed tokens for {server_name}")
        return tokens

    async def _perform_oauth_flow(
        self, server_name: str, oauth_config: OAuthConfig
    ) -> OAuthTokens:
        """Perform full OAuth authorization flow."""
        flow = OAuthFlow(oauth_config)

        print(f"\n🔐 Authentication required for {server_name}")
        print("=" * 60)

        tokens = await flow.authorize()

        # Save and cache tokens
        self.token_manager.save_tokens(server_name, tokens)
        self._active_tokens[server_name] = tokens

        print(f"✅ Successfully authenticated {server_name}\n")
        logger.info(f"Completed OAuth flow for {server_name}")

        return tokens

    def get_authorization_header(self, server_name: str) -> Optional[str]:
        """
        Get Authorization header value for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Authorization header value or None if not authenticated
        """
        if server_name in self._active_tokens:
            return self._active_tokens[server_name].get_authorization_header()
        return None

    def clear_tokens(self, server_name: str) -> None:
        """
        Clear tokens for a server (from memory and disk).

        Args:
            server_name: Name of the MCP server
        """
        if server_name in self._active_tokens:
            del self._active_tokens[server_name]
        self.token_manager.delete_tokens(server_name)

    async def prepare_server_headers(
        self, server_config: ServerConfig
    ) -> Dict[str, str]:
        """
        Prepare HTTP headers for server connection, including OAuth if configured.

        For remote MCP servers (HTTP/SSE), this uses the MCP OAuth specification.
        For other servers, it uses legacy OAuth configuration.

        Args:
            server_config: Server configuration

        Returns:
            Dictionary of HTTP headers
        """
        headers: Dict[str, str] = {}

        # Determine if this is a remote MCP server (has URL, no command)
        is_remote_mcp = server_config.url and not server_config.command

        if is_remote_mcp and server_config.url:
            # Use MCP OAuth specification for remote servers
            try:
                tokens = await self.ensure_authenticated_mcp(
                    server_config.name,
                    server_config.url,
                    scopes=server_config.oauth.scopes if server_config.oauth else None,
                )
                auth_header = tokens.get_authorization_header()
                headers["Authorization"] = auth_header
                print(
                    f"✓ Added Authorization header for {server_config.name}: {auth_header[:30]}..."
                )
                logger.info(
                    f"Added Authorization header for {server_config.name}: {auth_header[:20]}..."
                )
            except Exception as e:
                logger.error(
                    f"MCP OAuth authentication failed for {server_config.name}: {e}"
                )
                raise
        elif server_config.oauth:
            # Use legacy OAuth for servers with explicit config
            try:
                tokens = await self.ensure_authenticated(
                    server_config.name, server_config.oauth
                )
                headers["Authorization"] = tokens.get_authorization_header()
            except Exception as e:
                logger.error(
                    f"OAuth authentication failed for {server_config.name}: {e}"
                )
                raise

        # Merge with any existing headers from env
        if server_config.env:
            # Common header patterns
            for key, value in server_config.env.items():
                if key.lower().startswith("http_header_"):
                    header_name = key[12:].replace("_", "-")
                    headers[header_name] = value

        return headers
