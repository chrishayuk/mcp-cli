"""Tests for OAuthHandler."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.auth.oauth_config import OAuthConfig, OAuthTokens
from mcp_cli.auth.oauth_handler import OAuthHandler
from mcp_cli.config.config_manager import ServerConfig


class TestOAuthHandlerInit:
    """Test OAuthHandler initialization."""

    def test_init_without_token_manager(self):
        """Test initialization without token manager creates default."""
        handler = OAuthHandler()
        assert handler.token_manager is not None
        assert handler._active_tokens == {}

    def test_init_with_token_manager(self):
        """Test initialization with provided token manager."""
        mock_tm = MagicMock()
        handler = OAuthHandler(token_manager=mock_tm)
        assert handler.token_manager == mock_tm


class TestOAuthHandlerMCPAuth:
    """Test MCP OAuth authentication."""

    @pytest.fixture
    def handler(self):
        """Provide OAuthHandler with mock token manager."""
        mock_tm = MagicMock()
        return OAuthHandler(token_manager=mock_tm)

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-token",
            refresh_token="refresh-token",
            expires_in=3600,
            token_type="Bearer",
        )

    @pytest.mark.asyncio
    async def test_ensure_authenticated_mcp_cached_valid(self, handler, sample_tokens):
        """Test that cached valid tokens are returned."""
        handler._active_tokens["test-server"] = sample_tokens

        result = await handler.ensure_authenticated_mcp("test-server", "http://server")

        assert result == sample_tokens
        handler.token_manager.load_tokens.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_authenticated_mcp_from_storage(self, handler, sample_tokens):
        """Test loading valid tokens from storage."""
        handler.token_manager.load_tokens.return_value = sample_tokens

        result = await handler.ensure_authenticated_mcp("test-server", "http://server")

        assert result == sample_tokens
        assert handler._active_tokens["test-server"] == sample_tokens

    @pytest.mark.asyncio
    async def test_ensure_authenticated_mcp_refresh_failure_falls_through(
        self, handler, sample_tokens
    ):
        """Test that refresh failure falls through to full auth."""
        import time

        # Expired token - set issued_at in the past
        expired_tokens = OAuthTokens(
            access_token="old-token",
            refresh_token="refresh-token",
            expires_in=100,  # 100 seconds
            token_type="Bearer",
            issued_at=time.time() - 200,  # Issued 200 seconds ago
        )

        handler.token_manager.load_tokens.return_value = expired_tokens
        handler.token_manager.load_client_registration.return_value = MagicMock()

        # Mock MCPOAuthClient
        with patch("mcp_cli.auth.oauth_handler.MCPOAuthClient") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.discover_authorization_server = AsyncMock()
            # Make refresh fail
            mock_client.refresh_token = AsyncMock(
                side_effect=Exception("Refresh failed")
            )
            # But full auth succeeds
            mock_client.authorize = AsyncMock(return_value=sample_tokens)
            mock_client._client_registration = MagicMock()

            result = await handler.ensure_authenticated_mcp(
                "test-server", "http://server"
            )

            # Should get tokens from full auth flow
            assert result == sample_tokens
            mock_client.authorize.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_authenticated_mcp_full_flow(self, handler, sample_tokens):
        """Test full MCP OAuth flow."""
        handler.token_manager.load_tokens.return_value = None

        with patch("mcp_cli.auth.oauth_handler.MCPOAuthClient") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.authorize = AsyncMock(return_value=sample_tokens)
            mock_client._client_registration = MagicMock()

            result = await handler.ensure_authenticated_mcp(
                "test-server", "http://server"
            )

            assert result == sample_tokens
            mock_client.authorize.assert_called_once()
            handler.token_manager.save_tokens.assert_called_once()
            handler.token_manager.save_client_registration.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_authenticated_mcp_with_scopes(self, handler, sample_tokens):
        """Test MCP OAuth with custom scopes."""
        handler.token_manager.load_tokens.return_value = None

        with patch("mcp_cli.auth.oauth_handler.MCPOAuthClient") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.authorize = AsyncMock(return_value=sample_tokens)
            mock_client._client_registration = MagicMock()

            await handler.ensure_authenticated_mcp(
                "test-server", "http://server", scopes=["read", "write"]
            )

            mock_client.authorize.assert_called_once_with(["read", "write"])


class TestOAuthHandlerLegacyAuth:
    """Test legacy OAuth authentication."""

    @pytest.fixture
    def handler(self):
        """Provide OAuthHandler with mock token manager."""
        mock_tm = MagicMock()
        return OAuthHandler(token_manager=mock_tm)

    @pytest.fixture
    def oauth_config(self):
        """Provide OAuth config."""
        return OAuthConfig(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="http://auth",
            token_url="http://token",
            redirect_uri="http://callback",
        )

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-token",
            expires_in=3600,
            token_type="Bearer",
        )

    @pytest.mark.asyncio
    async def test_ensure_authenticated_cached(
        self, handler, oauth_config, sample_tokens
    ):
        """Test cached tokens are returned."""
        handler._active_tokens["test-server"] = sample_tokens

        result = await handler.ensure_authenticated("test-server", oauth_config)

        assert result == sample_tokens

    @pytest.mark.asyncio
    async def test_ensure_authenticated_from_storage(
        self, handler, oauth_config, sample_tokens
    ):
        """Test loading from storage."""
        handler.token_manager.load_tokens.return_value = sample_tokens

        result = await handler.ensure_authenticated("test-server", oauth_config)

        assert result == sample_tokens
        assert handler._active_tokens["test-server"] == sample_tokens

    @pytest.mark.asyncio
    async def test_refresh_tokens(self, handler, oauth_config, sample_tokens):
        """Test token refresh."""
        new_tokens = OAuthTokens(
            access_token="new-token",
            expires_in=3600,
            token_type="Bearer",
        )

        with patch("mcp_cli.auth.oauth_handler.OAuthFlow") as mock_flow_class:
            mock_flow = mock_flow_class.return_value
            mock_flow.refresh_token = AsyncMock(return_value=new_tokens)

            result = await handler._refresh_tokens(
                "test-server", oauth_config, "refresh-token"
            )

            assert result == new_tokens
            handler.token_manager.save_tokens.assert_called_once_with(
                "test-server", new_tokens
            )
            assert handler._active_tokens["test-server"] == new_tokens

    @pytest.mark.asyncio
    async def test_perform_oauth_flow(self, handler, oauth_config, sample_tokens):
        """Test performing full OAuth flow."""
        with patch("mcp_cli.auth.oauth_handler.OAuthFlow") as mock_flow_class:
            mock_flow = mock_flow_class.return_value
            mock_flow.authorize = AsyncMock(return_value=sample_tokens)

            result = await handler._perform_oauth_flow("test-server", oauth_config)

            assert result == sample_tokens
            handler.token_manager.save_tokens.assert_called_once()
            assert handler._active_tokens["test-server"] == sample_tokens

    @pytest.mark.asyncio
    async def test_ensure_authenticated_refresh_failure_falls_through(
        self, handler, oauth_config, sample_tokens
    ):
        """Test that refresh failure falls through to full auth."""
        import time

        expired_tokens = OAuthTokens(
            access_token="old-token",
            refresh_token="refresh-token",
            expires_in=100,
            token_type="Bearer",
            issued_at=time.time() - 200,  # Expired
        )

        handler.token_manager.load_tokens.return_value = expired_tokens

        with patch("mcp_cli.auth.oauth_handler.OAuthFlow") as mock_flow_class:
            mock_flow = mock_flow_class.return_value
            # Make refresh fail
            mock_flow.refresh_token = AsyncMock(side_effect=Exception("Refresh failed"))
            # But full auth succeeds
            mock_flow.authorize = AsyncMock(return_value=sample_tokens)

            result = await handler.ensure_authenticated("test-server", oauth_config)

            # Should get tokens from full auth
            assert result == sample_tokens
            mock_flow.authorize.assert_called_once()


class TestOAuthHandlerUtilityMethods:
    """Test utility methods."""

    @pytest.fixture
    def handler(self):
        """Provide OAuthHandler with mock token manager."""
        mock_tm = MagicMock()
        return OAuthHandler(token_manager=mock_tm)

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-token",
            expires_in=3600,
            token_type="Bearer",
        )

    def test_get_authorization_header(self, handler, sample_tokens):
        """Test getting authorization header."""
        handler._active_tokens["test-server"] = sample_tokens

        result = handler.get_authorization_header("test-server")

        assert result == "Bearer test-token"

    def test_get_authorization_header_not_found(self, handler):
        """Test getting header for non-existent server."""
        result = handler.get_authorization_header("nonexistent")

        assert result is None

    def test_clear_tokens(self, handler, sample_tokens):
        """Test clearing tokens."""
        handler._active_tokens["test-server"] = sample_tokens

        handler.clear_tokens("test-server")

        assert "test-server" not in handler._active_tokens
        handler.token_manager.delete_tokens.assert_called_once_with("test-server")

    def test_clear_tokens_not_in_cache(self, handler):
        """Test clearing tokens not in cache."""
        handler.clear_tokens("nonexistent")

        handler.token_manager.delete_tokens.assert_called_once_with("nonexistent")


class TestOAuthHandlerPrepareHeaders:
    """Test prepare_server_headers method."""

    @pytest.fixture
    def handler(self):
        """Provide OAuthHandler with mock token manager."""
        mock_tm = MagicMock()
        return OAuthHandler(token_manager=mock_tm)

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-token",
            expires_in=3600,
            token_type="Bearer",
        )

    @pytest.mark.asyncio
    async def test_prepare_headers_remote_mcp(self, handler, sample_tokens):
        """Test preparing headers for remote MCP server."""
        server_config = ServerConfig(
            name="test-server",
            url="http://server",
            command=None,
        )

        handler._active_tokens["test-server"] = sample_tokens

        headers = await handler.prepare_server_headers(server_config)

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_prepare_headers_legacy_oauth(self, handler, sample_tokens):
        """Test preparing headers for legacy OAuth."""
        oauth_config = OAuthConfig(
            client_id="test",
            authorization_url="http://auth",
            token_url="http://token",
            redirect_uri="http://callback",
        )

        server_config = ServerConfig(
            name="test-server",
            command="test",
            oauth=oauth_config,
        )

        handler._active_tokens["test-server"] = sample_tokens

        headers = await handler.prepare_server_headers(server_config)

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_prepare_headers_with_env_headers(self, handler):
        """Test preparing headers with environment headers."""
        server_config = ServerConfig(
            name="test-server",
            command="test",
            env={
                "HTTP_HEADER_X_API_KEY": "api-key-123",
                "HTTP_HEADER_X_CUSTOM": "custom-value",
                "OTHER_VAR": "should-not-appear",
            },
        )

        headers = await handler.prepare_server_headers(server_config)

        assert headers["X-API-KEY"] == "api-key-123"
        assert headers["X-CUSTOM"] == "custom-value"
        assert "OTHER-VAR" not in headers

    @pytest.mark.asyncio
    async def test_prepare_headers_no_auth(self, handler):
        """Test preparing headers with no authentication."""
        server_config = ServerConfig(
            name="test-server",
            command="test",
        )

        headers = await handler.prepare_server_headers(server_config)

        assert headers == {}

    @pytest.mark.asyncio
    async def test_prepare_headers_mcp_auth_error(self, handler):
        """Test error handling in MCP auth."""
        server_config = ServerConfig(
            name="test-server",
            url="http://server",
            command=None,
        )

        # Mock ensure_authenticated_mcp to raise error
        with patch.object(
            handler,
            "ensure_authenticated_mcp",
            AsyncMock(side_effect=Exception("Auth failed")),
        ):
            with pytest.raises(Exception, match="Auth failed"):
                await handler.prepare_server_headers(server_config)

    @pytest.mark.asyncio
    async def test_prepare_headers_legacy_auth_error(self, handler):
        """Test error handling in legacy auth."""
        oauth_config = OAuthConfig(
            client_id="test",
            authorization_url="http://auth",
            token_url="http://token",
            redirect_uri="http://callback",
        )

        server_config = ServerConfig(
            name="test-server",
            command="test",
            oauth=oauth_config,
        )

        # Mock ensure_authenticated to raise error
        with patch.object(
            handler,
            "ensure_authenticated",
            AsyncMock(side_effect=Exception("Auth failed")),
        ):
            with pytest.raises(Exception, match="Auth failed"):
                await handler.prepare_server_headers(server_config)
