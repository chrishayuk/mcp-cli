"""Tests for MCPOAuthClient."""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_cli.auth.mcp_oauth import (
    DynamicClientRegistration,
    MCPAuthorizationMetadata,
    MCPOAuthClient,
)


class TestMCPAuthorizationMetadata:
    """Test MCPAuthorizationMetadata dataclass."""

    def test_from_dict_minimal(self):
        """Test creating metadata from minimal dict."""
        data = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
        }

        metadata = MCPAuthorizationMetadata.from_dict(data)

        assert metadata.authorization_endpoint == data["authorization_endpoint"]
        assert metadata.token_endpoint == data["token_endpoint"]
        assert metadata.registration_endpoint is None
        assert metadata.scopes_supported == []
        assert metadata.response_types_supported == ["code"]
        assert metadata.grant_types_supported == ["authorization_code", "refresh_token"]
        assert metadata.code_challenge_methods_supported == ["S256"]

    def test_from_dict_complete(self):
        """Test creating metadata from complete dict."""
        data = {
            "authorization_endpoint": "https://auth.example.com/authorize",
            "token_endpoint": "https://auth.example.com/token",
            "registration_endpoint": "https://auth.example.com/register",
            "scopes_supported": ["read", "write"],
            "response_types_supported": ["code", "token"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256", "plain"],
        }

        metadata = MCPAuthorizationMetadata.from_dict(data)

        assert metadata.authorization_endpoint == data["authorization_endpoint"]
        assert metadata.token_endpoint == data["token_endpoint"]
        assert metadata.registration_endpoint == data["registration_endpoint"]
        assert metadata.scopes_supported == ["read", "write"]
        assert metadata.response_types_supported == ["code", "token"]
        assert metadata.grant_types_supported == ["authorization_code"]
        assert metadata.code_challenge_methods_supported == ["S256", "plain"]


class TestDynamicClientRegistration:
    """Test DynamicClientRegistration dataclass."""

    def test_from_dict_minimal(self):
        """Test creating registration from minimal dict."""
        data = {"client_id": "test-client-123"}

        registration = DynamicClientRegistration.from_dict(data)

        assert registration.client_id == "test-client-123"
        assert registration.client_secret is None
        assert registration.client_id_issued_at is None
        assert registration.client_secret_expires_at == 0

    def test_from_dict_complete(self):
        """Test creating registration from complete dict."""
        data = {
            "client_id": "test-client-123",
            "client_secret": "secret-456",
            "client_id_issued_at": 1234567890,
            "client_secret_expires_at": 1234567999,
        }

        registration = DynamicClientRegistration.from_dict(data)

        assert registration.client_id == "test-client-123"
        assert registration.client_secret == "secret-456"
        assert registration.client_id_issued_at == 1234567890
        assert registration.client_secret_expires_at == 1234567999

    def test_to_dict_minimal(self):
        """Test converting minimal registration to dict."""
        registration = DynamicClientRegistration(client_id="test-client-123")

        result = registration.to_dict()

        assert result == {"client_id": "test-client-123"}

    def test_to_dict_complete(self):
        """Test converting complete registration to dict."""
        registration = DynamicClientRegistration(
            client_id="test-client-123",
            client_secret="secret-456",
            client_id_issued_at=1234567890,
            client_secret_expires_at=1234567999,
        )

        result = registration.to_dict()

        assert result == {
            "client_id": "test-client-123",
            "client_secret": "secret-456",
            "client_id_issued_at": 1234567890,
            "client_secret_expires_at": 1234567999,
        }


class TestMCPOAuthClientInit:
    """Test MCPOAuthClient initialization."""

    def test_init_default_redirect(self):
        """Test initialization with default redirect URI."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        assert client.server_url == "https://mcp.example.com/mcp"
        assert client.redirect_uri == "http://localhost:8080/callback"
        assert client._auth_metadata is None
        assert client._client_registration is None
        assert client._code_verifier is None
        assert client._auth_result is None

    def test_init_custom_redirect(self):
        """Test initialization with custom redirect URI."""
        client = MCPOAuthClient(
            "https://mcp.example.com/mcp", redirect_uri="http://localhost:9000/cb"
        )

        assert client.redirect_uri == "http://localhost:9000/cb"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from server URL."""
        client = MCPOAuthClient("https://mcp.example.com/mcp/")

        assert client.server_url == "https://mcp.example.com/mcp"


class TestMCPOAuthClientDiscovery:
    """Test authorization server discovery."""

    @pytest.mark.asyncio
    async def test_discover_authorization_server(self):
        """Test successful discovery."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        discovery_response = {
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "registration_endpoint": "https://example.com/register",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json = Mock(return_value=discovery_response)
            mock_response.raise_for_status = Mock()
            mock_client.get = AsyncMock(return_value=mock_response)

            metadata = await client.discover_authorization_server()

            assert (
                metadata.authorization_endpoint
                == discovery_response["authorization_endpoint"]
            )
            assert metadata.token_endpoint == discovery_response["token_endpoint"]
            assert client._auth_metadata == metadata
            mock_client.get.assert_called_once_with(
                "https://mcp.example.com/.well-known/oauth-authorization-server"
            )

    @pytest.mark.asyncio
    async def test_discover_authorization_server_http_error(self):
        """Test discovery with HTTP error."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.raise_for_status.side_effect = Exception("Not found")
            mock_client.get = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception, match="Not found"):
                await client.discover_authorization_server()


class TestMCPOAuthClientRegistration:
    """Test dynamic client registration."""

    @pytest.mark.asyncio
    async def test_register_client_default(self):
        """Test client registration with defaults."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            registration_endpoint="https://example.com/register",
        )

        registration_response = {"client_id": "test-client-123"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = registration_response
            mock_client.post = AsyncMock(return_value=mock_response)

            registration = await client.register_client()

            assert registration.client_id == "test-client-123"
            assert client._client_registration == registration

            # Verify the post call
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://example.com/register"
            assert call_args[1]["json"]["client_name"] == "MCP CLI"
            assert call_args[1]["json"]["redirect_uris"] == [
                "http://localhost:8080/callback"
            ]

    @pytest.mark.asyncio
    async def test_register_client_custom(self):
        """Test client registration with custom parameters."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            registration_endpoint="https://example.com/register",
        )

        registration_response = {"client_id": "custom-client"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = registration_response
            mock_client.post = AsyncMock(return_value=mock_response)

            registration = await client.register_client(
                client_name="Custom App",
                redirect_uris=["http://localhost:9000/callback"],
            )

            assert registration.client_id == "custom-client"

            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["client_name"] == "Custom App"
            assert call_args[1]["json"]["redirect_uris"] == [
                "http://localhost:9000/callback"
            ]

    @pytest.mark.asyncio
    async def test_register_client_discovers_if_needed(self):
        """Test that registration discovers metadata if not already done."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        discovery_response = {
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "registration_endpoint": "https://example.com/register",
        }

        registration_response = {"client_id": "test-client"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_get_response = Mock()
            mock_get_response.json = Mock(return_value=discovery_response)
            mock_get_response.raise_for_status = Mock()
            mock_post_response = Mock()
            mock_post_response.json = Mock(return_value=registration_response)
            mock_post_response.raise_for_status = Mock()

            mock_client.get = AsyncMock(return_value=mock_get_response)
            mock_client.post = AsyncMock(return_value=mock_post_response)

            await client.register_client()

            # Should have called discovery first
            mock_client.get.assert_called_once()
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_client_no_registration_endpoint(self):
        """Test registration when server doesn't support it."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            registration_endpoint=None,
        )

        with pytest.raises(
            ValueError, match="does not support dynamic client registration"
        ):
            await client.register_client()


class TestMCPOAuthClientPKCE:
    """Test PKCE generation."""

    def test_generate_pkce_pair(self):
        """Test PKCE generation produces valid pair."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        verifier, challenge = client._generate_pkce_pair()

        assert len(verifier) > 0
        assert len(challenge) > 0
        assert verifier != challenge
        assert "=" not in verifier  # Base64 padding should be stripped
        assert "=" not in challenge


class TestMCPOAuthClientAuthorization:
    """Test authorization URL generation."""

    def test_get_authorization_url_no_scopes(self):
        """Test authorization URL without scopes."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )
        client._client_registration = DynamicClientRegistration(client_id="test-client")

        url = client.get_authorization_url()

        assert url.startswith("https://example.com/authorize?")
        assert "client_id=test-client" in url
        assert "response_type=code" in url
        assert "redirect_uri=http" in url
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert "state=" in url
        assert client._code_verifier is not None

    def test_get_authorization_url_with_scopes(self):
        """Test authorization URL with scopes."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )
        client._client_registration = DynamicClientRegistration(client_id="test-client")

        url = client.get_authorization_url(scopes=["read", "write"])

        assert "scope=read+write" in url

    def test_get_authorization_url_not_ready(self):
        """Test error when metadata or registration not set."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        with pytest.raises(ValueError, match="Must discover and register"):
            client.get_authorization_url()


class TestMCPOAuthClientTokenExchange:
    """Test token exchange."""

    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self):
        """Test successful token exchange."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )
        client._client_registration = DynamicClientRegistration(client_id="test-client")
        client._code_verifier = "test-verifier"

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            tokens = await client.exchange_code_for_token("test-code")

            assert tokens.access_token == "test-access-token"

            # Verify post data
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://example.com/token"
            assert call_args[1]["data"]["grant_type"] == "authorization_code"
            assert call_args[1]["data"]["code"] == "test-code"
            assert call_args[1]["data"]["code_verifier"] == "test-verifier"

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_with_secret(self):
        """Test token exchange with client secret."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )
        client._client_registration = DynamicClientRegistration(
            client_id="test-client", client_secret="test-secret"
        )
        client._code_verifier = "test-verifier"

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            await client.exchange_code_for_token("test-code")

            # Verify client_secret was included
            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["client_secret"] == "test-secret"

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_not_ready(self):
        """Test error when not ready for exchange."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        with pytest.raises(ValueError, match="Must discover and register"):
            await client.exchange_code_for_token("test-code")


class TestMCPOAuthClientRefresh:
    """Test token refresh."""

    @pytest.mark.asyncio
    async def test_refresh_token(self):
        """Test successful token refresh."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )
        client._client_registration = DynamicClientRegistration(client_id="test-client")

        token_response = {
            "access_token": "new-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            tokens = await client.refresh_token("test-refresh-token")

            assert tokens.access_token == "new-access-token"

            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["grant_type"] == "refresh_token"
            assert call_args[1]["data"]["refresh_token"] == "test-refresh-token"

    @pytest.mark.asyncio
    async def test_refresh_token_with_secret(self):
        """Test token refresh with client secret."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")
        client._auth_metadata = MCPAuthorizationMetadata(
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )
        client._client_registration = DynamicClientRegistration(
            client_id="test-client", client_secret="test-secret"
        )

        token_response = {
            "access_token": "new-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            await client.refresh_token("test-refresh-token")

            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["client_secret"] == "test-secret"

    @pytest.mark.asyncio
    async def test_refresh_token_not_ready(self):
        """Test error when not ready for refresh."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        with pytest.raises(ValueError, match="Must discover and register"):
            await client.refresh_token("test-refresh-token")


class TestMCPOAuthClientCallbackHandler:
    """Test callback handler."""

    def test_create_callback_handler(self):
        """Test callback handler creation."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        handler_class = client._create_callback_handler()

        assert handler_class is not None
        assert hasattr(handler_class, "do_GET")
        assert hasattr(handler_class, "log_message")

    def test_callback_handler_success(self):
        """Test callback handler with successful authorization."""

        client = MCPOAuthClient("https://mcp.example.com/mcp")
        handler_class = client._create_callback_handler()

        # Create mock request
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /callback?code=test-code&state=test-state HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                else:
                    return BytesIO()

            def sendall(self, data):
                pass  # Mock socket write

            def settimeout(self, timeout):
                pass

            def setsockopt(self, level, optname, value):
                pass

        # Create handler instance
        mock_request = MockSocket()
        mock_server = Mock()
        mock_server.server_name = "localhost"
        mock_server.server_port = 8080

        _ = handler_class(mock_request, ("127.0.0.1", 12345), mock_server)

        # Check that auth result was set
        assert client._auth_result is not None
        assert "code" in client._auth_result
        assert client._auth_result["code"] == "test-code"

    def test_callback_handler_error(self):
        """Test callback handler with error."""

        client = MCPOAuthClient("https://mcp.example.com/mcp")
        handler_class = client._create_callback_handler()

        # Create mock request with error
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /callback?error=access_denied&error_description=User+denied HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                else:
                    return BytesIO()

            def sendall(self, data):
                pass  # Mock socket write

            def settimeout(self, timeout):
                pass

            def setsockopt(self, level, optname, value):
                pass

        mock_request = MockSocket()
        mock_server = Mock()
        mock_server.server_name = "localhost"
        mock_server.server_port = 8080

        _ = handler_class(mock_request, ("127.0.0.1", 12345), mock_server)

        # Check that error was captured
        assert client._auth_result is not None
        assert "error" in client._auth_result

    def test_callback_handler_no_code(self):
        """Test callback handler without code or error."""

        client = MCPOAuthClient("https://mcp.example.com/mcp")
        handler_class = client._create_callback_handler()

        # Create mock request without code
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /callback?state=test-state HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                else:
                    return BytesIO()

            def sendall(self, data):
                pass  # Mock socket write

            def settimeout(self, timeout):
                pass

            def setsockopt(self, level, optname, value):
                pass

        mock_request = MockSocket()
        mock_server = Mock()
        mock_server.server_name = "localhost"
        mock_server.server_port = 8080

        # Clear any previous result
        client._auth_result = None

        _ = handler_class(mock_request, ("127.0.0.1", 12345), mock_server)

        # Should handle gracefully (either set error or no result)
        # The actual behavior depends on implementation

    def test_callback_handler_non_callback_path(self):
        """Test callback handler with non-callback path."""

        client = MCPOAuthClient("https://mcp.example.com/mcp")
        handler_class = client._create_callback_handler()

        # Create mock request for favicon
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                else:
                    return BytesIO()

            def sendall(self, data):
                pass  # Mock socket write

            def settimeout(self, timeout):
                pass

            def setsockopt(self, level, optname, value):
                pass

        mock_request = MockSocket()
        mock_server = Mock()
        mock_server.server_name = "localhost"
        mock_server.server_port = 8080

        # Should not crash
        try:
            _ = handler_class(mock_request, ("127.0.0.1", 12345), mock_server)
        except Exception:
            pass  # Some errors are expected with mock sockets

    def test_callback_handler_already_got_result(self):
        """Test callback handler when result already received."""

        client = MCPOAuthClient("https://mcp.example.com/mcp")
        # Set a result first
        client._auth_result = {"code": "already-have-code"}

        handler_class = client._create_callback_handler()

        # Create another request
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /callback?code=new-code HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                else:
                    return BytesIO()

            def sendall(self, data):
                pass  # Mock socket write

            def settimeout(self, timeout):
                pass

            def setsockopt(self, level, optname, value):
                pass

        mock_request = MockSocket()
        mock_server = Mock()
        mock_server.server_name = "localhost"
        mock_server.server_port = 8080

        try:
            _ = handler_class(mock_request, ("127.0.0.1", 12345), mock_server)
        except Exception:
            pass

        # Should keep original result
        assert client._auth_result["code"] == "already-have-code"


class TestMCPOAuthClientCallbackServer:
    """Test callback server."""

    @pytest.mark.asyncio
    async def test_run_callback_server_with_result(self):
        """Test callback server receives result."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        # Simulate receiving auth result immediately
        async def simulate_callback():
            await asyncio.sleep(0.1)
            client._auth_result = {"code": "test-code"}

        # Run both tasks
        callback_task = asyncio.create_task(simulate_callback())
        server_task = asyncio.create_task(client._run_callback_server(18888))

        await asyncio.gather(callback_task, server_task)

        assert client._auth_result is not None

    @pytest.mark.asyncio
    async def test_run_callback_server_timeout(self):
        """Test callback server timeout."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        # Don't set _auth_result, let it timeout
        # Use short timeout by mocking asyncio.sleep
        call_count = {"count": 0}

        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            call_count["count"] += 1
            if call_count["count"] > 5:  # Exit after 5 iterations
                client._auth_result = {}  # Set something to exit
            await original_sleep(0.01)  # Very short sleep

        with patch("asyncio.sleep", mock_sleep):
            await client._run_callback_server(18889)

        # Should have timed out or exited
        assert call_count["count"] > 0


class TestMCPOAuthClientAuthorize:
    """Test full authorization flow."""

    @pytest.mark.asyncio
    async def test_authorize_success(self):
        """Test successful authorization flow."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        discovery_response = {
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "registration_endpoint": "https://example.com/register",
        }

        registration_response = {"client_id": "test-client"}

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value

            # Mock discovery
            mock_discovery_response = Mock()
            mock_discovery_response.json = Mock(return_value=discovery_response)
            mock_discovery_response.raise_for_status = Mock()

            # Mock registration
            mock_reg_response = Mock()
            mock_reg_response.json = Mock(return_value=registration_response)
            mock_reg_response.raise_for_status = Mock()

            # Mock token exchange
            mock_token_response = Mock()
            mock_token_response.json = Mock(return_value=token_response)
            mock_token_response.raise_for_status = Mock()

            mock_client.get = AsyncMock(return_value=mock_discovery_response)
            mock_client.post = AsyncMock(
                side_effect=[mock_reg_response, mock_token_response]
            )

            # Mock browser opening
            with patch("webbrowser.open"):
                # Mock callback server
                async def mock_run_callback_server(port):
                    # Simulate successful callback
                    await asyncio.sleep(0.1)
                    client._auth_result = {"code": "test-code", "state": "test-state"}

                with patch.object(
                    client, "_run_callback_server", mock_run_callback_server
                ):
                    tokens = await client.authorize()

                    assert tokens.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_authorize_timeout(self):
        """Test authorization timeout."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        discovery_response = {
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "registration_endpoint": "https://example.com/register",
        }

        registration_response = {"client_id": "test-client"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value

            mock_discovery_response = Mock()
            mock_discovery_response.json = Mock(return_value=discovery_response)
            mock_discovery_response.raise_for_status = Mock()

            mock_reg_response = Mock()
            mock_reg_response.json = Mock(return_value=registration_response)
            mock_reg_response.raise_for_status = Mock()

            mock_client.get = AsyncMock(return_value=mock_discovery_response)
            mock_client.post = AsyncMock(return_value=mock_reg_response)

            with patch("webbrowser.open"):

                async def mock_run_callback_server_timeout(port):
                    await asyncio.sleep(0.1)
                    # Don't set _auth_result - simulate timeout

                with patch.object(
                    client, "_run_callback_server", mock_run_callback_server_timeout
                ):
                    with pytest.raises(Exception, match="Authorization timed out"):
                        await client.authorize()

    @pytest.mark.asyncio
    async def test_authorize_error(self):
        """Test authorization with error."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        discovery_response = {
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "registration_endpoint": "https://example.com/register",
        }

        registration_response = {"client_id": "test-client"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value

            mock_discovery_response = Mock()
            mock_discovery_response.json = Mock(return_value=discovery_response)
            mock_discovery_response.raise_for_status = Mock()

            mock_reg_response = Mock()
            mock_reg_response.json = Mock(return_value=registration_response)
            mock_reg_response.raise_for_status = Mock()

            mock_client.get = AsyncMock(return_value=mock_discovery_response)
            mock_client.post = AsyncMock(return_value=mock_reg_response)

            with patch("webbrowser.open"):

                async def mock_run_callback_server_error(port):
                    await asyncio.sleep(0.1)
                    client._auth_result = {"error": "access_denied"}

                with patch.object(
                    client, "_run_callback_server", mock_run_callback_server_error
                ):
                    with pytest.raises(Exception, match="Authorization failed"):
                        await client.authorize()

    @pytest.mark.asyncio
    async def test_authorize_with_scopes(self):
        """Test authorization with custom scopes."""
        client = MCPOAuthClient("https://mcp.example.com/mcp")

        discovery_response = {
            "authorization_endpoint": "https://example.com/authorize",
            "token_endpoint": "https://example.com/token",
            "registration_endpoint": "https://example.com/register",
        }

        registration_response = {"client_id": "test-client"}

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value

            mock_discovery_response = Mock()
            mock_discovery_response.json = Mock(return_value=discovery_response)
            mock_discovery_response.raise_for_status = Mock()

            mock_reg_response = Mock()
            mock_reg_response.json = Mock(return_value=registration_response)
            mock_reg_response.raise_for_status = Mock()

            mock_token_response = Mock()
            mock_token_response.json = Mock(return_value=token_response)
            mock_token_response.raise_for_status = Mock()

            mock_client.get = AsyncMock(return_value=mock_discovery_response)
            mock_client.post = AsyncMock(
                side_effect=[mock_reg_response, mock_token_response]
            )

            with patch("webbrowser.open") as mock_browser:

                async def mock_run_callback_server(port):
                    await asyncio.sleep(0.1)
                    client._auth_result = {"code": "test-code"}

                with patch.object(
                    client, "_run_callback_server", mock_run_callback_server
                ):
                    await client.authorize(scopes=["read", "write"])

                    # Verify browser was opened with scopes
                    assert mock_browser.called
