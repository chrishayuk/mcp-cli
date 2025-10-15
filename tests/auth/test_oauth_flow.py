"""Tests for OAuthFlow."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_cli.auth.oauth_config import OAuthConfig
from mcp_cli.auth.oauth_flow import OAuthFlow


class TestOAuthFlowInit:
    """Test OAuthFlow initialization."""

    def test_init(self):
        """Test initialization."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )

        flow = OAuthFlow(config)

        assert flow.config == config
        assert flow._auth_result is None
        assert flow._code_verifier is None


class TestOAuthFlowPKCE:
    """Test PKCE generation."""

    def test_generate_pkce_pair(self):
        """Test PKCE pair generation."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        verifier, challenge = flow._generate_pkce_pair()

        assert len(verifier) > 0
        assert len(challenge) > 0
        assert verifier != challenge
        assert "=" not in verifier
        assert "=" not in challenge


class TestOAuthFlowAuthorizationURL:
    """Test authorization URL generation."""

    def test_get_authorization_url_basic(self):
        """Test basic authorization URL generation."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        url = flow.get_authorization_url()

        assert url.startswith("https://example.com/authorize?")
        assert "client_id=test-client" in url
        assert "response_type=code" in url
        assert "redirect_uri=http" in url

    def test_get_authorization_url_with_scopes(self):
        """Test authorization URL with scopes."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
            scopes=["read", "write"],
        )
        flow = OAuthFlow(config)

        url = flow.get_authorization_url()

        assert "scope=read+write" in url

    def test_get_authorization_url_with_pkce(self):
        """Test authorization URL with PKCE."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
            use_pkce=True,
        )
        flow = OAuthFlow(config)

        url = flow.get_authorization_url()

        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert flow._code_verifier is not None

    def test_get_authorization_url_with_extra_params(self):
        """Test authorization URL with extra parameters."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
            extra_auth_params={"prompt": "consent", "access_type": "offline"},
        )
        flow = OAuthFlow(config)

        url = flow.get_authorization_url()

        assert "prompt=consent" in url
        assert "access_type=offline" in url


class TestOAuthFlowTokenExchange:
    """Test token exchange."""

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_success(self):
        """Test successful token exchange."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            tokens = await flow.exchange_code_for_token("test-code")

            assert tokens.access_token == "test-access-token"
            assert tokens.token_type == "Bearer"

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://example.com/token"
            assert call_args[1]["data"]["grant_type"] == "authorization_code"
            assert call_args[1]["data"]["code"] == "test-code"

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_with_secret(self):
        """Test token exchange with client secret."""
        config = OAuthConfig(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            await flow.exchange_code_for_token("test-code")

            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["client_secret"] == "test-secret"

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_with_pkce(self):
        """Test token exchange with PKCE."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
            use_pkce=True,
        )
        flow = OAuthFlow(config)
        flow._code_verifier = "test-verifier"

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            await flow.exchange_code_for_token("test-code")

            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["code_verifier"] == "test-verifier"

    @pytest.mark.asyncio
    async def test_exchange_code_for_token_failure(self):
        """Test token exchange failure."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid code"
            mock_client.post = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception, match="Token exchange failed"):
                await flow.exchange_code_for_token("bad-code")


class TestOAuthFlowRefreshToken:
    """Test token refresh."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Test successful token refresh."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        token_response = {
            "access_token": "new-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            tokens = await flow.refresh_token("test-refresh-token")

            assert tokens.access_token == "new-access-token"

            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["grant_type"] == "refresh_token"
            assert call_args[1]["data"]["refresh_token"] == "test-refresh-token"

    @pytest.mark.asyncio
    async def test_refresh_token_with_secret(self):
        """Test token refresh with client secret."""
        config = OAuthConfig(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        token_response = {
            "access_token": "new-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            await flow.refresh_token("test-refresh-token")

            call_args = mock_client.post.call_args
            assert call_args[1]["data"]["client_secret"] == "test-secret"

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self):
        """Test token refresh failure."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid refresh token"
            mock_client.post = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception, match="Token refresh failed"):
                await flow.refresh_token("bad-refresh-token")


class TestOAuthFlowCallbackHandler:
    """Test callback handler creation."""

    def test_create_callback_handler(self):
        """Test callback handler class creation."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        handler_class = flow._create_callback_handler()

        assert handler_class is not None
        assert hasattr(handler_class, "do_GET")
        assert hasattr(handler_class, "log_message")


class TestOAuthFlowCallbackHandlerDetailed:
    """Test callback handler in detail."""

    def test_callback_handler_with_error(self):
        """Test callback handler receiving error."""
        from io import BytesIO

        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)
        handler_class = flow._create_callback_handler()

        # Create mock request with error
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /?error=access_denied&error_description=User+denied HTTP/1.1\r\nHost: localhost\r\n\r\n"
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
        assert flow._auth_result is not None
        assert "error" in flow._auth_result

    def test_callback_handler_with_code(self):
        """Test callback handler receiving authorization code."""
        from io import BytesIO

        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)
        handler_class = flow._create_callback_handler()

        # Create mock request with code
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /?code=test-auth-code&state=test-state HTTP/1.1\r\nHost: localhost\r\n\r\n"
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

        # Check that code was captured
        assert flow._auth_result is not None
        assert "code" in flow._auth_result
        assert flow._auth_result["code"] == "test-auth-code"

    def test_callback_handler_no_code_no_error(self):
        """Test callback handler with neither code nor error."""
        from io import BytesIO

        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)
        handler_class = flow._create_callback_handler()

        # Create mock request without code or error
        class MockSocket:
            def makefile(self, mode, buffsize=-1):
                if "r" in mode:
                    return BytesIO(
                        b"GET /?state=test-state HTTP/1.1\r\nHost: localhost\r\n\r\n"
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

        # Should set error result
        assert flow._auth_result is not None
        assert "error" in flow._auth_result


class TestOAuthFlowCallbackServer:
    """Test callback server functionality."""

    @pytest.mark.asyncio
    async def test_run_callback_server_receives_result(self):
        """Test callback server receives authorization result."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        # Simulate receiving result
        async def simulate_callback():
            await asyncio.sleep(0.1)
            flow._auth_result = {"code": "test-code"}

        callback_task = asyncio.create_task(simulate_callback())
        server_task = asyncio.create_task(flow._run_callback_server(18890))

        await asyncio.gather(callback_task, server_task)

        assert flow._auth_result is not None

    @pytest.mark.asyncio
    async def test_run_callback_server_timeout(self):
        """Test callback server timeout handling."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        # Mock asyncio.sleep to simulate quick timeout
        call_count = {"count": 0}
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            call_count["count"] += 1
            if call_count["count"] > 5:
                flow._auth_result = {}  # Exit condition
            await original_sleep(0.01)

        with patch("asyncio.sleep", mock_sleep):
            await flow._run_callback_server(18891)

        assert call_count["count"] > 0


class TestOAuthFlowAuthorize:
    """Test full authorization flow."""

    @pytest.mark.asyncio
    async def test_authorize_success(self):
        """Test successful authorization."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            with patch("webbrowser.open"):

                async def mock_run_callback_server(port):
                    await asyncio.sleep(0.1)
                    flow._auth_result = {"code": "test-code"}

                with patch.object(
                    flow, "_run_callback_server", mock_run_callback_server
                ):
                    tokens = await flow.authorize()

                    assert tokens.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_authorize_timeout(self):
        """Test authorization timeout."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        with patch("webbrowser.open"):

            async def mock_run_callback_server(port):
                await asyncio.sleep(0.1)
                # Don't set _auth_result - simulate timeout

            with patch.object(flow, "_run_callback_server", mock_run_callback_server):
                with pytest.raises(Exception, match="Authorization timed out"):
                    await flow.authorize()

    @pytest.mark.asyncio
    async def test_authorize_error(self):
        """Test authorization with error."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        with patch("webbrowser.open"):

            async def mock_run_callback_server(port):
                await asyncio.sleep(0.1)
                flow._auth_result = {"error": "access_denied"}

            with patch.object(flow, "_run_callback_server", mock_run_callback_server):
                with pytest.raises(Exception, match="Authorization failed"):
                    await flow.authorize()

    @pytest.mark.asyncio
    async def test_authorize_callback_server_exception(self):
        """Test authorization with callback server exception."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:8080/callback",
        )
        flow = OAuthFlow(config)

        with patch("webbrowser.open"):

            async def mock_run_callback_server(port):
                raise Exception("Server error")

            with patch.object(flow, "_run_callback_server", mock_run_callback_server):
                with pytest.raises(Exception, match="Callback server error"):
                    await flow.authorize()

    @pytest.mark.asyncio
    async def test_authorize_custom_port(self):
        """Test authorization with custom redirect URI port."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            redirect_uri="http://localhost:9000/callback",
        )
        flow = OAuthFlow(config)

        token_response = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = token_response
            mock_client.post = AsyncMock(return_value=mock_response)

            with patch("webbrowser.open"):

                async def mock_run_callback_server(port):
                    assert port == 9000  # Should use custom port
                    await asyncio.sleep(0.1)
                    flow._auth_result = {"code": "test-code"}

                with patch.object(
                    flow, "_run_callback_server", mock_run_callback_server
                ):
                    await flow.authorize()
