# tests/config/test_server_models.py
"""Tests for config/server_models.py module."""

import pytest
from pydantic import ValidationError

from mcp_cli.config.server_models import (
    HTTPServerConfig,
    STDIOServerConfig,
    OAuthConfig,
    UnifiedServerConfig,
    ServerConfigInput,
)


class TestHTTPServerConfig:
    """Test HTTPServerConfig model."""

    def test_valid_http_config(self):
        """Test creating valid HTTP server config."""
        config = HTTPServerConfig(
            name="test-http",
            url="http://localhost:8080",
        )
        assert config.name == "test-http"
        assert config.url == "http://localhost:8080"
        assert config.headers == {}
        assert config.disabled is False

    def test_http_config_with_https(self):
        """Test HTTP config with HTTPS URL."""
        config = HTTPServerConfig(
            name="secure-server",
            url="https://api.example.com",
        )
        assert config.url == "https://api.example.com"

    def test_http_config_with_headers(self):
        """Test HTTP config with headers."""
        config = HTTPServerConfig(
            name="with-headers",
            url="http://localhost:8080",
            headers={"Authorization": "Bearer token123"},
        )
        assert config.headers["Authorization"] == "Bearer token123"

    def test_http_config_disabled(self):
        """Test HTTP config disabled flag."""
        config = HTTPServerConfig(
            name="disabled-server",
            url="http://localhost:8080",
            disabled=True,
        )
        assert config.disabled is True

    def test_http_config_invalid_url(self):
        """Test HTTP config with invalid URL."""
        with pytest.raises(ValidationError) as exc_info:
            HTTPServerConfig(
                name="bad-url",
                url="ftp://localhost:8080",  # Not http/https
            )
        assert "http://" in str(exc_info.value) or "URL" in str(exc_info.value)

    def test_http_config_immutable(self):
        """Test HTTP config is immutable."""
        config = HTTPServerConfig(name="test", url="http://localhost:8080")
        with pytest.raises(ValidationError):
            config.name = "new-name"


class TestSTDIOServerConfig:
    """Test STDIOServerConfig model."""

    def test_valid_stdio_config(self):
        """Test creating valid STDIO server config."""
        config = STDIOServerConfig(
            name="stdio-server",
            command="python",
        )
        assert config.name == "stdio-server"
        assert config.command == "python"
        assert config.args == []
        assert config.env == {}
        assert config.disabled is False

    def test_stdio_config_with_args(self):
        """Test STDIO config with arguments."""
        config = STDIOServerConfig(
            name="python-server",
            command="python",
            args=["-m", "server"],
        )
        assert config.args == ["-m", "server"]

    def test_stdio_config_with_env(self):
        """Test STDIO config with environment variables."""
        config = STDIOServerConfig(
            name="env-server",
            command="node",
            env={"NODE_ENV": "production"},
        )
        assert config.env["NODE_ENV"] == "production"

    def test_stdio_config_empty_command(self):
        """Test STDIO config with empty command."""
        with pytest.raises(ValidationError) as exc_info:
            STDIOServerConfig(
                name="bad-server",
                command="   ",  # Empty/whitespace only
            )
        assert "empty" in str(exc_info.value).lower()

    def test_stdio_config_command_stripped(self):
        """Test STDIO config command is stripped."""
        config = STDIOServerConfig(
            name="test",
            command="  python  ",
        )
        assert config.command == "python"


class TestOAuthConfig:
    """Test OAuthConfig model."""

    def test_valid_oauth_config(self):
        """Test creating valid OAuth config."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        assert config.client_id == "test-client"
        assert config.client_secret is None
        assert config.scopes == []
        assert config.redirect_uri == "http://localhost:8080/callback"

    def test_oauth_config_with_secret(self):
        """Test OAuth config with client secret."""
        config = OAuthConfig(
            client_id="test-client",
            client_secret="secret123",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        assert config.client_secret == "secret123"

    def test_oauth_config_with_scopes(self):
        """Test OAuth config with scopes."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            scopes=["read", "write", "admin"],
        )
        assert config.scopes == ["read", "write", "admin"]

    def test_oauth_config_custom_redirect_uri(self):
        """Test OAuth config with custom redirect URI."""
        config = OAuthConfig(
            client_id="test-client",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            redirect_uri="http://localhost:9000/oauth/callback",
        )
        assert config.redirect_uri == "http://localhost:9000/oauth/callback"


class TestUnifiedServerConfig:
    """Test UnifiedServerConfig model."""

    def test_http_server_config(self):
        """Test creating HTTP server via UnifiedServerConfig."""
        config = UnifiedServerConfig(
            name="http-server",
            url="http://localhost:8080",
        )
        assert config.name == "http-server"
        assert config.url == "http://localhost:8080"
        assert config.is_http is True
        assert config.is_stdio is False

    def test_stdio_server_config(self):
        """Test creating STDIO server via UnifiedServerConfig."""
        config = UnifiedServerConfig(
            name="stdio-server",
            command="python",
            args=["-m", "server"],
        )
        assert config.name == "stdio-server"
        assert config.command == "python"
        assert config.is_stdio is True
        assert config.is_http is False

    def test_empty_name_validation(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            UnifiedServerConfig(
                name="   ",  # Empty/whitespace
                command="python",
            )
        assert "empty" in str(exc_info.value).lower()

    def test_url_format_validation(self):
        """Test URL format validation."""
        with pytest.raises(ValidationError):
            UnifiedServerConfig(
                name="bad-url-server",
                url="ftp://localhost:8080",
            )

    def test_empty_command_validation(self):
        """Test empty command validation."""
        with pytest.raises(ValidationError):
            UnifiedServerConfig(
                name="empty-cmd-server",
                command="   ",
            )

    def test_neither_url_nor_command_error(self):
        """Test error when neither url nor command provided."""
        with pytest.raises(ValidationError) as exc_info:
            UnifiedServerConfig(name="bad-server")
        assert (
            "url" in str(exc_info.value).lower()
            or "command" in str(exc_info.value).lower()
        )

    def test_both_url_and_command_error(self):
        """Test error when both url and command provided."""
        with pytest.raises(ValidationError) as exc_info:
            UnifiedServerConfig(
                name="both-server",
                url="http://localhost:8080",
                command="python",
            )
        assert "both" in str(exc_info.value).lower()

    def test_to_http_config(self):
        """Test converting to HTTPServerConfig."""
        unified = UnifiedServerConfig(
            name="http-test",
            url="http://localhost:8080",
            headers={"Auth": "token"},
            disabled=True,
        )
        http_config = unified.to_http_config()

        assert isinstance(http_config, HTTPServerConfig)
        assert http_config.name == "http-test"
        assert http_config.url == "http://localhost:8080"
        assert http_config.headers["Auth"] == "token"
        assert http_config.disabled is True

    def test_to_http_config_error_for_stdio(self):
        """Test to_http_config raises error for STDIO server."""
        unified = UnifiedServerConfig(
            name="stdio-test",
            command="python",
        )
        with pytest.raises(ValueError) as exc_info:
            unified.to_http_config()
        assert "not an HTTP server" in str(exc_info.value)

    def test_to_stdio_config(self):
        """Test converting to STDIOServerConfig."""
        unified = UnifiedServerConfig(
            name="stdio-test",
            command="python",
            args=["-m", "server"],
            env={"KEY": "value"},
            disabled=True,
        )
        stdio_config = unified.to_stdio_config()

        assert isinstance(stdio_config, STDIOServerConfig)
        assert stdio_config.name == "stdio-test"
        assert stdio_config.command == "python"
        assert stdio_config.args == ["-m", "server"]
        assert stdio_config.env["KEY"] == "value"
        assert stdio_config.disabled is True

    def test_to_stdio_config_error_for_http(self):
        """Test to_stdio_config raises error for HTTP server."""
        unified = UnifiedServerConfig(
            name="http-test",
            url="http://localhost:8080",
        )
        with pytest.raises(ValueError) as exc_info:
            unified.to_stdio_config()
        assert "not a STDIO server" in str(exc_info.value)

    def test_unified_config_with_oauth(self):
        """Test UnifiedServerConfig with OAuth."""
        oauth = OAuthConfig(
            client_id="test",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        config = UnifiedServerConfig(
            name="oauth-server",
            url="http://localhost:8080",
            oauth=oauth,
        )
        assert config.oauth is not None
        assert config.oauth.client_id == "test"


class TestServerConfigInput:
    """Test ServerConfigInput model."""

    def test_basic_stdio_input(self):
        """Test basic STDIO server input."""
        input_config = ServerConfigInput(
            command="python",
            args=["-m", "server"],
        )
        unified = input_config.to_unified("test-server")

        assert unified.name == "test-server"
        assert unified.command == "python"
        assert unified.args == ["-m", "server"]

    def test_basic_http_input(self):
        """Test basic HTTP server input."""
        input_config = ServerConfigInput(
            url="http://localhost:8080",
        )
        unified = input_config.to_unified("http-server")

        assert unified.name == "http-server"
        assert unified.url == "http://localhost:8080"

    def test_input_with_dict_oauth(self):
        """Test input with OAuth as dict."""
        input_config = ServerConfigInput(
            url="http://localhost:8080",
            oauth={
                "client_id": "test-client",
                "authorization_url": "https://auth.example.com/authorize",
                "token_url": "https://auth.example.com/token",
            },
        )
        unified = input_config.to_unified("oauth-server")

        assert unified.oauth is not None
        assert unified.oauth.client_id == "test-client"

    def test_input_with_oauth_object(self):
        """Test input with OAuth as OAuthConfig object."""
        oauth = OAuthConfig(
            client_id="test-client",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        input_config = ServerConfigInput(
            url="http://localhost:8080",
            oauth=oauth,
        )
        unified = input_config.to_unified("oauth-server")

        assert unified.oauth is not None
        assert unified.oauth.client_id == "test-client"

    def test_input_with_env_and_headers(self):
        """Test input with env vars and headers."""
        input_config = ServerConfigInput(
            command="python",
            env={"KEY": "value"},
        )
        unified = input_config.to_unified("env-server")

        assert unified.env["KEY"] == "value"

    def test_input_ignores_extra_fields(self):
        """Test that extra fields are ignored."""
        # This tests the extra="ignore" config
        input_config = ServerConfigInput.model_validate(
            {
                "command": "python",
                "extra_field": "ignored",
                "another_extra": 123,
            }
        )
        unified = input_config.to_unified("test")
        assert unified.command == "python"

    def test_input_disabled_flag(self):
        """Test disabled flag in input."""
        input_config = ServerConfigInput(
            command="python",
            disabled=True,
        )
        unified = input_config.to_unified("disabled-server")
        assert unified.disabled is True
