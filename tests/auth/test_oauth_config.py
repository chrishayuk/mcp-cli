"""Tests for OAuth configuration models."""

import time
from mcp_cli.auth.oauth_config import OAuthConfig, OAuthTokens


class TestOAuthConfig:
    """Test OAuthConfig model."""

    def test_oauth_config_creation(self):
        """Test creating OAuthConfig."""
        config = OAuthConfig(
            authorization_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            client_id="test-client-id",
        )

        assert config.authorization_url == "https://example.com/oauth/authorize"
        assert config.token_url == "https://example.com/oauth/token"
        assert config.client_id == "test-client-id"
        assert config.client_secret is None
        assert config.scopes == []
        assert config.redirect_uri == "http://localhost:8080/callback"
        assert config.use_pkce is True
        assert config.extra_auth_params == {}

    def test_oauth_config_with_all_fields(self):
        """Test OAuthConfig with all fields."""
        config = OAuthConfig(
            authorization_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            client_id="test-client-id",
            client_secret="test-secret",
            scopes=["read", "write"],
            redirect_uri="http://localhost:3000/callback",
            use_pkce=False,
            extra_auth_params={"audience": "https://api.example.com"},
        )

        assert config.client_secret == "test-secret"
        assert config.scopes == ["read", "write"]
        assert config.redirect_uri == "http://localhost:3000/callback"
        assert config.use_pkce is False
        assert config.extra_auth_params == {"audience": "https://api.example.com"}

    def test_oauth_config_from_dict(self):
        """Test creating OAuthConfig from dictionary."""
        data = {
            "authorization_url": "https://example.com/oauth/authorize",
            "token_url": "https://example.com/oauth/token",
            "client_id": "test-client-id",
            "client_secret": "test-secret",
            "scopes": ["read", "write"],
            "redirect_uri": "http://localhost:3000/callback",
            "use_pkce": False,
            "extra_auth_params": {"audience": "https://api.example.com"},
        }

        config = OAuthConfig.model_validate(data)

        assert config.authorization_url == data["authorization_url"]
        assert config.token_url == data["token_url"]
        assert config.client_id == data["client_id"]
        assert config.client_secret == data["client_secret"]
        assert config.scopes == data["scopes"]
        assert config.redirect_uri == data["redirect_uri"]
        assert config.use_pkce == data["use_pkce"]
        assert config.extra_auth_params == data["extra_auth_params"]

    def test_oauth_config_from_dict_minimal(self):
        """Test creating OAuthConfig from minimal dictionary."""
        data = {
            "authorization_url": "https://example.com/oauth/authorize",
            "token_url": "https://example.com/oauth/token",
            "client_id": "test-client-id",
        }

        config = OAuthConfig.model_validate(data)

        assert config.client_secret is None
        assert config.scopes == []
        assert config.redirect_uri == "http://localhost:8080/callback"
        assert config.use_pkce is True
        assert config.extra_auth_params == {}

    def test_oauth_config_to_dict(self):
        """Test converting OAuthConfig to dictionary."""
        config = OAuthConfig(
            authorization_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            client_id="test-client-id",
            client_secret="test-secret",
            scopes=["read", "write"],
            redirect_uri="http://localhost:3000/callback",
            use_pkce=False,
            extra_auth_params={"audience": "https://api.example.com"},
        )

        data = config.model_dump()

        assert data["authorization_url"] == config.authorization_url
        assert data["token_url"] == config.token_url
        assert data["client_id"] == config.client_id
        assert data["client_secret"] == config.client_secret
        assert data["scopes"] == config.scopes
        assert data["redirect_uri"] == config.redirect_uri
        assert data["use_pkce"] == config.use_pkce
        assert data["extra_auth_params"] == config.extra_auth_params

    def test_oauth_config_to_dict_without_optional_fields(self):
        """Test converting OAuthConfig to dictionary without optional fields."""
        config = OAuthConfig(
            authorization_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            client_id="test-client-id",
        )

        data = config.model_dump(exclude_none=True, exclude_defaults=True)

        assert "client_secret" not in data
        assert "extra_auth_params" not in data
        assert "scopes" not in data or data["scopes"] == []

    def test_oauth_config_round_trip(self):
        """Test round-trip conversion to/from dictionary."""
        original = OAuthConfig(
            authorization_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            client_id="test-client-id",
            client_secret="test-secret",
            scopes=["read", "write"],
            extra_auth_params={"key": "value"},
        )

        data = original.model_dump()
        restored = OAuthConfig.model_validate(data)

        assert restored.authorization_url == original.authorization_url
        assert restored.token_url == original.token_url
        assert restored.client_id == original.client_id
        assert restored.client_secret == original.client_secret
        assert restored.scopes == original.scopes
        assert restored.extra_auth_params == original.extra_auth_params


class TestOAuthTokens:
    """Test OAuthTokens model."""

    def test_oauth_tokens_creation(self):
        """Test creating OAuthTokens."""
        tokens = OAuthTokens(access_token="test-access-token")

        assert tokens.access_token == "test-access-token"
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in is None
        assert tokens.refresh_token is None
        assert tokens.scope is None
        assert tokens.issued_at is None

    def test_oauth_tokens_with_all_fields(self):
        """Test OAuthTokens with all fields."""
        issued_at = time.time()
        tokens = OAuthTokens(
            access_token="test-access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="test-refresh-token",
            scope="read write",
            issued_at=issued_at,
        )

        assert tokens.access_token == "test-access-token"
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in == 3600
        assert tokens.refresh_token == "test-refresh-token"
        assert tokens.scope == "read write"
        assert tokens.issued_at == issued_at

    def test_oauth_tokens_from_dict(self):
        """Test creating OAuthTokens from dictionary."""
        issued_at = time.time()
        data = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token",
            "scope": "read write",
            "issued_at": issued_at,
        }

        tokens = OAuthTokens.model_validate(data)

        assert tokens.access_token == data["access_token"]
        assert tokens.token_type == data["token_type"]
        assert tokens.expires_in == data["expires_in"]
        assert tokens.refresh_token == data["refresh_token"]
        assert tokens.scope == data["scope"]
        assert tokens.issued_at == data["issued_at"]

    def test_oauth_tokens_from_dict_minimal(self):
        """Test creating OAuthTokens from minimal dictionary."""
        data = {"access_token": "test-access-token"}

        tokens = OAuthTokens.model_validate(data)

        assert tokens.access_token == "test-access-token"
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in is None
        assert tokens.refresh_token is None
        assert tokens.scope is None
        assert tokens.issued_at is None

    def test_oauth_tokens_to_dict(self):
        """Test converting OAuthTokens to dictionary."""
        issued_at = time.time()
        tokens = OAuthTokens(
            access_token="test-access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="test-refresh-token",
            scope="read write",
            issued_at=issued_at,
        )

        data = tokens.model_dump()

        assert data["access_token"] == tokens.access_token
        assert data["token_type"] == tokens.token_type
        assert data["expires_in"] == tokens.expires_in
        assert data["refresh_token"] == tokens.refresh_token
        assert data["scope"] == tokens.scope
        assert data["issued_at"] == tokens.issued_at

    def test_oauth_tokens_to_dict_minimal(self):
        """Test converting minimal OAuthTokens to dictionary."""
        tokens = OAuthTokens(access_token="test-access-token")

        data = tokens.model_dump(exclude_none=True, exclude_defaults=True)

        assert data["access_token"] == "test-access-token"
        # token_type may or may not be included depending on if it's default
        assert "expires_in" not in data
        assert "refresh_token" not in data
        assert "scope" not in data
        assert "issued_at" not in data

    def test_oauth_tokens_round_trip(self):
        """Test round-trip conversion to/from dictionary."""
        issued_at = time.time()
        original = OAuthTokens(
            access_token="test-access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="test-refresh-token",
            scope="read write",
            issued_at=issued_at,
        )

        data = original.model_dump()
        restored = OAuthTokens.model_validate(data)

        assert restored.access_token == original.access_token
        assert restored.token_type == original.token_type
        assert restored.expires_in == original.expires_in
        assert restored.refresh_token == original.refresh_token
        assert restored.scope == original.scope
        assert restored.issued_at == original.issued_at

    def test_is_expired_without_expiry(self):
        """Test is_expired when expires_in is None."""
        tokens = OAuthTokens(access_token="test-access-token")
        assert tokens.is_expired() is False

    def test_is_expired_without_issued_at(self):
        """Test is_expired when issued_at is None."""
        tokens = OAuthTokens(access_token="test-access-token", expires_in=3600)
        assert tokens.is_expired() is False

    def test_is_expired_fresh_token(self):
        """Test is_expired with fresh token."""
        tokens = OAuthTokens(
            access_token="test-access-token",
            expires_in=3600,
            issued_at=time.time(),
        )
        assert tokens.is_expired() is False

    def test_is_expired_old_token(self):
        """Test is_expired with expired token."""
        # Token issued 2 hours ago with 1 hour expiry
        tokens = OAuthTokens(
            access_token="test-access-token",
            expires_in=3600,
            issued_at=time.time() - 7200,
        )
        assert tokens.is_expired() is True

    def test_is_expired_near_expiry(self):
        """Test is_expired when token is near expiry (within 5 minutes)."""
        # Token issued 56 minutes ago with 1 hour expiry (4 minutes left)
        tokens = OAuthTokens(
            access_token="test-access-token",
            expires_in=3600,
            issued_at=time.time() - 3360,
        )
        # Should be considered expired (within 5 minute buffer)
        assert tokens.is_expired() is True

    def test_get_authorization_header_bearer(self):
        """Test get_authorization_header with Bearer token."""
        tokens = OAuthTokens(
            access_token="test-access-token",
            token_type="Bearer",
        )
        header = tokens.get_authorization_header()
        assert header == "Bearer test-access-token"

    def test_get_authorization_header_lowercase_bearer(self):
        """Test get_authorization_header with lowercase bearer."""
        tokens = OAuthTokens(
            access_token="test-access-token",
            token_type="bearer",
        )
        header = tokens.get_authorization_header()
        # Should capitalize Bearer per RFC 6750
        assert header == "Bearer test-access-token"

    def test_get_authorization_header_other_type(self):
        """Test get_authorization_header with other token type."""
        tokens = OAuthTokens(
            access_token="test-access-token",
            token_type="MAC",
        )
        header = tokens.get_authorization_header()
        assert header == "MAC test-access-token"
