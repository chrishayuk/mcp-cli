"""Tests for token types and models."""

import time

import pytest

from mcp_cli.auth.token_types import (
    APIKeyToken,
    BasicAuthToken,
    BearerToken,
    StoredToken,
    TokenType,
)


class TestTokenType:
    """Test TokenType enum."""

    def test_token_type_values(self):
        """Test TokenType enum values."""
        assert TokenType.OAUTH == "oauth"
        assert TokenType.BEARER == "bearer"
        assert TokenType.API_KEY == "api_key"
        assert TokenType.BASIC_AUTH == "basic_auth"


class TestStoredToken:
    """Test StoredToken dataclass."""

    def test_to_dict_without_metadata(self):
        """Test converting to dict without metadata."""
        token = StoredToken(
            token_type=TokenType.BEARER,
            name="test-token",
            data={"token": "abc123"},
        )

        result = token.model_dump()

        assert result["token_type"] == "bearer"
        assert result["name"] == "test-token"
        assert result["data"] == {"token": "abc123"}
        assert result["metadata"] is None

    def test_to_dict_with_metadata(self):
        """Test converting to dict with metadata."""
        token = StoredToken(
            token_type=TokenType.BEARER,
            name="test-token",
            data={"token": "abc123"},
            metadata={"created_at": "2024-01-01"},
        )

        result = token.model_dump()

        assert result["metadata"] == {"created_at": "2024-01-01"}

    def test_from_dict_without_metadata(self):
        """Test creating from dict without metadata."""
        data = {
            "token_type": "bearer",
            "name": "test-token",
            "data": {"token": "abc123"},
        }

        token = StoredToken.model_validate(data)

        assert token.token_type == TokenType.BEARER
        assert token.name == "test-token"
        assert token.data == {"token": "abc123"}
        assert token.metadata is None

    def test_from_dict_with_metadata(self):
        """Test creating from dict with metadata."""
        data = {
            "token_type": "bearer",
            "name": "test-token",
            "data": {"token": "abc123"},
            "metadata": {"created_at": "2024-01-01"},
        }

        token = StoredToken.model_validate(data)

        assert token.metadata == {"created_at": "2024-01-01"}

    def test_get_display_info_oauth_full(self):
        """Test display info for OAuth token with all fields."""
        token = StoredToken(
            token_type=TokenType.OAUTH,
            name="test-server",
            data={
                "access_token": "access123",
                "refresh_token": "refresh456",
                "expires_at": 1234567890.0,
                "issued_at": 1234567000.0,
            },
        )

        info = token.get_display_info()

        assert info["name"] == "test-server"
        assert info["type"] == "oauth"
        assert info["has_refresh_token"] is True
        assert info["expires_at"] == 1234567890.0
        assert info["issued_at"] == 1234567000.0

    def test_get_display_info_oauth_no_refresh(self):
        """Test display info for OAuth token without refresh token."""
        token = StoredToken(
            token_type=TokenType.OAUTH,
            name="test-server",
            data={"access_token": "access123"},
        )

        info = token.get_display_info()

        assert info["has_refresh_token"] is False
        assert "expires_at" not in info
        assert "issued_at" not in info

    def test_get_display_info_bearer_with_expiry(self):
        """Test display info for Bearer token with expiry."""
        token = StoredToken(
            token_type=TokenType.BEARER,
            name="test-token",
            data={
                "token": "bearer123",
                "expires_at": 1234567890.0,
            },
        )

        info = token.get_display_info()

        assert info["name"] == "test-token"
        assert info["type"] == "bearer"
        assert info["has_token"] is True
        assert info["expires_at"] == 1234567890.0

    def test_get_display_info_bearer_no_expiry(self):
        """Test display info for Bearer token without expiry."""
        token = StoredToken(
            token_type=TokenType.BEARER,
            name="test-token",
            data={"token": "bearer123"},
        )

        info = token.get_display_info()

        assert info["has_token"] is True
        assert "expires_at" not in info

    def test_get_display_info_api_key_long(self):
        """Test display info for API key (long key)."""
        token = StoredToken(
            token_type=TokenType.API_KEY,
            name="openai-key",
            data={
                "provider": "openai",
                "key": "sk-1234567890abcdefghij",
            },
        )

        info = token.get_display_info()

        assert info["name"] == "openai-key"
        assert info["type"] == "api_key"
        assert info["provider"] == "openai"
        assert info["key_preview"] == "sk-1...ghij"

    def test_get_display_info_api_key_short(self):
        """Test display info for API key (short key)."""
        token = StoredToken(
            token_type=TokenType.API_KEY,
            name="test-key",
            data={
                "provider": "test",
                "key": "short",
            },
        )

        info = token.get_display_info()

        assert info["key_preview"] == "****"

    def test_get_display_info_api_key_no_key(self):
        """Test display info for API key without key field."""
        token = StoredToken(
            token_type=TokenType.API_KEY,
            name="test-key",
            data={"provider": "test"},
        )

        info = token.get_display_info()

        assert info["provider"] == "test"
        assert "key_preview" not in info

    def test_get_display_info_api_key_no_provider(self):
        """Test display info for API key without provider."""
        token = StoredToken(
            token_type=TokenType.API_KEY,
            name="test-key",
            data={"key": "sk-1234567890"},
        )

        info = token.get_display_info()

        assert info["provider"] == "unknown"

    def test_get_display_info_basic_auth(self):
        """Test display info for Basic auth."""
        token = StoredToken(
            token_type=TokenType.BASIC_AUTH,
            name="test-auth",
            data={
                "username": "testuser",
                "password": "secret",
            },
        )

        info = token.get_display_info()

        assert info["name"] == "test-auth"
        assert info["type"] == "basic_auth"
        assert info["username"] == "testuser"
        assert "password" not in info

    def test_get_display_info_basic_auth_no_username(self):
        """Test display info for Basic auth without username."""
        token = StoredToken(
            token_type=TokenType.BASIC_AUTH,
            name="test-auth",
            data={"password": "secret"},
        )

        info = token.get_display_info()

        assert info["username"] == "unknown"

    def test_get_display_info_with_metadata(self):
        """Test display info includes metadata."""
        token = StoredToken(
            token_type=TokenType.BEARER,
            name="test-token",
            data={"token": "bearer123"},
            metadata={"created": "2024-01-01", "source": "test"},
        )

        info = token.get_display_info()

        assert info["metadata"] == {"created": "2024-01-01", "source": "test"}


class TestBearerToken:
    """Test BearerToken dataclass."""

    def test_to_stored_token_without_expiry(self):
        """Test converting to StoredToken without expiry."""
        bearer = BearerToken(token="test-token-123")

        stored = bearer.to_stored_token("my-token")

        assert stored.token_type == TokenType.BEARER
        assert stored.name == "my-token"
        assert stored.data["token"] == "test-token-123"
        assert "expires_at" not in stored.data

    def test_to_stored_token_with_expiry(self):
        """Test converting to StoredToken with expiry."""
        bearer = BearerToken(
            token="test-token-123",
            expires_at=1234567890.0,
        )

        stored = bearer.to_stored_token("my-token")

        assert stored.data["expires_at"] == 1234567890.0

    def test_from_stored_token(self):
        """Test creating from StoredToken."""
        stored = StoredToken(
            token_type=TokenType.BEARER,
            name="my-token",
            data={
                "token": "test-token-123",
                "expires_at": 1234567890.0,
            },
        )

        bearer = BearerToken.from_stored_token(stored)

        assert bearer.token == "test-token-123"
        assert bearer.expires_at == 1234567890.0

    def test_from_stored_token_wrong_type(self):
        """Test error when creating from wrong token type."""
        stored = StoredToken(
            token_type=TokenType.API_KEY,
            name="my-token",
            data={"key": "test"},
        )

        with pytest.raises(ValueError, match="Expected BEARER token"):
            BearerToken.from_stored_token(stored)

    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry set."""
        bearer = BearerToken(token="test-token")

        assert bearer.is_expired() is False

    def test_is_expired_not_expired(self):
        """Test is_expired when token is not expired."""
        future_time = time.time() + 1000
        bearer = BearerToken(
            token="test-token",
            expires_at=future_time,
        )

        assert bearer.is_expired() is False

    def test_is_expired_expired(self):
        """Test is_expired when token is expired."""
        past_time = time.time() - 1000
        bearer = BearerToken(
            token="test-token",
            expires_at=past_time,
        )

        assert bearer.is_expired() is True

    def test_is_expired_within_buffer(self):
        """Test is_expired with buffer seconds."""
        # Token expires in 200 seconds, but buffer is 300
        near_future = time.time() + 200
        bearer = BearerToken(
            token="test-token",
            expires_at=near_future,
        )

        assert bearer.is_expired(buffer_seconds=300) is True

    def test_is_expired_outside_buffer(self):
        """Test is_expired outside buffer."""
        # Token expires in 400 seconds, buffer is 300
        future_time = time.time() + 400
        bearer = BearerToken(
            token="test-token",
            expires_at=future_time,
        )

        assert bearer.is_expired(buffer_seconds=300) is False


class TestAPIKeyToken:
    """Test APIKeyToken dataclass."""

    def test_to_stored_token_minimal(self):
        """Test converting to StoredToken with minimal data."""
        api_key = APIKeyToken(
            provider="openai",
            key="sk-test123",
        )

        stored = api_key.to_stored_token("my-key")

        assert stored.token_type == TokenType.API_KEY
        assert stored.name == "my-key"
        assert stored.data["provider"] == "openai"
        assert stored.data["key"] == "sk-test123"
        assert "organization_id" not in stored.data
        assert "project_id" not in stored.data

    def test_to_stored_token_with_org(self):
        """Test converting to StoredToken with organization."""
        api_key = APIKeyToken(
            provider="openai",
            key="sk-test123",
            organization_id="org-123",
        )

        stored = api_key.to_stored_token("my-key")

        assert stored.data["organization_id"] == "org-123"

    def test_to_stored_token_with_project(self):
        """Test converting to StoredToken with project."""
        api_key = APIKeyToken(
            provider="openai",
            key="sk-test123",
            project_id="proj-456",
        )

        stored = api_key.to_stored_token("my-key")

        assert stored.data["project_id"] == "proj-456"

    def test_to_stored_token_full(self):
        """Test converting to StoredToken with all fields."""
        api_key = APIKeyToken(
            provider="openai",
            key="sk-test123",
            organization_id="org-123",
            project_id="proj-456",
        )

        stored = api_key.to_stored_token("my-key")

        assert stored.data["organization_id"] == "org-123"
        assert stored.data["project_id"] == "proj-456"

    def test_from_stored_token_minimal(self):
        """Test creating from StoredToken with minimal data."""
        stored = StoredToken(
            token_type=TokenType.API_KEY,
            name="my-key",
            data={
                "provider": "anthropic",
                "key": "sk-ant-test123",
            },
        )

        api_key = APIKeyToken.from_stored_token(stored)

        assert api_key.provider == "anthropic"
        assert api_key.key == "sk-ant-test123"
        assert api_key.organization_id is None
        assert api_key.project_id is None

    def test_from_stored_token_full(self):
        """Test creating from StoredToken with all fields."""
        stored = StoredToken(
            token_type=TokenType.API_KEY,
            name="my-key",
            data={
                "provider": "openai",
                "key": "sk-test123",
                "organization_id": "org-123",
                "project_id": "proj-456",
            },
        )

        api_key = APIKeyToken.from_stored_token(stored)

        assert api_key.organization_id == "org-123"
        assert api_key.project_id == "proj-456"

    def test_from_stored_token_wrong_type(self):
        """Test error when creating from wrong token type."""
        stored = StoredToken(
            token_type=TokenType.BEARER,
            name="my-token",
            data={"token": "test"},
        )

        with pytest.raises(ValueError, match="Expected API_KEY token"):
            APIKeyToken.from_stored_token(stored)


class TestBasicAuthToken:
    """Test BasicAuthToken dataclass."""

    def test_to_stored_token(self):
        """Test converting to StoredToken."""
        basic = BasicAuthToken(
            username="testuser",
            password="testpass",
        )

        stored = basic.to_stored_token("my-auth")

        assert stored.token_type == TokenType.BASIC_AUTH
        assert stored.name == "my-auth"
        assert stored.data["username"] == "testuser"
        assert stored.data["password"] == "testpass"

    def test_from_stored_token(self):
        """Test creating from StoredToken."""
        stored = StoredToken(
            token_type=TokenType.BASIC_AUTH,
            name="my-auth",
            data={
                "username": "testuser",
                "password": "testpass",
            },
        )

        basic = BasicAuthToken.from_stored_token(stored)

        assert basic.username == "testuser"
        assert basic.password == "testpass"

    def test_from_stored_token_wrong_type(self):
        """Test error when creating from wrong token type."""
        stored = StoredToken(
            token_type=TokenType.BEARER,
            name="my-token",
            data={"token": "test"},
        )

        with pytest.raises(ValueError, match="Expected BASIC_AUTH token"):
            BasicAuthToken.from_stored_token(stored)

    def test_get_auth_header(self):
        """Test generating Basic Auth header."""
        basic = BasicAuthToken(
            username="testuser",
            password="testpass",
        )

        header = basic.get_auth_header()

        assert header.startswith("Basic ")

        # Decode and verify
        import base64

        encoded = header.replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "testuser:testpass"

    def test_get_auth_header_special_chars(self):
        """Test generating Basic Auth header with special characters."""
        basic = BasicAuthToken(
            username="test@user.com",
            password="p@ss:w0rd!",
        )

        header = basic.get_auth_header()

        # Decode and verify
        import base64

        encoded = header.replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "test@user.com:p@ss:w0rd!"


class TestTokenTypeRoundTrips:
    """Test round-trip conversions between token types."""

    def test_bearer_token_roundtrip(self):
        """Test Bearer token can be converted back and forth."""
        original = BearerToken(
            token="test-token-123",
            expires_at=1234567890.0,
        )

        stored = original.to_stored_token("my-token")
        restored = BearerToken.from_stored_token(stored)

        assert restored.token == original.token
        assert restored.expires_at == original.expires_at

    def test_api_key_token_roundtrip(self):
        """Test API key token can be converted back and forth."""
        original = APIKeyToken(
            provider="openai",
            key="sk-test123",
            organization_id="org-123",
            project_id="proj-456",
        )

        stored = original.to_stored_token("my-key")
        restored = APIKeyToken.from_stored_token(stored)

        assert restored.provider == original.provider
        assert restored.key == original.key
        assert restored.organization_id == original.organization_id
        assert restored.project_id == original.project_id

    def test_basic_auth_token_roundtrip(self):
        """Test Basic auth token can be converted back and forth."""
        original = BasicAuthToken(
            username="testuser",
            password="testpass",
        )

        stored = original.to_stored_token("my-auth")
        restored = BasicAuthToken.from_stored_token(stored)

        assert restored.username == original.username
        assert restored.password == original.password

    def test_stored_token_roundtrip(self):
        """Test StoredToken serialization round-trip."""
        original = StoredToken(
            token_type=TokenType.OAUTH,
            name="test-server",
            data={
                "access_token": "access123",
                "refresh_token": "refresh456",
            },
            metadata={"created": "2024-01-01"},
        )

        dict_form = original.model_dump()
        restored = StoredToken.model_validate(dict_form)

        assert restored.token_type == original.token_type
        assert restored.name == original.name
        assert restored.data == original.data
        assert restored.metadata == original.metadata
