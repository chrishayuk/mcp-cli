"""Tests for SecureTokenStore abstract interface."""

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.secure_token_store import SecureTokenStore, TokenStorageError


class ConcreteTokenStore(SecureTokenStore):
    """Concrete implementation for testing."""

    def __init__(self):
        """Initialize test store."""
        self._storage = {}

    def store_token(self, server_name: str, tokens: OAuthTokens) -> None:
        """Store OAuth tokens."""
        self._storage[f"oauth:{server_name}"] = json.dumps(tokens.to_dict())

    def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
        """Retrieve OAuth tokens."""
        key = f"oauth:{server_name}"
        if key in self._storage:
            data = json.loads(self._storage[key])
            return OAuthTokens.from_dict(data)
        return None

    def delete_token(self, server_name: str) -> bool:
        """Delete OAuth tokens."""
        key = f"oauth:{server_name}"
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    def has_token(self, server_name: str) -> bool:
        """Check if token exists."""
        return f"oauth:{server_name}" in self._storage

    def _store_raw(self, key: str, value: str) -> None:
        """Store raw value."""
        self._storage[key] = value

    def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw value."""
        return self._storage.get(key)

    def _delete_raw(self, key: str) -> bool:
        """Delete raw value."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False


class TestSecureTokenStoreInterface:
    """Test SecureTokenStore interface."""

    @pytest.fixture
    def store(self):
        """Provide concrete store instance."""
        return ConcreteTokenStore()

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            expires_in=3600,
            token_type="Bearer",
        )

    def test_store_and_retrieve_oauth_token(self, store, sample_tokens):
        """Test storing and retrieving OAuth tokens."""
        store.store_token("test-server", sample_tokens)
        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
        assert retrieved.access_token == sample_tokens.access_token
        assert retrieved.refresh_token == sample_tokens.refresh_token

    def test_has_token(self, store, sample_tokens):
        """Test checking if token exists."""
        assert not store.has_token("test-server")

        store.store_token("test-server", sample_tokens)
        assert store.has_token("test-server")

    def test_delete_token(self, store, sample_tokens):
        """Test deleting tokens."""
        store.store_token("test-server", sample_tokens)
        assert store.delete_token("test-server") is True
        assert not store.has_token("test-server")
        assert store.delete_token("test-server") is False

    def test_store_generic_token(self, store):
        """Test storing generic token."""
        store.store_generic("my-key", "my-value", "test-ns")

        # Verify it was stored
        retrieved = store.retrieve_generic("my-key", "test-ns")
        assert retrieved == "my-value"

    def test_retrieve_generic_token_nonexistent(self, store):
        """Test retrieving nonexistent generic token."""
        result = store.retrieve_generic("nonexistent", "test-ns")
        assert result is None

    def test_retrieve_generic_token_json_error(self, store):
        """Test retrieving generic token with invalid JSON."""
        # Store invalid JSON directly
        store._store_raw("test-ns:bad-key", "invalid json {")

        # Should return None on JSON error
        result = store.retrieve_generic("bad-key", "test-ns")
        assert result is None

    def test_retrieve_generic_token_missing_field(self, store):
        """Test retrieving generic token with missing field in JSON."""
        from mcp_cli.auth.token_types import TokenType

        # Store JSON that's valid but missing 'token' in data
        invalid_stored = {
            "token_type": TokenType.BEARER.value,
            "name": "test",
            "data": {},  # Missing 'token' field
        }
        store._store_raw("test-ns:bad-key", json.dumps(invalid_stored))

        # Should return None when token field is missing
        result = store.retrieve_generic("bad-key", "test-ns")
        assert result is None

    def test_delete_generic_token(self, store):
        """Test deleting generic token."""
        store.store_generic("my-key", "my-value", "test-ns")
        assert store.delete_generic("my-key", "test-ns") is True
        assert store.retrieve_generic("my-key", "test-ns") is None
        assert store.delete_generic("my-key", "test-ns") is False

    def test_list_keys_default_implementation(self, store):
        """Test default list_keys implementation."""
        # Default implementation returns empty list
        keys = store.list_keys()
        assert keys == []

        keys = store.list_keys("some-namespace")
        assert keys == []

    def test_clear_all_default_implementation(self, store):
        """Test default clear_all implementation."""
        # Store some tokens
        store.store_generic("key1", "value1", "ns1")
        store.store_generic("key2", "value2", "ns1")
        store.store_generic("key3", "value3", "ns2")

        # Default implementation relies on list_keys which returns []
        # So it should return 0
        count = store.clear_all()
        assert count == 0

    def test_clear_all_with_namespace(self, store):
        """Test clear_all with namespace filtering."""
        count = store.clear_all("test-ns")
        assert count == 0

    def test_sanitize_name(self, store):
        """Test server name sanitization."""
        assert store._sanitize_name("simple-name") == "simple-name"
        assert store._sanitize_name("name:with:colons") == "name:with:colons"
        assert store._sanitize_name("name@domain.com") == "name_domain_com"
        assert store._sanitize_name("server/path") == "server_path"
        assert store._sanitize_name("test server!") == "test_server_"


class TestSecureTokenStoreWithListKeys:
    """Test SecureTokenStore with list_keys implementation."""

    class StoreWithListKeys(ConcreteTokenStore):
        """Store implementation with list_keys support."""

        def list_keys(self, namespace: Optional[str] = None) -> list[str]:
            """List all keys, optionally filtered by namespace."""
            keys = []
            for full_key in self._storage.keys():
                if ":" in full_key:
                    ns, key = full_key.split(":", 1)
                    if namespace is None or ns == namespace:
                        keys.append(key)
            return keys

    @pytest.fixture
    def store(self):
        """Provide store with list_keys support."""
        return self.StoreWithListKeys()

    def test_clear_all_with_list_keys(self, store):
        """Test clear_all when list_keys is implemented."""
        # Store tokens in different namespaces
        store.store_generic("key1", "value1", "ns1")
        store.store_generic("key2", "value2", "ns1")
        store.store_generic("key3", "value3", "ns2")

        # Clear ns1 only
        count = store.clear_all("ns1")
        assert count == 2
        assert store.retrieve_generic("key1", "ns1") is None
        assert store.retrieve_generic("key2", "ns1") is None
        assert store.retrieve_generic("key3", "ns2") == "value3"

    def test_clear_all_no_namespace(self, store):
        """Test clear_all without namespace filter."""
        # The clear_all method with no namespace will list keys but when deleting
        # it uses "generic" as default namespace. So store in generic namespace.
        store.store_generic("key1", "value1", "generic")
        store.store_generic("key2", "value2", "generic")

        count = store.clear_all(None)
        assert count == 2


class TestTokenStorageError:
    """Test TokenStorageError exception."""

    def test_token_storage_error_creation(self):
        """Test creating TokenStorageError."""
        error = TokenStorageError("Test error")
        assert str(error) == "Test error"

    def test_token_storage_error_inheritance(self):
        """Test TokenStorageError inherits from Exception."""
        error = TokenStorageError("Test error")
        assert isinstance(error, Exception)

    def test_token_storage_error_raise(self):
        """Test raising TokenStorageError."""
        with pytest.raises(TokenStorageError, match="Test error"):
            raise TokenStorageError("Test error")


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that SecureTokenStore cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SecureTokenStore()

    def test_missing_store_token_raises_error(self):
        """Test that missing store_token raises TypeError."""

        class IncompleteStore(SecureTokenStore):
            def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
                pass

            def delete_token(self, server_name: str) -> bool:
                pass

            def has_token(self, server_name: str) -> bool:
                pass

            def _store_raw(self, key: str, value: str) -> None:
                pass

            def _retrieve_raw(self, key: str) -> Optional[str]:
                pass

            def _delete_raw(self, key: str) -> bool:
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStore()
