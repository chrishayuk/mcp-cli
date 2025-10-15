"""Tests for token storage backends."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.secure_token_store import TokenStorageError
from mcp_cli.auth.stores.encrypted_file_store import EncryptedFileTokenStore


class TestEncryptedFileTokenStore:
    """Test EncryptedFileTokenStore implementation."""

    @pytest.fixture
    def token_dir(self, tmp_path):
        """Provide temporary token directory."""
        return tmp_path / "tokens"

    @pytest.fixture
    def store(self, token_dir):
        """Provide EncryptedFileTokenStore instance."""
        return EncryptedFileTokenStore(
            token_dir=token_dir, password="test-password-123"
        )

    def test_init_with_default_token_dir(self):
        """Test initialization with default token directory."""
        store = EncryptedFileTokenStore(password="test-password")
        assert store.token_dir == Path.home() / ".mcp_cli" / "tokens"

    def test_init_with_env_password(self, token_dir, monkeypatch):
        """Test initialization with password from environment."""
        monkeypatch.setenv("MCP_CLI_ENCRYPTION_KEY", "env-password")
        store = EncryptedFileTokenStore(token_dir=token_dir)
        assert store.fernet is not None

    def test_init_with_prompted_password(self, token_dir, monkeypatch):
        """Test initialization with prompted password."""
        monkeypatch.setattr("getpass.getpass", lambda _: "prompted-password")
        store = EncryptedFileTokenStore(token_dir=token_dir)
        assert store.fernet is not None

    def test_init_missing_cryptography(self, token_dir, monkeypatch):
        """Test error when cryptography library is missing."""
        # This test would require mocking the import, skip for now
        # as it's an edge case that's hard to test
        pass

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            expires_in=3600,
            token_type="Bearer",
        )

    def test_init_creates_directory(self, token_dir):
        """Test that initialization creates token directory."""
        _ = EncryptedFileTokenStore(token_dir=token_dir, password="test-password")

        assert token_dir.exists()
        assert token_dir.stat().st_mode & 0o777 == 0o700

    def test_store_and_retrieve_token(self, store, sample_tokens):
        """Test storing and retrieving OAuth tokens."""
        server_name = "test-server"

        # Store token
        store.store_token(server_name, sample_tokens)

        # Retrieve token
        retrieved = store.retrieve_token(server_name)

        assert retrieved is not None
        assert retrieved.access_token == sample_tokens.access_token
        assert retrieved.refresh_token == sample_tokens.refresh_token
        assert retrieved.expires_in == sample_tokens.expires_in
        assert retrieved.token_type == sample_tokens.token_type

    def test_retrieve_nonexistent_token(self, store):
        """Test retrieving a token that doesn't exist."""
        result = store.retrieve_token("nonexistent")
        assert result is None

    def test_delete_token(self, store, sample_tokens):
        """Test deleting a token."""
        server_name = "test-server"

        # Store token
        store.store_token(server_name, sample_tokens)
        assert store.has_token(server_name)

        # Delete token
        result = store.delete_token(server_name)
        assert result is True
        assert not store.has_token(server_name)

        # Try to delete again
        result = store.delete_token(server_name)
        assert result is False

    def test_has_token(self, store, sample_tokens):
        """Test checking if token exists."""
        server_name = "test-server"

        assert not store.has_token(server_name)

        store.store_token(server_name, sample_tokens)
        assert store.has_token(server_name)

    def test_store_raw(self, store):
        """Test storing raw string values."""
        key = "test-key"
        value = "test-value-12345"

        store._store_raw(key, value)

        # Verify file was created and encrypted
        safe_key = store._sanitize_name(key)
        file_path = store.token_dir / f"{safe_key}.enc"
        assert file_path.exists()

        # Verify we can't read it without decryption
        with open(file_path, "rb") as f:
            raw_data = f.read()
        assert value.encode() not in raw_data

    def test_retrieve_raw(self, store):
        """Test retrieving raw string values."""
        key = "test-key"
        value = "test-value-12345"

        store._store_raw(key, value)
        retrieved = store._retrieve_raw(key)

        assert retrieved == value

    def test_delete_raw(self, store):
        """Test deleting raw values."""
        key = "test-key"
        value = "test-value-12345"

        store._store_raw(key, value)
        assert store._delete_raw(key) is True
        assert store._retrieve_raw(key) is None
        assert store._delete_raw(key) is False

    def test_encryption_with_different_passwords(self, token_dir):
        """Test that different passwords can't decrypt each other's data."""
        # Store with first password
        store1 = EncryptedFileTokenStore(token_dir=token_dir, password="password1")
        store1._store_raw("test-key", "test-value")

        # Try to retrieve with different password
        store2 = EncryptedFileTokenStore(token_dir=token_dir, password="password2")

        with pytest.raises(TokenStorageError):
            store2._retrieve_raw("test-key")

    def test_raw_storage_with_special_chars(self, store):
        """Test raw storage handles special characters."""
        # Store with key containing special chars
        key = "test:server@domain"
        value = "test-value-12345"

        store._store_raw(key, value)
        retrieved = store._retrieve_raw(key)

        assert retrieved == value

        # Clean up
        store._delete_raw(key)

    def test_file_permissions(self, store, sample_tokens):
        """Test that stored files have correct permissions."""
        server_name = "test-server"
        store.store_token(server_name, sample_tokens)

        safe_name = store._sanitize_name(server_name)
        file_path = store.token_dir / f"{safe_name}.enc"

        # Check file permissions are user-only read/write
        import stat

        mode = file_path.stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_store_generic(self, store):
        """Test generic token storage."""
        key = "my-token"
        value = "token-value-12345"
        namespace = "test-ns"

        store.store_generic(key, value, namespace)

        # Verify we can retrieve it
        retrieved = store.retrieve_generic(key, namespace)
        assert retrieved == value

    def test_delete_generic(self, store):
        """Test generic token deletion."""
        key = "my-token"
        value = "token-value-12345"
        namespace = "test-ns"

        store.store_generic(key, value, namespace)
        assert store.delete_generic(key, namespace) is True
        assert store.retrieve_generic(key, namespace) is None
        assert store.delete_generic(key, namespace) is False

    def test_store_token_error_handling(self, store, sample_tokens, monkeypatch):
        """Test error handling when storing token fails."""

        def mock_open_error(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open_error)

        with pytest.raises(TokenStorageError, match="Failed to store encrypted token"):
            store.store_token("test-server", sample_tokens)

    def test_retrieve_token_json_error(self, store, token_dir):
        """Test error when token file contains invalid JSON."""
        # Create a file with invalid encrypted data
        token_path = token_dir / "test-server.enc"
        token_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid encrypted data that will cause JSON decode error
        with open(token_path, "wb") as f:
            # Encrypt invalid JSON
            encrypted = store.fernet.encrypt(b"invalid json {")
            f.write(encrypted)

        with pytest.raises(TokenStorageError, match="Failed to parse token data"):
            store.retrieve_token("test-server")

    def test_retrieve_token_decrypt_error(self, store, token_dir, sample_tokens):
        """Test error when decryption fails."""
        # Store with one password
        store.store_token("test-server", sample_tokens)

        # Create new store with different password
        store2 = EncryptedFileTokenStore(
            token_dir=token_dir, password="different-password"
        )

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve encrypted token"
        ):
            store2.retrieve_token("test-server")

    def test_delete_token_error_handling(self, store, monkeypatch):
        """Test error handling when deleting token fails."""
        # Create a token first
        from mcp_cli.auth.oauth_config import OAuthTokens

        tokens = OAuthTokens(access_token="test", expires_in=3600, token_type="Bearer")
        store.store_token("test-server", tokens)

        def mock_unlink_error(self):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("pathlib.Path.unlink", mock_unlink_error)

        with pytest.raises(TokenStorageError, match="Failed to delete encrypted token"):
            store.delete_token("test-server")

    def test_store_raw_error_handling(self, store, monkeypatch):
        """Test error handling when storing raw value fails."""

        def mock_open_error(*args, **kwargs):
            raise IOError("Disk full")

        monkeypatch.setattr("builtins.open", mock_open_error)

        with pytest.raises(TokenStorageError, match="Failed to store encrypted value"):
            store._store_raw("test-key", "test-value")

    def test_retrieve_raw_error_handling(self, store, token_dir):
        """Test error handling when retrieving raw value fails."""
        # Store a value first
        store._store_raw("test-key", "test-value")

        # Create new store with different password to cause decrypt error
        store2 = EncryptedFileTokenStore(
            token_dir=token_dir, password="different-password"
        )

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve encrypted value"
        ):
            store2._retrieve_raw("test-key")

    def test_delete_raw_error_handling(self, store, monkeypatch):
        """Test error handling when deleting raw value fails."""
        store._store_raw("test-key", "test-value")

        def mock_unlink_error(self):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("pathlib.Path.unlink", mock_unlink_error)

        with pytest.raises(TokenStorageError, match="Failed to delete encrypted value"):
            store._delete_raw("test-key")


class TestTokenStoreInterface:
    """Test the abstract SecureTokenStore interface behavior."""

    @pytest.fixture
    def store(self, tmp_path):
        """Provide a concrete store implementation."""
        return EncryptedFileTokenStore(
            token_dir=tmp_path / "tokens", password="test-password"
        )

    def test_store_and_retrieve_oauth_tokens(self, store):
        """Test OAuth token storage and retrieval."""
        tokens = OAuthTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_in=3600,
            token_type="Bearer",
        )

        store.store_token("test-server", tokens)
        retrieved = store.retrieve_token("test-server")

        assert retrieved.access_token == tokens.access_token
        assert retrieved.refresh_token == tokens.refresh_token

    def test_generic_token_operations(self, store):
        """Test generic token storage operations."""
        # Store
        store.store_generic("key1", "value1", "ns1")
        store.store_generic("key2", "value2", "ns2")

        # Retrieve
        assert store.retrieve_generic("key1", "ns1") == "value1"
        assert store.retrieve_generic("key2", "ns2") == "value2"
        assert store.retrieve_generic("nonexistent", "ns1") is None

        # Delete
        assert store.delete_generic("key1", "ns1") is True
        assert store.retrieve_generic("key1", "ns1") is None


@pytest.mark.skipif(
    not hasattr(__import__("sys"), "platform")
    or __import__("sys").platform != "darwin",
    reason="Keychain tests only run on macOS",
)
class TestKeychainStoreIntegration:
    """Integration tests for Keychain store (macOS only)."""

    def test_keychain_available(self):
        """Test that keyring library is available."""
        try:
            import keyring

            assert keyring is not None
        except ImportError:
            pytest.skip("keyring library not available")

    def test_keychain_store_creation(self):
        """Test creating a Keychain store."""
        try:
            from mcp_cli.auth.stores.keychain_store import KeychainTokenStore

            store = KeychainTokenStore()
            assert store is not None
        except ImportError:
            pytest.skip("KeychainTokenStore not available")


class TestTokenStoreFactory:
    """Test TokenStoreFactory."""

    def test_detect_backend(self):
        """Test backend detection."""
        from mcp_cli.auth.token_store_factory import TokenStoreFactory

        backend = TokenStoreFactory._detect_backend()
        assert backend is not None

    def test_create_encrypted_file_store(self, tmp_path):
        """Test creating encrypted file store."""
        from mcp_cli.auth.token_store_factory import (
            TokenStoreBackend,
            TokenStoreFactory,
        )

        store = TokenStoreFactory.create(
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            token_dir=tmp_path / "tokens",
            password="test-password",
        )

        assert isinstance(store, EncryptedFileTokenStore)

    def test_get_available_backends(self):
        """Test getting available backends."""
        from mcp_cli.auth.token_store_factory import TokenStoreFactory

        backends = TokenStoreFactory.get_available_backends()
        assert len(backends) > 0
        # Encrypted file should always be available
        assert any("encrypted" in str(b).lower() for b in backends)

    @patch.dict(
        "os.environ", {"VAULT_ADDR": "http://vault:8200", "VAULT_TOKEN": "test"}
    )
    def test_vault_detection(self):
        """Test that Vault is detected when env vars are set."""
        from mcp_cli.auth.token_store_factory import TokenStoreFactory

        backend = TokenStoreFactory._detect_backend()
        # With env vars set, Vault should be detected (if hvac is installed)
        # This will fall back to another backend if hvac is not installed
        assert backend is not None
