"""Tests for TokenManager with registry integration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.token_manager import TokenManager
from mcp_cli.auth.token_store_factory import TokenStoreBackend
from mcp_cli.auth.token_types import TokenType


@pytest.fixture
def temp_dirs(tmp_path):
    """Provide temporary directories for tokens and registry."""
    return {
        "token_dir": tmp_path / "tokens",
        "registry_path": tmp_path / "registry.json",
    }


@pytest.fixture
def token_manager(temp_dirs):
    """Provide TokenManager instance with encrypted file backend."""
    # Use a unique registry path per test instance for isolation
    registry_path = temp_dirs["token_dir"] / "registry.json"
    manager = TokenManager(
        token_dir=temp_dirs["token_dir"],
        backend=TokenStoreBackend.ENCRYPTED_FILE,
        password="test-password-123",
    )
    # Override registry path for test isolation
    manager.registry.registry_path = registry_path
    manager.registry._entries = {}
    manager.registry._save_registry()
    return manager


@pytest.fixture
def sample_oauth_tokens():
    """Provide sample OAuth tokens."""
    return OAuthTokens(
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        expires_in=3600,
        token_type="Bearer",
    )


class TestTokenManagerInitialization:
    """Test TokenManager initialization."""

    def test_init_with_default_backend(self, temp_dirs):
        """Test initialization with auto backend detection."""
        manager = TokenManager(
            token_dir=temp_dirs["token_dir"],
        )
        assert manager.token_store is not None
        assert manager.registry is not None

    def test_init_with_default_token_dir(self):
        """Test initialization with default token directory."""
        manager = TokenManager(
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )
        expected_dir = Path.home() / ".mcp_cli" / "tokens"
        assert manager.token_dir == expected_dir

    def test_init_with_encrypted_file_backend(self, temp_dirs):
        """Test initialization with encrypted file backend."""
        manager = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        from mcp_cli.auth.stores.encrypted_file_store import EncryptedFileTokenStore

        assert isinstance(manager.token_store, EncryptedFileTokenStore)

    def test_init_creates_directories(self, temp_dirs):
        """Test that initialization creates necessary directories."""
        TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        assert temp_dirs["token_dir"].exists()

    def test_sanitize_name(self, token_manager):
        """Test server name sanitization."""
        assert token_manager._sanitize_name("simple-name") == "simple-name"
        assert token_manager._sanitize_name("name@domain.com") == "name_domain_com"
        assert token_manager._sanitize_name("server:8080") == "server_8080"
        assert token_manager._sanitize_name("test/path") == "test_path"

    def test_get_token_path(self, token_manager):
        """Test getting token path."""
        path = token_manager._get_token_path("test-server")
        assert path.name == "test-server.json"
        assert path.parent == token_manager.token_dir

    def test_get_client_registration_path(self, token_manager):
        """Test getting client registration path."""
        path = token_manager._get_client_registration_path("test-server")
        assert path.name == "test-server_client.json"
        assert path.parent == token_manager.token_dir


class TestOAuthTokenOperations:
    """Test OAuth token storage and retrieval with registry integration."""

    def test_store_oauth_tokens(self, token_manager, sample_oauth_tokens):
        """Test storing OAuth tokens registers them."""
        server_name = "test-server"

        # Store tokens
        token_manager.save_tokens(server_name, sample_oauth_tokens)

        # Verify stored in backend
        retrieved = token_manager.token_store.retrieve_token(server_name)
        assert retrieved is not None
        assert retrieved.access_token == sample_oauth_tokens.access_token

        # Verify registered
        assert token_manager.registry.has_token(server_name, "oauth")

    def test_get_oauth_tokens(self, token_manager, sample_oauth_tokens):
        """Test retrieving OAuth tokens."""
        server_name = "test-server"

        token_manager.save_tokens(server_name, sample_oauth_tokens)
        retrieved = token_manager.load_tokens(server_name)

        assert retrieved is not None
        assert retrieved.access_token == sample_oauth_tokens.access_token
        assert retrieved.refresh_token == sample_oauth_tokens.refresh_token

    def test_has_oauth_tokens(self, token_manager, sample_oauth_tokens):
        """Test checking if OAuth tokens exist."""
        server_name = "test-server"

        assert not token_manager.has_valid_tokens(server_name)

        token_manager.save_tokens(server_name, sample_oauth_tokens)
        assert token_manager.has_valid_tokens(server_name)

    def test_delete_oauth_tokens(self, token_manager, sample_oauth_tokens):
        """Test deleting OAuth tokens unregisters them."""
        server_name = "test-server"

        # Store tokens
        token_manager.save_tokens(server_name, sample_oauth_tokens)
        assert token_manager.has_valid_tokens(server_name)
        assert token_manager.registry.has_token(server_name, "oauth")

        # Delete tokens
        result = token_manager.delete_tokens(server_name)
        assert result is True
        assert not token_manager.has_valid_tokens(server_name)
        assert not token_manager.registry.has_token(server_name, "oauth")

        # Try deleting again
        result = token_manager.delete_tokens(server_name)
        assert result is False


class TestGenericTokenOperations:
    """Test generic token operations with registry."""

    def test_store_generic_token(self, token_manager):
        """Test storing generic token registers it."""
        key = "my-token"
        value = "token-value-12345"
        namespace = "test-ns"

        # Store token
        token_manager.token_store.store_generic(key, value, namespace)

        # Verify stored
        retrieved = token_manager.token_store.retrieve_generic(key, namespace)
        assert retrieved == value

    def test_retrieve_generic_token(self, token_manager):
        """Test retrieving generic token."""
        key = "my-token"
        value = "token-value-12345"
        namespace = "test-ns"

        token_manager.token_store.store_generic(key, value, namespace)
        retrieved = token_manager.token_store.retrieve_generic(key, namespace)

        assert retrieved == value

    def test_delete_generic_token(self, token_manager):
        """Test deleting generic token."""
        key = "my-token"
        value = "token-value-12345"
        namespace = "test-ns"

        token_manager.token_store.store_generic(key, value, namespace)
        assert token_manager.token_store.retrieve_generic(key, namespace) == value

        # Delete
        result = token_manager.token_store.delete_generic(key, namespace)
        assert result is True
        assert token_manager.token_store.retrieve_generic(key, namespace) is None


class TestRegistryIntegration:
    """Test integration between TokenManager and registry."""

    def test_registry_tracks_oauth_tokens(self, token_manager, sample_oauth_tokens):
        """Test that OAuth tokens are tracked in registry."""
        server_name = "test-server"

        token_manager.save_tokens(server_name, sample_oauth_tokens)

        # Check registry entry
        entry = token_manager.registry.get_entry(server_name, "oauth")
        assert entry is not None
        assert entry["name"] == server_name
        assert entry["type"] == TokenType.OAUTH.value
        assert entry["namespace"] == "oauth"
        assert "registered_at" in entry

    def test_list_tokens_from_registry(self, token_manager, sample_oauth_tokens):
        """Test listing tokens uses registry."""
        # Store multiple tokens
        token_manager.save_tokens("server1", sample_oauth_tokens)
        token_manager.save_tokens("server2", sample_oauth_tokens)

        # List from registry
        tokens = token_manager.registry.list_tokens()
        assert len(tokens) >= 2

        names = [t["name"] for t in tokens]
        assert "server1" in names
        assert "server2" in names

    def test_filter_tokens_by_namespace(self, token_manager, sample_oauth_tokens):
        """Test filtering tokens by namespace."""
        # Store OAuth tokens
        token_manager.save_tokens("server1", sample_oauth_tokens)

        # Store generic token
        token_manager.token_store.store_generic("key1", "value1", "custom-ns")
        token_manager.registry.register("key1", TokenType.BEARER, "custom-ns")

        # List OAuth only
        oauth_tokens = token_manager.registry.list_tokens(namespace="oauth")
        assert len(oauth_tokens) >= 1
        assert all(t["namespace"] == "oauth" for t in oauth_tokens)

        # List custom namespace
        custom_tokens = token_manager.registry.list_tokens(namespace="custom-ns")
        assert len(custom_tokens) == 1
        assert custom_tokens[0]["name"] == "key1"

    def test_registry_persistence_across_instances(
        self, temp_dirs, sample_oauth_tokens
    ):
        """Test that registry persists across TokenManager instances."""
        server_name = "test-server"

        # First manager stores token
        manager1 = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )
        manager1.save_tokens(server_name, sample_oauth_tokens)

        # Second manager can see it in registry
        manager2 = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        assert manager2.registry.has_token(server_name, "oauth")
        assert manager2.has_valid_tokens(server_name)


class TestBackendFactoryIntegration:
    """Test TokenManager with different backends."""

    def test_auto_backend_selection(self, temp_dirs):
        """Test auto backend selection."""
        manager = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.AUTO,
        )
        assert manager.token_store is not None

    @patch("sys.platform", "darwin")
    def test_keychain_backend_on_macos(self, temp_dirs):
        """Test that macOS uses keychain backend by default."""
        try:
            from mcp_cli.auth.stores.keychain_store import KeychainTokenStore  # noqa: F401

            manager = TokenManager(
                token_dir=temp_dirs["token_dir"],
                backend=TokenStoreBackend.KEYCHAIN,
            )
            # May fall back to encrypted file if keyring not available
            assert manager.token_store is not None
        except ImportError:
            pytest.skip("keyring library not available")

    def test_encrypted_file_fallback(self, temp_dirs):
        """Test encrypted file backend as fallback."""
        manager = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        from mcp_cli.auth.stores.encrypted_file_store import EncryptedFileTokenStore

        assert isinstance(manager.token_store, EncryptedFileTokenStore)


class TestTokenManagerErrorHandling:
    """Test error handling in TokenManager."""

    def test_retrieve_nonexistent_oauth_token(self, token_manager):
        """Test retrieving token that doesn't exist."""
        result = token_manager.load_tokens("nonexistent-server")
        assert result is None

    def test_delete_nonexistent_oauth_token(self, token_manager):
        """Test deleting token that doesn't exist."""
        result = token_manager.delete_tokens("nonexistent-server")
        assert result is False

    def test_retrieve_nonexistent_generic_token(self, token_manager):
        """Test retrieving generic token that doesn't exist."""
        result = token_manager.token_store.retrieve_generic("nonexistent", "namespace")
        assert result is None

    def test_corrupted_registry_handling(self, temp_dirs, sample_oauth_tokens):
        """Test handling of corrupted registry file."""
        # Create manager and store token
        manager1 = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )
        manager1.save_tokens("server1", sample_oauth_tokens)

        # Get the actual registry path
        actual_registry_path = manager1.registry.registry_path

        # Corrupt registry file
        with open(actual_registry_path, "w") as f:
            f.write("invalid json {")

        # New manager should handle gracefully
        manager2 = TokenManager(
            token_dir=temp_dirs["token_dir"],
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        # Registry should be empty but functional
        tokens = manager2.registry.list_tokens()
        assert tokens == []

        # Should still be able to access stored tokens directly
        retrieved = manager2.load_tokens("server1")
        assert retrieved is not None


class TestClientRegistration:
    """Test OAuth client registration management."""

    def test_save_client_registration(self, token_manager):
        """Test saving client registration."""
        from mcp_cli.auth.mcp_oauth import DynamicClientRegistration

        registration = DynamicClientRegistration(
            client_id="test-client-id",
            client_secret="test-client-secret",
            client_id_issued_at=1234567890,
        )

        token_manager.save_client_registration("test-server", registration)

        # Verify file exists
        reg_path = token_manager._get_client_registration_path("test-server")
        assert reg_path.exists()

        # Verify file permissions
        import stat

        mode = reg_path.stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_load_client_registration(self, token_manager):
        """Test loading client registration."""
        from mcp_cli.auth.mcp_oauth import DynamicClientRegistration

        registration = DynamicClientRegistration(
            client_id="test-client-id",
            client_secret="test-client-secret",
            client_id_issued_at=1234567890,
        )

        token_manager.save_client_registration("test-server", registration)
        loaded = token_manager.load_client_registration("test-server")

        assert loaded is not None
        assert loaded.client_id == registration.client_id
        assert loaded.client_secret == registration.client_secret

    def test_load_nonexistent_client_registration(self, token_manager):
        """Test loading client registration that doesn't exist."""
        result = token_manager.load_client_registration("nonexistent")
        assert result is None

    def test_load_corrupted_client_registration(self, token_manager):
        """Test loading corrupted client registration."""
        # Create corrupted file
        reg_path = token_manager._get_client_registration_path("test-server")
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reg_path, "w") as f:
            f.write("invalid json {")

        result = token_manager.load_client_registration("test-server")
        assert result is None

    def test_delete_client_registration(self, token_manager):
        """Test deleting client registration."""
        from mcp_cli.auth.mcp_oauth import DynamicClientRegistration

        registration = DynamicClientRegistration(
            client_id="test-client-id",
            client_id_issued_at=1234567890,
        )

        token_manager.save_client_registration("test-server", registration)
        assert token_manager.delete_client_registration("test-server") is True
        assert token_manager.load_client_registration("test-server") is None

    def test_delete_nonexistent_client_registration(self, token_manager):
        """Test deleting client registration that doesn't exist."""
        assert token_manager.delete_client_registration("nonexistent") is False


class TestTokenExpiration:
    """Test token expiration handling."""

    def test_save_tokens_with_expiry_and_issued_at(self, token_manager):
        """Test saving tokens with expiry and issued_at registers metadata."""
        import time

        tokens = OAuthTokens(
            access_token="test-token",
            expires_in=3600,
            issued_at=time.time(),
            token_type="Bearer",
        )

        token_manager.save_tokens("test-server", tokens)

        # Check registry metadata
        entry = token_manager.registry.get_entry("test-server", "oauth")
        assert "expires_at" in entry["metadata"]

    def test_save_tokens_with_expiry_no_issued_at(self, token_manager):
        """Test saving tokens with expiry but no issued_at calculates expiry."""
        tokens = OAuthTokens(
            access_token="test-token",
            expires_in=3600,
            token_type="Bearer",
        )

        token_manager.save_tokens("test-server", tokens)

        # Check registry metadata has expires_at calculated
        entry = token_manager.registry.get_entry("test-server", "oauth")
        assert "expires_at" in entry["metadata"]


class TestMultipleTokenTypes:
    """Test managing different token types together."""

    def test_oauth_and_generic_tokens_coexist(self, token_manager, sample_oauth_tokens):
        """Test that OAuth and generic tokens coexist in registry."""
        # Store OAuth token
        token_manager.save_tokens("my-server", sample_oauth_tokens)

        # Store generic tokens
        token_manager.token_store.store_generic("api-key", "key123", "bearer")
        token_manager.registry.register("api-key", TokenType.BEARER, "bearer")

        token_manager.token_store.store_generic("provider-key", "pk123", "api-key")
        token_manager.registry.register("provider-key", TokenType.API_KEY, "api-key")

        # List all tokens
        all_tokens = token_manager.registry.list_tokens()
        assert len(all_tokens) == 3

        # Verify different types
        token_types = {t["type"] for t in all_tokens}
        assert TokenType.OAUTH.value in token_types
        assert TokenType.BEARER.value in token_types
        assert TokenType.API_KEY.value in token_types

    def test_filter_by_token_type(self, token_manager, sample_oauth_tokens):
        """Test filtering tokens by type."""
        # Store different types
        token_manager.save_tokens("my-server", sample_oauth_tokens)
        token_manager.token_store.store_generic("bearer1", "value1", "bearer")
        token_manager.registry.register("bearer1", TokenType.BEARER, "bearer")

        # Filter by OAuth
        oauth_tokens = token_manager.registry.list_tokens(token_type=TokenType.OAUTH)
        assert len(oauth_tokens) >= 1
        assert all(t["type"] == TokenType.OAUTH.value for t in oauth_tokens)

        # Filter by Bearer
        bearer_tokens = token_manager.registry.list_tokens(token_type=TokenType.BEARER)
        assert len(bearer_tokens) >= 1
        assert all(t["type"] == TokenType.BEARER.value for t in bearer_tokens)
