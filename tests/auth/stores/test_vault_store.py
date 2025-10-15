"""Tests for VaultTokenStore."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.secure_token_store import TokenStorageError
from mcp_cli.auth.stores.vault_store import VaultTokenStore


class TestVaultTokenStoreInit:
    """Test VaultTokenStore initialization."""

    @pytest.fixture
    def mock_hvac(self):
        """Provide mock hvac module."""
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client
        mock_hvac.exceptions = Mock()
        mock_hvac.exceptions.InvalidPath = type("InvalidPath", (Exception,), {})
        return mock_hvac

    def test_init_with_parameters(self, mock_hvac):
        """Test initialization with parameters."""
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            store = VaultTokenStore(
                vault_url="http://vault:8200",
                vault_token="test-token",
                mount_point="custom-mount",
                path_prefix="custom/prefix",
                namespace="custom-ns",
            )

            assert store.vault_url == "http://vault:8200"
            assert store.vault_token == "test-token"
            assert store.mount_point == "custom-mount"
            assert store.path_prefix == "custom/prefix"
            assert store.namespace == "custom-ns"

    @patch.dict(
        os.environ, {"VAULT_ADDR": "http://env-vault:8200", "VAULT_TOKEN": "env-token"}
    )
    def test_init_with_env_vars(self, mock_hvac):
        """Test initialization with environment variables."""
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            store = VaultTokenStore()

            assert store.vault_url == "http://env-vault:8200"
            assert store.vault_token == "env-token"

    def test_init_without_hvac_raises_error(self):
        """Test initialization without hvac raises error."""
        with patch.dict("sys.modules", {"hvac": None}):
            with pytest.raises(TokenStorageError, match="hvac library not installed"):
                VaultTokenStore(vault_url="http://vault:8200", vault_token="test")

    def test_init_without_url_raises_error(self, mock_hvac):
        """Test initialization without URL raises error."""
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            with pytest.raises(TokenStorageError, match="Vault URL not provided"):
                VaultTokenStore(vault_token="test-token")

    def test_init_without_token_raises_error(self, mock_hvac):
        """Test initialization without token raises error."""
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            with pytest.raises(TokenStorageError, match="Vault token not provided"):
                VaultTokenStore(vault_url="http://vault:8200")

    def test_init_authentication_failure(self, mock_hvac):
        """Test initialization with authentication failure."""
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = False
        mock_hvac.Client.return_value = mock_client

        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            with pytest.raises(TokenStorageError, match="Vault authentication failed"):
                VaultTokenStore(vault_url="http://vault:8200", vault_token="bad-token")

    def test_init_client_creation_failure(self, mock_hvac):
        """Test initialization with client creation failure."""
        mock_hvac.Client.side_effect = Exception("Connection error")

        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            with pytest.raises(
                TokenStorageError, match="Failed to initialize Vault client"
            ):
                VaultTokenStore(vault_url="http://vault:8200", vault_token="test-token")


class TestVaultTokenStoreOperations:
    """Test VaultTokenStore operations."""

    @pytest.fixture
    def mock_hvac(self):
        """Provide mock hvac module."""
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client
        mock_hvac.exceptions = Mock()
        mock_hvac.exceptions.InvalidPath = type("InvalidPath", (Exception,), {})
        return mock_hvac

    @pytest.fixture
    def store(self, mock_hvac):
        """Provide VaultTokenStore with mocked hvac."""
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            store = VaultTokenStore(
                vault_url="http://vault:8200",
                vault_token="test-token",
            )
            return store

    @pytest.fixture
    def sample_tokens(self):
        """Provide sample OAuth tokens."""
        return OAuthTokens(
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            expires_in=3600,
            token_type="Bearer",
        )

    def test_get_vault_path(self, store):
        """Test getting Vault path for server."""
        path = store._get_vault_path("test-server")
        assert path == "mcp-cli/oauth/test-server"

    def test_get_vault_path_special_chars(self, store):
        """Test getting Vault path with special characters."""
        path = store._get_vault_path("test:server@domain")
        # Should be sanitized
        assert "mcp-cli/oauth/" in path

    def test_store_token_kv_v2(self, store, sample_tokens):
        """Test storing token in Vault KV v2."""
        store.store_token("test-server", sample_tokens)

        # Verify KV v2 method was called
        store.client.secrets.kv.v2.create_or_update_secret.assert_called_once()

        call_args = store.client.secrets.kv.v2.create_or_update_secret.call_args[1]
        assert call_args["path"] == "mcp-cli/oauth/test-server"
        assert call_args["mount_point"] == "secret"
        assert "access_token" in call_args["secret"]
        assert call_args["secret"]["access_token"] == "test-access-token"
        assert "issued_at" in call_args["secret"]

    def test_store_token_kv_v1_fallback(self, store, sample_tokens, mock_hvac):
        """Test storing token falls back to KV v1."""
        # Make v2 raise InvalidPath
        store.client.secrets.kv.v2.create_or_update_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        store.store_token("test-server", sample_tokens)

        # Verify v1 method was called
        store.client.secrets.kv.v1.create_or_update_secret.assert_called_once()

    def test_store_token_error_handling(self, store, sample_tokens):
        """Test error handling when storing token."""
        store.client.secrets.kv.v2.create_or_update_secret.side_effect = Exception(
            "Vault error"
        )

        with pytest.raises(TokenStorageError, match="Failed to store token in Vault"):
            store.store_token("test-server", sample_tokens)

    def test_retrieve_token_kv_v2(self, store, sample_tokens):
        """Test retrieving token from Vault KV v2."""
        # Mock response
        mock_response = {"data": {"data": sample_tokens.to_dict()}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
        assert retrieved.access_token == sample_tokens.access_token

    def test_retrieve_token_kv_v1_fallback(self, store, sample_tokens, mock_hvac):
        """Test retrieving token falls back to KV v1."""
        # Make v2 raise InvalidPath
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        # Mock v1 response
        mock_response = {"data": sample_tokens.to_dict()}
        store.client.secrets.kv.v1.read_secret.return_value = mock_response

        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
        assert retrieved.access_token == sample_tokens.access_token

    def test_retrieve_token_nonexistent(self, store, mock_hvac):
        """Test retrieving nonexistent token."""
        # Make both v2 and v1 raise InvalidPath
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )
        store.client.secrets.kv.v1.read_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store.retrieve_token("nonexistent")

        assert result is None

    def test_retrieve_token_error_handling(self, store):
        """Test error handling when retrieving token."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = Exception(
            "Vault error"
        )

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve token from Vault"
        ):
            store.retrieve_token("test-server")

    def test_delete_token_kv_v2(self, store, mock_hvac):
        """Test deleting token from Vault KV v2."""
        # Mock has_token to return True
        mock_response = {"data": {"data": {}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        result = store.delete_token("test-server")

        assert result is True
        store.client.secrets.kv.v2.delete_metadata_and_all_versions.assert_called_once()

    def test_delete_token_kv_v1_fallback(self, store, mock_hvac):
        """Test deleting token falls back to KV v1."""
        # Mock has_token to return True
        mock_response = {"data": {"data": {}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        # Make v2 delete raise InvalidPath
        store.client.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store.delete_token("test-server")

        assert result is True
        store.client.secrets.kv.v1.delete_secret.assert_called_once()

    def test_delete_token_nonexistent(self, store, mock_hvac):
        """Test deleting nonexistent token."""
        # Mock has_token to return False
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )
        store.client.secrets.kv.v1.read_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store.delete_token("nonexistent")

        assert result is False

    def test_delete_token_error_handling(self, store):
        """Test error handling when deleting token."""
        # Mock has_token to return True
        mock_response = {"data": {"data": {}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        store.client.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = (
            Exception("Vault error")
        )

        with pytest.raises(
            TokenStorageError, match="Failed to delete token from Vault"
        ):
            store.delete_token("test-server")

    def test_has_token_exists_kv_v2(self, store):
        """Test checking if token exists in KV v2."""
        mock_response = {"data": {"data": {}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        result = store.has_token("test-server")

        assert result is True

    def test_has_token_exists_kv_v1(self, store, mock_hvac):
        """Test checking if token exists in KV v1."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        mock_response = {"data": {}}
        store.client.secrets.kv.v1.read_secret.return_value = mock_response

        result = store.has_token("test-server")

        assert result is True

    def test_has_token_not_exists(self, store, mock_hvac):
        """Test checking if token doesn't exist."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )
        store.client.secrets.kv.v1.read_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store.has_token("nonexistent")

        assert result is False

    def test_has_token_error_handling(self, store):
        """Test error handling in has_token returns False."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = Exception(
            "Vault error"
        )

        result = store.has_token("test-server")

        assert result is False


class TestVaultTokenStoreRawOperations:
    """Test VaultTokenStore raw storage operations."""

    @pytest.fixture
    def mock_hvac(self):
        """Provide mock hvac module."""
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client
        mock_hvac.exceptions = Mock()
        mock_hvac.exceptions.InvalidPath = type("InvalidPath", (Exception,), {})
        return mock_hvac

    @pytest.fixture
    def store(self, mock_hvac):
        """Provide VaultTokenStore with mocked hvac."""
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            store = VaultTokenStore(
                vault_url="http://vault:8200",
                vault_token="test-token",
            )
            return store

    def test_store_raw_kv_v2(self, store):
        """Test storing raw value in KV v2."""
        store._store_raw("test-key", "test-value")

        store.client.secrets.kv.v2.create_or_update_secret.assert_called_once()
        call_args = store.client.secrets.kv.v2.create_or_update_secret.call_args[1]
        assert call_args["secret"] == {"value": "test-value"}

    def test_store_raw_kv_v1_fallback(self, store, mock_hvac):
        """Test storing raw value falls back to KV v1."""
        store.client.secrets.kv.v2.create_or_update_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        store._store_raw("test-key", "test-value")

        store.client.secrets.kv.v1.create_or_update_secret.assert_called_once()

    def test_store_raw_error_handling(self, store):
        """Test error handling when storing raw value."""
        store.client.secrets.kv.v2.create_or_update_secret.side_effect = Exception(
            "Vault error"
        )

        with pytest.raises(TokenStorageError, match="Failed to store value in Vault"):
            store._store_raw("test-key", "test-value")

    def test_retrieve_raw_kv_v2(self, store):
        """Test retrieving raw value from KV v2."""
        mock_response = {"data": {"data": {"value": "test-value"}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        result = store._retrieve_raw("test-key")

        assert result == "test-value"

    def test_retrieve_raw_kv_v1_fallback(self, store, mock_hvac):
        """Test retrieving raw value falls back to KV v1."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        mock_response = {"data": {"value": "test-value"}}
        store.client.secrets.kv.v1.read_secret.return_value = mock_response

        result = store._retrieve_raw("test-key")

        assert result == "test-value"

    def test_retrieve_raw_nonexistent(self, store, mock_hvac):
        """Test retrieving nonexistent raw value."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )
        store.client.secrets.kv.v1.read_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store._retrieve_raw("nonexistent")

        assert result is None

    def test_retrieve_raw_error_handling(self, store):
        """Test error handling when retrieving raw value."""
        store.client.secrets.kv.v2.read_secret_version.side_effect = Exception(
            "Vault error"
        )

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve value from Vault"
        ):
            store._retrieve_raw("test-key")

    def test_delete_raw_kv_v2(self, store):
        """Test deleting raw value from KV v2."""
        # Mock _retrieve_raw to return value
        mock_response = {"data": {"data": {"value": "test-value"}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        result = store._delete_raw("test-key")

        assert result is True
        store.client.secrets.kv.v2.delete_metadata_and_all_versions.assert_called_once()

    def test_delete_raw_kv_v1_fallback(self, store, mock_hvac):
        """Test deleting raw value falls back to KV v1."""
        # Mock _retrieve_raw to return value
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )
        mock_response = {"data": {"value": "test-value"}}
        store.client.secrets.kv.v1.read_secret.return_value = mock_response

        # Make v2 delete raise InvalidPath
        store.client.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store._delete_raw("test-key")

        assert result is True
        store.client.secrets.kv.v1.delete_secret.assert_called_once()

    def test_delete_raw_nonexistent(self, store, mock_hvac):
        """Test deleting nonexistent raw value."""
        # Mock _retrieve_raw to return None
        store.client.secrets.kv.v2.read_secret_version.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )
        store.client.secrets.kv.v1.read_secret.side_effect = (
            mock_hvac.exceptions.InvalidPath
        )

        result = store._delete_raw("nonexistent")

        assert result is False

    def test_delete_raw_error_handling(self, store):
        """Test error handling when deleting raw value."""
        # Mock _retrieve_raw to return value
        mock_response = {"data": {"data": {"value": "test-value"}}}
        store.client.secrets.kv.v2.read_secret_version.return_value = mock_response

        store.client.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = (
            Exception("Vault error")
        )

        with pytest.raises(
            TokenStorageError, match="Failed to delete value from Vault"
        ):
            store._delete_raw("test-key")
