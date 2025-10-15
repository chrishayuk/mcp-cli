"""Tests for SecretServiceTokenStore."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.secure_token_store import TokenStorageError
from mcp_cli.auth.stores.linux_store import SecretServiceTokenStore


class TestSecretServiceTokenStoreInit:
    """Test SecretServiceTokenStore initialization."""

    @patch("platform.system", return_value="Linux")
    def test_init_on_linux_with_keyring(self, mock_platform):
        """Test initialization on Linux with keyring available."""
        mock_keyring = MagicMock()
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "SecretServiceKeyring"
        mock_keyring.get_keyring.return_value = mock_backend

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = SecretServiceTokenStore()
            assert store.keyring == mock_keyring
            assert store.SERVICE_NAME == "mcp-cli-oauth"

    @patch("platform.system", return_value="Darwin")
    def test_init_on_non_linux_raises_error(self, mock_platform):
        """Test initialization on non-Linux raises error."""
        with pytest.raises(
            TokenStorageError,
            match="Secret Service storage is only available on Linux",
        ):
            SecretServiceTokenStore()

    @patch("platform.system", return_value="Linux")
    def test_init_without_keyring_raises_error(self, mock_platform):
        """Test initialization without keyring raises error."""
        with patch.dict("sys.modules", {"keyring": None}):
            with pytest.raises(
                TokenStorageError, match="keyring library not installed"
            ):
                SecretServiceTokenStore()

    @patch("platform.system", return_value="Linux")
    def test_init_with_fail_backend_raises_error(self, mock_platform):
        """Test initialization with fail backend raises error."""
        mock_keyring = MagicMock()
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "FailKeyring"
        mock_keyring.get_keyring.return_value = mock_backend

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            with pytest.raises(
                TokenStorageError, match="No keyring backend available"
            ):
                SecretServiceTokenStore()


class TestSecretServiceTokenStoreOperations:
    """Test SecretServiceTokenStore operations."""

    @pytest.fixture
    def store(self):
        """Provide SecretServiceTokenStore with mocked keyring."""
        mock_keyring = MagicMock()
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "SecretServiceKeyring"
        mock_keyring.get_keyring.return_value = mock_backend

        with patch("platform.system", return_value="Linux"):
            with patch.dict("sys.modules", {"keyring": mock_keyring}):
                store = SecretServiceTokenStore()
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

    def test_store_token(self, store, sample_tokens):
        """Test storing OAuth tokens."""
        store.store_token("test-server", sample_tokens)

        store.keyring.set_password.assert_called_once()
        service, username, password = store.keyring.set_password.call_args[0]

        assert service == "mcp-cli-oauth"
        assert username == "test-server"

        token_data = json.loads(password)
        assert token_data["access_token"] == "test-access-token"
        assert "issued_at" in token_data

    def test_store_token_error_handling(self, store, sample_tokens):
        """Test error handling when storing token."""
        store.keyring.set_password.side_effect = Exception("Service error")

        with pytest.raises(
            TokenStorageError, match="Failed to store token in Secret Service"
        ):
            store.store_token("test-server", sample_tokens)

    def test_retrieve_token(self, store, sample_tokens):
        """Test retrieving OAuth tokens."""
        token_json = json.dumps(sample_tokens.to_dict())
        store.keyring.get_password.return_value = token_json

        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
        assert retrieved.access_token == sample_tokens.access_token

    def test_retrieve_token_nonexistent(self, store):
        """Test retrieving nonexistent token."""
        store.keyring.get_password.return_value = None

        result = store.retrieve_token("nonexistent")

        assert result is None

    def test_retrieve_token_json_decode_error(self, store):
        """Test error when token data is invalid JSON."""
        store.keyring.get_password.return_value = "invalid json {"

        with pytest.raises(TokenStorageError, match="Failed to parse token data"):
            store.retrieve_token("test-server")

    def test_retrieve_token_error_handling(self, store):
        """Test error handling when retrieving token."""
        store.keyring.get_password.side_effect = Exception("Service error")

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve token from Secret Service"
        ):
            store.retrieve_token("test-server")

    def test_delete_token(self, store):
        """Test deleting OAuth tokens."""
        store.keyring.get_password.return_value = "some-token"

        result = store.delete_token("test-server")

        assert result is True
        store.keyring.delete_password.assert_called_once()

    def test_delete_token_nonexistent(self, store):
        """Test deleting nonexistent token."""
        store.keyring.get_password.return_value = None

        result = store.delete_token("nonexistent")

        assert result is False
        store.keyring.delete_password.assert_not_called()

    def test_delete_token_error_handling(self, store):
        """Test error handling when deleting token."""
        store.keyring.get_password.return_value = "some-token"
        store.keyring.delete_password.side_effect = Exception("Service error")

        with pytest.raises(
            TokenStorageError, match="Failed to delete token from Secret Service"
        ):
            store.delete_token("test-server")

    def test_has_token_exists(self, store):
        """Test checking if token exists."""
        store.keyring.get_password.return_value = "some-token"

        result = store.has_token("test-server")

        assert result is True

    def test_has_token_not_exists(self, store):
        """Test checking if token doesn't exist."""
        store.keyring.get_password.return_value = None

        result = store.has_token("nonexistent")

        assert result is False

    def test_has_token_error_handling(self, store):
        """Test error handling in has_token returns False."""
        store.keyring.get_password.side_effect = Exception("Service error")

        result = store.has_token("test-server")

        assert result is False

    def test_store_raw(self, store):
        """Test storing raw value."""
        store._store_raw("test-key", "test-value")

        store.keyring.set_password.assert_called_once_with(
            "mcp-cli-oauth", "test-key", "test-value"
        )

    def test_store_raw_error_handling(self, store):
        """Test error handling when storing raw value."""
        store.keyring.set_password.side_effect = Exception("Service error")

        with pytest.raises(
            TokenStorageError, match="Failed to store value in Secret Service"
        ):
            store._store_raw("test-key", "test-value")

    def test_retrieve_raw(self, store):
        """Test retrieving raw value."""
        store.keyring.get_password.return_value = "test-value"

        result = store._retrieve_raw("test-key")

        assert result == "test-value"

    def test_retrieve_raw_error_handling(self, store):
        """Test error handling when retrieving raw value."""
        store.keyring.get_password.side_effect = Exception("Service error")

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve value from Secret Service"
        ):
            store._retrieve_raw("test-key")

    def test_delete_raw(self, store):
        """Test deleting raw value."""
        store.keyring.get_password.return_value = "test-value"

        result = store._delete_raw("test-key")

        assert result is True
        store.keyring.delete_password.assert_called_once()

    def test_delete_raw_nonexistent(self, store):
        """Test deleting nonexistent raw value."""
        store.keyring.get_password.return_value = None

        result = store._delete_raw("nonexistent")

        assert result is False
        store.keyring.delete_password.assert_not_called()

    def test_delete_raw_error_handling(self, store):
        """Test error handling when deleting raw value."""
        store.keyring.get_password.return_value = "test-value"
        store.keyring.delete_password.side_effect = Exception("Service error")

        with pytest.raises(
            TokenStorageError, match="Failed to delete value from Secret Service"
        ):
            store._delete_raw("test-key")
