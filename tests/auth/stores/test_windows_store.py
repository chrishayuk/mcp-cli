"""Tests for CredentialManagerTokenStore."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.secure_token_store import TokenStorageError
from mcp_cli.auth.stores.windows_store import CredentialManagerTokenStore


class TestCredentialManagerTokenStoreInit:
    """Test CredentialManagerTokenStore initialization."""

    @patch("platform.system", return_value="Windows")
    def test_init_on_windows_with_keyring(self, mock_platform):
        """Test initialization on Windows with keyring available."""
        mock_keyring = MagicMock()
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = CredentialManagerTokenStore()
            assert store.keyring == mock_keyring
            assert store.SERVICE_NAME == "mcp-cli-oauth"

    @patch("platform.system", return_value="Darwin")
    def test_init_on_non_windows_raises_error(self, mock_platform):
        """Test initialization on non-Windows raises error."""
        with pytest.raises(
            TokenStorageError,
            match="Windows Credential Manager is only available on Windows",
        ):
            CredentialManagerTokenStore()

    @patch("platform.system", return_value="Windows")
    def test_init_without_keyring_raises_error(self, mock_platform):
        """Test initialization without keyring raises error."""
        with patch.dict("sys.modules", {"keyring": None}):
            with pytest.raises(
                TokenStorageError, match="keyring library not installed"
            ):
                CredentialManagerTokenStore()


class TestCredentialManagerTokenStoreOperations:
    """Test CredentialManagerTokenStore operations."""

    @pytest.fixture
    def store(self):
        """Provide CredentialManagerTokenStore with mocked keyring."""
        mock_keyring = MagicMock()
        with patch("platform.system", return_value="Windows"):
            with patch.dict("sys.modules", {"keyring": mock_keyring}):
                store = CredentialManagerTokenStore()
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
        store.keyring.set_password.side_effect = Exception("Credential Manager error")

        with pytest.raises(
            TokenStorageError, match="Failed to store token in Credential Manager"
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
        store.keyring.get_password.side_effect = Exception("Credential Manager error")

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve token from Credential Manager"
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
        store.keyring.delete_password.side_effect = Exception(
            "Credential Manager error"
        )

        with pytest.raises(
            TokenStorageError, match="Failed to delete token from Credential Manager"
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
        store.keyring.get_password.side_effect = Exception("Credential Manager error")

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
        store.keyring.set_password.side_effect = Exception("Credential Manager error")

        with pytest.raises(
            TokenStorageError, match="Failed to store value in Credential Manager"
        ):
            store._store_raw("test-key", "test-value")

    def test_retrieve_raw(self, store):
        """Test retrieving raw value."""
        store.keyring.get_password.return_value = "test-value"

        result = store._retrieve_raw("test-key")

        assert result == "test-value"

    def test_retrieve_raw_error_handling(self, store):
        """Test error handling when retrieving raw value."""
        store.keyring.get_password.side_effect = Exception("Credential Manager error")

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve value from Credential Manager"
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
        store.keyring.delete_password.side_effect = Exception(
            "Credential Manager error"
        )

        with pytest.raises(
            TokenStorageError, match="Failed to delete value from Credential Manager"
        ):
            store._delete_raw("test-key")
