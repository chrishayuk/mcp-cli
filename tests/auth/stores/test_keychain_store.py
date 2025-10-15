"""Tests for KeychainTokenStore."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.secure_token_store import TokenStorageError
from mcp_cli.auth.stores.keychain_store import KeychainTokenStore


class TestKeychainTokenStoreInit:
    """Test KeychainTokenStore initialization."""

    @patch("platform.system", return_value="Darwin")
    def test_init_on_macos_with_keyring(self, mock_platform):
        """Test initialization on macOS with keyring available."""
        mock_keyring = MagicMock()
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            store = KeychainTokenStore()
            assert store.keyring == mock_keyring
            assert store.SERVICE_NAME == "mcp-cli-oauth"

    @patch("platform.system", return_value="Linux")
    def test_init_on_non_macos_raises_error(self, mock_platform):
        """Test initialization on non-macOS raises error."""
        with pytest.raises(
            TokenStorageError, match="Keychain storage is only available on macOS"
        ):
            KeychainTokenStore()

    @patch("platform.system", return_value="Darwin")
    def test_init_without_keyring_raises_error(self, mock_platform):
        """Test initialization without keyring raises error."""
        with patch.dict("sys.modules", {"keyring": None}):
            with pytest.raises(
                TokenStorageError, match="keyring library not installed"
            ):
                KeychainTokenStore()


class TestKeychainTokenStoreOperations:
    """Test KeychainTokenStore operations."""

    @pytest.fixture
    def mock_keyring(self):
        """Provide mock keyring."""
        mock = MagicMock()
        mock.set_password = MagicMock()
        mock.get_password = MagicMock()
        mock.delete_password = MagicMock()
        return mock

    @pytest.fixture
    def store(self, mock_keyring):
        """Provide KeychainTokenStore with mocked keyring."""
        with patch("platform.system", return_value="Darwin"):
            with patch.dict("sys.modules", {"keyring": mock_keyring}):
                store = KeychainTokenStore()
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

        # Verify keyring.set_password was called
        store.keyring.set_password.assert_called_once()
        service, username, password = store.keyring.set_password.call_args[0]

        assert service == "mcp-cli-oauth"
        assert username == "test-server"

        # Verify the token was serialized
        token_data = json.loads(password)
        assert token_data["access_token"] == "test-access-token"
        assert token_data["refresh_token"] == "test-refresh-token"
        assert "issued_at" in token_data

    def test_store_token_adds_issued_at(self, store, sample_tokens):
        """Test storing tokens adds issued_at if missing."""
        # Ensure issued_at is None
        sample_tokens.issued_at = None

        store.store_token("test-server", sample_tokens)

        # Get the stored JSON
        password = store.keyring.set_password.call_args[0][2]
        token_data = json.loads(password)

        assert "issued_at" in token_data
        assert token_data["issued_at"] is not None

    def test_store_token_error_handling(self, store, sample_tokens):
        """Test error handling when storing token."""
        store.keyring.set_password.side_effect = Exception("Keyring error")

        with pytest.raises(
            TokenStorageError, match="Failed to store token in Keychain"
        ):
            store.store_token("test-server", sample_tokens)

    def test_retrieve_token(self, store, sample_tokens):
        """Test retrieving OAuth tokens."""
        # Mock get_password to return token JSON
        token_json = json.dumps(sample_tokens.to_dict())
        store.keyring.get_password.return_value = token_json

        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
        assert retrieved.access_token == sample_tokens.access_token
        assert retrieved.refresh_token == sample_tokens.refresh_token

        store.keyring.get_password.assert_called_once_with(
            "mcp-cli-oauth", "test-server"
        )

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
        store.keyring.get_password.side_effect = Exception("Keyring error")

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve token from Keychain"
        ):
            store.retrieve_token("test-server")

    def test_delete_token(self, store):
        """Test deleting OAuth tokens."""
        # Mock that token exists
        store.keyring.get_password.return_value = "some-token"

        result = store.delete_token("test-server")

        assert result is True
        store.keyring.delete_password.assert_called_once_with(
            "mcp-cli-oauth", "test-server"
        )

    def test_delete_token_nonexistent(self, store):
        """Test deleting nonexistent token."""
        store.keyring.get_password.return_value = None

        result = store.delete_token("nonexistent")

        assert result is False
        store.keyring.delete_password.assert_not_called()

    def test_delete_token_error_handling(self, store):
        """Test error handling when deleting token."""
        store.keyring.get_password.return_value = "some-token"
        store.keyring.delete_password.side_effect = Exception("Keyring error")

        with pytest.raises(
            TokenStorageError, match="Failed to delete token from Keychain"
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
        store.keyring.get_password.side_effect = Exception("Keyring error")

        result = store.has_token("test-server")

        assert result is False


class TestKeychainTokenStoreRawOperations:
    """Test KeychainTokenStore raw storage operations."""

    @pytest.fixture
    def store(self):
        """Provide KeychainTokenStore with mocked keyring."""
        mock_keyring = MagicMock()
        with patch("platform.system", return_value="Darwin"):
            with patch.dict("sys.modules", {"keyring": mock_keyring}):
                store = KeychainTokenStore()
                return store

    def test_store_raw(self, store):
        """Test storing raw value."""
        store._store_raw("test-key", "test-value")

        store.keyring.set_password.assert_called_once_with(
            "mcp-cli-oauth", "test-key", "test-value"
        )

    def test_store_raw_error_handling(self, store):
        """Test error handling when storing raw value."""
        store.keyring.set_password.side_effect = Exception("Keyring error")

        with pytest.raises(
            TokenStorageError, match="Failed to store value in Keychain"
        ):
            store._store_raw("test-key", "test-value")

    def test_retrieve_raw(self, store):
        """Test retrieving raw value."""
        store.keyring.get_password.return_value = "test-value"

        result = store._retrieve_raw("test-key")

        assert result == "test-value"
        store.keyring.get_password.assert_called_once_with("mcp-cli-oauth", "test-key")

    def test_retrieve_raw_error_handling(self, store):
        """Test error handling when retrieving raw value."""
        store.keyring.get_password.side_effect = Exception("Keyring error")

        with pytest.raises(
            TokenStorageError, match="Failed to retrieve value from Keychain"
        ):
            store._retrieve_raw("test-key")

    def test_delete_raw(self, store):
        """Test deleting raw value."""
        store.keyring.get_password.return_value = "test-value"

        result = store._delete_raw("test-key")

        assert result is True
        store.keyring.delete_password.assert_called_once_with(
            "mcp-cli-oauth", "test-key"
        )

    def test_delete_raw_nonexistent(self, store):
        """Test deleting nonexistent raw value."""
        store.keyring.get_password.return_value = None

        result = store._delete_raw("nonexistent")

        assert result is False
        store.keyring.delete_password.assert_not_called()

    def test_delete_raw_error_handling(self, store):
        """Test error handling when deleting raw value."""
        store.keyring.get_password.return_value = "test-value"
        store.keyring.delete_password.side_effect = Exception("Keyring error")

        with pytest.raises(
            TokenStorageError, match="Failed to delete value from Keychain"
        ):
            store._delete_raw("test-key")


class TestKeychainTokenStoreNameSanitization:
    """Test name sanitization in KeychainTokenStore."""

    @pytest.fixture
    def store(self):
        """Provide KeychainTokenStore with mocked keyring."""
        mock_keyring = MagicMock()
        with patch("platform.system", return_value="Darwin"):
            with patch.dict("sys.modules", {"keyring": mock_keyring}):
                return KeychainTokenStore()

    def test_sanitize_special_characters(self, store, sample_tokens=None):
        """Test that special characters are sanitized."""
        if sample_tokens is None:
            sample_tokens = OAuthTokens(
                access_token="test",
                token_type="Bearer",
                expires_in=3600,
            )

        # Store with special characters in name
        store.store_token("test:server@domain", sample_tokens)

        # Check that sanitized name was used
        call_args = store.keyring.set_password.call_args[0]
        assert call_args[1] == "test:server_domain"  # @ replaced with _
