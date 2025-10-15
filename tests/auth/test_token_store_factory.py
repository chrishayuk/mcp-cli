"""Tests for TokenStoreFactory."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mcp_cli.auth.secure_token_store import TokenStorageError
from mcp_cli.auth.token_store_factory import TokenStoreBackend, TokenStoreFactory


class TestTokenStoreBackend:
    """Test TokenStoreBackend enum."""

    def test_backend_values(self):
        """Test TokenStoreBackend enum values."""
        assert TokenStoreBackend.KEYCHAIN == "keychain"
        assert TokenStoreBackend.CREDENTIAL_MANAGER == "windows"
        assert TokenStoreBackend.SECRET_SERVICE == "secretservice"
        assert TokenStoreBackend.VAULT == "vault"
        assert TokenStoreBackend.ENCRYPTED_FILE == "encrypted"
        assert TokenStoreBackend.AUTO == "auto"


class TestTokenStoreFactoryCreate:
    """Test TokenStoreFactory.create method."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Provide temporary directory."""
        return tmp_path / "tokens"

    def test_create_encrypted_file_backend(self, temp_dir):
        """Test creating encrypted file backend."""
        store = TokenStoreFactory.create(
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            token_dir=temp_dir,
            password="test-password",
        )

        from mcp_cli.auth.stores.encrypted_file_store import EncryptedFileTokenStore

        assert isinstance(store, EncryptedFileTokenStore)

    def test_create_auto_backend(self, temp_dir):
        """Test creating backend with AUTO detection."""
        store = TokenStoreFactory.create(
            backend=TokenStoreBackend.AUTO,
            token_dir=temp_dir,
            password="test-password",
        )

        # Should create some backend (likely encrypted file as fallback)
        assert store is not None

    @patch("sys.platform", "darwin")
    def test_create_keychain_backend_on_macos(self):
        """Test creating keychain backend on macOS."""
        try:
            store = TokenStoreFactory.create(
                backend=TokenStoreBackend.KEYCHAIN,
            )

            from mcp_cli.auth.stores.keychain_store import KeychainTokenStore

            assert isinstance(store, KeychainTokenStore)
        except (ImportError, TokenStorageError):
            pytest.skip("keyring library not available")

    def test_create_windows_backend_with_mocking(self, temp_dir):
        """Test creating Windows backend with proper mocking."""
        # Mock the Windows store at its actual import path
        with patch("mcp_cli.auth.stores.windows_store.CredentialManagerTokenStore") as mock_store:
            mock_instance = MagicMock()
            mock_store.return_value = mock_instance

            store = TokenStoreFactory.create(
                backend=TokenStoreBackend.CREDENTIAL_MANAGER,
            )

            assert store == mock_instance
            mock_store.assert_called_once()

    def test_create_linux_backend_with_mocking(self, temp_dir):
        """Test creating Linux backend with proper mocking."""
        # Mock the Linux store at its actual import path
        with patch("mcp_cli.auth.stores.linux_store.SecretServiceTokenStore") as mock_store:
            mock_instance = MagicMock()
            mock_store.return_value = mock_instance

            store = TokenStoreFactory.create(
                backend=TokenStoreBackend.SECRET_SERVICE,
            )

            assert store == mock_instance
            mock_store.assert_called_once()

    def test_create_vault_backend_with_mocking(self, temp_dir):
        """Test creating Vault backend with mocking."""
        with patch("mcp_cli.auth.stores.vault_store.VaultTokenStore") as mock_store:
            mock_instance = MagicMock()
            mock_store.return_value = mock_instance

            store = TokenStoreFactory.create(
                backend=TokenStoreBackend.VAULT,
                vault_url="http://vault:8200",
                vault_token="test-token",
            )

            assert store == mock_instance
            mock_store.assert_called_once_with(
                vault_url="http://vault:8200",
                vault_token="test-token",
                mount_point="secret",
                path_prefix="mcp-cli/oauth",
                namespace=None,
            )

    def test_create_vault_backend_with_all_params(self, temp_dir):
        """Test creating Vault backend with all parameters."""
        with patch("mcp_cli.auth.stores.vault_store.VaultTokenStore") as mock_store:
            mock_instance = MagicMock()
            mock_store.return_value = mock_instance

            store = TokenStoreFactory.create(
                backend=TokenStoreBackend.VAULT,
                vault_url="http://vault:8200",
                vault_token="test-token",
                vault_mount_point="custom-mount",
                vault_path_prefix="custom/prefix",
                vault_namespace="custom-ns",
            )

            assert store == mock_instance
            mock_store.assert_called_once_with(
                vault_url="http://vault:8200",
                vault_token="test-token",
                mount_point="custom-mount",
                path_prefix="custom/prefix",
                namespace="custom-ns",
            )

    def test_create_unknown_backend_raises_error(self, temp_dir):
        """Test that unknown backend raises TokenStorageError then falls back."""
        # Create a custom backend value that will hit the else clause
        class FakeBackend:
            pass

        fake_backend = FakeBackend()

        # This should hit the unknown backend path and fallback to encrypted
        store = TokenStoreFactory.create(
            backend=fake_backend,  # type: ignore
            token_dir=temp_dir,
            password="test-password",
        )

        from mcp_cli.auth.stores.encrypted_file_store import EncryptedFileTokenStore
        assert isinstance(store, EncryptedFileTokenStore)

    def test_create_fallback_to_encrypted_file(self, temp_dir):
        """Test fallback to encrypted file when backend fails."""
        # Mock KeychainTokenStore to raise TokenStorageError
        with patch("mcp_cli.auth.stores.keychain_store.KeychainTokenStore") as mock_store:
            mock_store.side_effect = TokenStorageError("Keychain not available")

            store = TokenStoreFactory.create(
                backend=TokenStoreBackend.KEYCHAIN,
                token_dir=temp_dir,
                password="test-password",
            )

            from mcp_cli.auth.stores.encrypted_file_store import EncryptedFileTokenStore
            assert isinstance(store, EncryptedFileTokenStore)

    def test_create_encrypted_file_failure_raises(self, temp_dir):
        """Test that encrypted file backend failure re-raises error."""
        with patch(
            "mcp_cli.auth.stores.encrypted_file_store.EncryptedFileTokenStore",
            side_effect=TokenStorageError("Encryption failed"),
        ):
            with pytest.raises(TokenStorageError, match="Encryption failed"):
                TokenStoreFactory.create(
                    backend=TokenStoreBackend.ENCRYPTED_FILE,
                    token_dir=temp_dir,
                    password="test-password",
                )


class TestTokenStoreFactoryDetectBackend:
    """Test TokenStoreFactory._detect_backend method."""

    @patch.dict(os.environ, {"VAULT_ADDR": "http://vault:8200", "VAULT_TOKEN": "test"})
    def test_detect_vault_when_env_set(self):
        """Test that Vault is detected when environment variables are set."""
        backend = TokenStoreFactory._detect_backend()
        assert backend == TokenStoreBackend.VAULT

    @patch.dict(os.environ, {"VAULT_ADDR": "http://vault:8200"}, clear=False)
    @patch("platform.system", return_value="Darwin")
    def test_detect_keychain_on_macos_without_vault(self, mock_platform):
        """Test keychain detection on macOS without Vault env."""
        # Clear VAULT_TOKEN if it exists
        if "VAULT_TOKEN" in os.environ:
            del os.environ["VAULT_TOKEN"]

        backend = TokenStoreFactory._detect_backend()
        assert backend == TokenStoreBackend.KEYCHAIN

    @patch("platform.system", return_value="Darwin")
    def test_detect_keychain_on_macos(self, mock_platform):
        """Test keychain detection on macOS."""
        # Clear Vault env vars
        with patch.dict(os.environ, {}, clear=True):
            backend = TokenStoreFactory._detect_backend()
            assert backend == TokenStoreBackend.KEYCHAIN

    @patch("platform.system", return_value="Windows")
    def test_detect_windows_on_windows(self, mock_platform):
        """Test Windows credential manager detection."""
        with patch.dict(os.environ, {}, clear=True):
            backend = TokenStoreFactory._detect_backend()
            assert backend == TokenStoreBackend.CREDENTIAL_MANAGER

    @patch("platform.system", return_value="Linux")
    def test_detect_linux_with_keyring(self, mock_platform):
        """Test Linux secret service detection with keyring available."""
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "SecretServiceKeyring"

        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring", return_value=mock_backend):
                backend = TokenStoreFactory._detect_backend()
                assert backend == TokenStoreBackend.SECRET_SERVICE

    @patch("platform.system", return_value="Linux")
    def test_detect_linux_with_fail_keyring(self, mock_platform):
        """Test Linux falls back when keyring backend fails."""
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "FailKeyring"

        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring", return_value=mock_backend):
                backend = TokenStoreFactory._detect_backend()
                assert backend == TokenStoreBackend.ENCRYPTED_FILE

    @patch("platform.system", return_value="Linux")
    def test_detect_linux_without_keyring(self, mock_platform):
        """Test Linux detection without keyring installed."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring", side_effect=ImportError):
                backend = TokenStoreFactory._detect_backend()
                assert backend == TokenStoreBackend.ENCRYPTED_FILE

    @patch("platform.system", return_value="FreeBSD")
    def test_detect_fallback_on_unknown_platform(self, mock_platform):
        """Test fallback to encrypted file on unknown platform."""
        with patch.dict(os.environ, {}, clear=True):
            backend = TokenStoreFactory._detect_backend()
            assert backend == TokenStoreBackend.ENCRYPTED_FILE


class TestTokenStoreFactoryGetAvailableBackends:
    """Test TokenStoreFactory.get_available_backends method."""

    @patch("platform.system", return_value="Darwin")
    def test_get_available_on_macos_with_keyring(self, mock_platform):
        """Test getting available backends on macOS with keyring."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring"):
                backends = TokenStoreFactory.get_available_backends()

                assert TokenStoreBackend.KEYCHAIN in backends
                assert TokenStoreBackend.ENCRYPTED_FILE in backends

    @patch("platform.system", return_value="Darwin")
    def test_get_available_on_macos_without_keyring(self, mock_platform):
        """Test getting available backends on macOS without keyring."""
        import sys
        with patch.dict(os.environ, {}, clear=True):
            # Temporarily remove keyring from sys.modules
            keyring_backup = sys.modules.get("keyring")
            if "keyring" in sys.modules:
                del sys.modules["keyring"]

            try:
                # Mock import to fail
                with patch.dict(sys.modules, {"keyring": None}):
                    backends = TokenStoreFactory.get_available_backends()

                    # Keychain should not be available without keyring
                    assert TokenStoreBackend.KEYCHAIN not in backends
                    # Encrypted file should still be available
                    assert TokenStoreBackend.ENCRYPTED_FILE in backends
            finally:
                # Restore keyring
                if keyring_backup is not None:
                    sys.modules["keyring"] = keyring_backup

    @patch("platform.system", return_value="Windows")
    def test_get_available_on_windows_with_keyring(self, mock_platform):
        """Test getting available backends on Windows with keyring."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring"):
                backends = TokenStoreFactory.get_available_backends()

                assert TokenStoreBackend.CREDENTIAL_MANAGER in backends
                assert TokenStoreBackend.ENCRYPTED_FILE in backends

    @patch("platform.system", return_value="Windows")
    def test_get_available_on_windows_without_keyring(self, mock_platform):
        """Test getting available backends on Windows without keyring."""
        import sys
        with patch.dict(os.environ, {}, clear=True):
            # Temporarily remove keyring from sys.modules
            keyring_backup = sys.modules.get("keyring")
            if "keyring" in sys.modules:
                del sys.modules["keyring"]

            try:
                # Mock import to fail
                with patch.dict(sys.modules, {"keyring": None}):
                    backends = TokenStoreFactory.get_available_backends()

                    # Credential manager should not be available without keyring
                    assert TokenStoreBackend.CREDENTIAL_MANAGER not in backends
                    # Encrypted file should still be available
                    assert TokenStoreBackend.ENCRYPTED_FILE in backends
            finally:
                # Restore keyring
                if keyring_backup is not None:
                    sys.modules["keyring"] = keyring_backup

    @patch("platform.system", return_value="Linux")
    def test_get_available_on_linux_with_keyring(self, mock_platform):
        """Test getting available backends on Linux with working keyring."""
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "SecretServiceKeyring"

        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring", return_value=mock_backend):
                backends = TokenStoreFactory.get_available_backends()

                assert TokenStoreBackend.SECRET_SERVICE in backends
                assert TokenStoreBackend.ENCRYPTED_FILE in backends

    @patch("platform.system", return_value="Linux")
    def test_get_available_on_linux_with_fail_keyring(self, mock_platform):
        """Test getting available backends on Linux with fail keyring."""
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "FailKeyring"

        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring", return_value=mock_backend):
                backends = TokenStoreFactory.get_available_backends()

                assert TokenStoreBackend.SECRET_SERVICE not in backends
                assert TokenStoreBackend.ENCRYPTED_FILE in backends

    @patch("platform.system", return_value="Linux")
    def test_get_available_on_linux_without_keyring(self, mock_platform):
        """Test getting available backends on Linux without keyring."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("keyring.get_keyring", side_effect=ImportError):
                backends = TokenStoreFactory.get_available_backends()

                assert TokenStoreBackend.SECRET_SERVICE not in backends
                assert TokenStoreBackend.ENCRYPTED_FILE in backends

    @pytest.mark.skip(reason="Requires hvac library to be installed")
    @patch.dict(os.environ, {"VAULT_ADDR": "http://vault:8200", "VAULT_TOKEN": "test"})
    @patch("platform.system", return_value="FreeBSD")  # Use platform without specific backend
    def test_get_available_with_vault(self, mock_platform):
        """Test getting available backends with Vault configured."""
        backends = TokenStoreFactory.get_available_backends()

        assert TokenStoreBackend.VAULT in backends
        assert TokenStoreBackend.ENCRYPTED_FILE in backends

    @patch.dict(os.environ, {"VAULT_ADDR": "http://vault:8200", "VAULT_TOKEN": "test"})
    def test_get_available_with_vault_but_no_hvac(self):
        """Test Vault backend not available without hvac library."""
        import sys
        # Temporarily remove hvac from sys.modules
        hvac_backup = sys.modules.get("hvac")
        if "hvac" in sys.modules:
            del sys.modules["hvac"]

        try:
            # Mock import to fail
            with patch.dict(sys.modules, {"hvac": None}):
                backends = TokenStoreFactory.get_available_backends()

                # Vault should not be in the list
                assert TokenStoreBackend.VAULT not in backends
                # Encrypted file should still be available
                assert TokenStoreBackend.ENCRYPTED_FILE in backends
        finally:
            # Restore hvac
            if hvac_backup is not None:
                sys.modules["hvac"] = hvac_backup

    def test_get_available_without_cryptography(self):
        """Test getting available backends without cryptography."""
        import sys
        with patch.dict(os.environ, {}, clear=True):
            # Temporarily remove cryptography from sys.modules
            crypto_modules_backup = {}
            for key in list(sys.modules.keys()):
                if "cryptography" in key:
                    crypto_modules_backup[key] = sys.modules[key]
                    del sys.modules[key]

            try:
                # Mock import to fail
                crypto_patch = {"cryptography": None, "cryptography.fernet": None}
                with patch.dict(sys.modules, crypto_patch):
                    backends = TokenStoreFactory.get_available_backends()

                    # Encrypted file should not be available
                    assert TokenStoreBackend.ENCRYPTED_FILE not in backends
            finally:
                # Restore cryptography modules
                for key, value in crypto_modules_backup.items():
                    sys.modules[key] = value

    @patch("platform.system", return_value="FreeBSD")
    def test_get_available_minimal_platform(self, mock_platform):
        """Test getting available backends on minimal platform."""
        with patch.dict(os.environ, {}, clear=True):
            backends = TokenStoreFactory.get_available_backends()

            # At minimum, encrypted file should be available
            assert TokenStoreBackend.ENCRYPTED_FILE in backends
            # Platform-specific backends should not be present
            assert TokenStoreBackend.KEYCHAIN not in backends
            assert TokenStoreBackend.CREDENTIAL_MANAGER not in backends
            assert TokenStoreBackend.SECRET_SERVICE not in backends


class TestTokenStoreFactoryIntegration:
    """Integration tests for TokenStoreFactory."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Provide temporary directory."""
        return tmp_path / "tokens"

    def test_factory_creates_working_store(self, temp_dir):
        """Test that factory creates a working store."""
        from mcp_cli.auth.oauth_config import OAuthTokens

        store = TokenStoreFactory.create(
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            token_dir=temp_dir,
            password="test-password",
        )

        # Test basic operations
        tokens = OAuthTokens(
            access_token="test-token",
            token_type="Bearer",
            expires_in=3600,
        )

        store.store_token("test-server", tokens)
        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
        assert retrieved.access_token == tokens.access_token

    def test_factory_with_auto_detection(self, temp_dir):
        """Test factory with auto-detection creates working store."""
        from mcp_cli.auth.oauth_config import OAuthTokens

        store = TokenStoreFactory.create(
            backend=TokenStoreBackend.AUTO,
            token_dir=temp_dir,
            password="test-password",
        )

        # Should create a working store
        tokens = OAuthTokens(access_token="test-token", token_type="Bearer")
        store.store_token("test-server", tokens)
        retrieved = store.retrieve_token("test-server")

        assert retrieved is not None
