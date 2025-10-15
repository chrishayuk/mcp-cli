"""Factory for creating appropriate token store backend."""

import os
import platform
from enum import Enum
from pathlib import Path
from typing import Optional

from .secure_token_store import SecureTokenStore, TokenStorageError


class TokenStoreBackend(str, Enum):
    """Available token storage backends."""

    KEYCHAIN = "keychain"  # macOS Keychain
    CREDENTIAL_MANAGER = "windows"  # Windows Credential Manager
    SECRET_SERVICE = "secretservice"  # Linux Secret Service (GNOME/KDE)
    VAULT = "vault"  # HashiCorp Vault
    ENCRYPTED_FILE = "encrypted"  # Encrypted file (fallback)
    AUTO = "auto"  # Auto-detect best available backend


class TokenStoreFactory:
    """Factory for creating token store backends."""

    @staticmethod
    def create(
        backend: TokenStoreBackend = TokenStoreBackend.AUTO,
        token_dir: Optional[Path] = None,
        password: Optional[str] = None,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        vault_mount_point: str = "secret",
        vault_path_prefix: str = "mcp-cli/oauth",
        vault_namespace: Optional[str] = None,
    ) -> SecureTokenStore:
        """
        Create appropriate token store backend.

        Args:
            backend: Backend to use (default: AUTO for auto-detection)
            token_dir: Directory for file-based storage
            password: Password for encrypted file storage
            vault_url: HashiCorp Vault URL
            vault_token: HashiCorp Vault token
            vault_mount_point: Vault KV mount point
            vault_path_prefix: Vault path prefix for tokens
            vault_namespace: Vault namespace (Enterprise)

        Returns:
            Configured token store backend

        Raises:
            TokenStorageError: If backend creation fails
        """
        if backend == TokenStoreBackend.AUTO:
            backend = TokenStoreFactory._detect_backend()

        # Try to create the requested backend
        try:
            if backend == TokenStoreBackend.KEYCHAIN:
                from .stores.keychain_store import KeychainTokenStore

                return KeychainTokenStore()

            elif backend == TokenStoreBackend.CREDENTIAL_MANAGER:
                from .stores.windows_store import CredentialManagerTokenStore

                return CredentialManagerTokenStore()

            elif backend == TokenStoreBackend.SECRET_SERVICE:
                from .stores.linux_store import SecretServiceTokenStore

                return SecretServiceTokenStore()

            elif backend == TokenStoreBackend.VAULT:
                from .stores.vault_store import VaultTokenStore

                return VaultTokenStore(
                    vault_url=vault_url,
                    vault_token=vault_token,
                    mount_point=vault_mount_point,
                    path_prefix=vault_path_prefix,
                    namespace=vault_namespace,
                )

            elif backend == TokenStoreBackend.ENCRYPTED_FILE:
                from .stores.encrypted_file_store import EncryptedFileTokenStore

                return EncryptedFileTokenStore(token_dir=token_dir, password=password)

            else:
                raise TokenStorageError(f"Unknown backend: {backend}")

        except TokenStorageError as e:
            # If the requested backend fails and we're in auto mode,
            # fall back to encrypted file storage
            if backend != TokenStoreBackend.ENCRYPTED_FILE:
                from .stores.encrypted_file_store import EncryptedFileTokenStore

                return EncryptedFileTokenStore(token_dir=token_dir, password=password)
            else:
                raise e

    @staticmethod
    def _detect_backend() -> TokenStoreBackend:
        """
        Auto-detect best available backend for current platform.

        Returns:
            Recommended backend for current platform
        """
        system = platform.system()

        # Check for Vault configuration first (cross-platform)
        if os.getenv("VAULT_ADDR") and os.getenv("VAULT_TOKEN"):
            return TokenStoreBackend.VAULT

        # Platform-specific defaults
        if system == "Darwin":
            return TokenStoreBackend.KEYCHAIN
        elif system == "Windows":
            return TokenStoreBackend.CREDENTIAL_MANAGER
        elif system == "Linux":
            # Check if a keyring backend is available
            try:
                import keyring

                backend = keyring.get_keyring()
                if "fail" not in backend.__class__.__name__.lower():
                    return TokenStoreBackend.SECRET_SERVICE
            except ImportError:
                pass

        # Fallback to encrypted file storage
        return TokenStoreBackend.ENCRYPTED_FILE

    @staticmethod
    def get_available_backends() -> list[TokenStoreBackend]:
        """
        Get list of available backends on current platform.

        Returns:
            List of backends that can be initialized
        """
        available = []
        system = platform.system()

        # Check platform-specific backends
        if system == "Darwin":
            try:
                import keyring

                available.append(TokenStoreBackend.KEYCHAIN)
            except ImportError:
                pass

        elif system == "Windows":
            try:
                import keyring

                available.append(TokenStoreBackend.CREDENTIAL_MANAGER)
            except ImportError:
                pass

        elif system == "Linux":
            try:
                import keyring

                backend = keyring.get_keyring()
                if "fail" not in backend.__class__.__name__.lower():
                    available.append(TokenStoreBackend.SECRET_SERVICE)
            except ImportError:
                pass

        # Check for Vault
        if os.getenv("VAULT_ADDR") and os.getenv("VAULT_TOKEN"):
            try:
                import hvac  # noqa: F401

                available.append(TokenStoreBackend.VAULT)
            except ImportError:
                pass

        # Encrypted file is always available (fallback)
        try:
            from cryptography.fernet import Fernet  # noqa: F401

            available.append(TokenStoreBackend.ENCRYPTED_FILE)
        except ImportError:
            pass

        return available
