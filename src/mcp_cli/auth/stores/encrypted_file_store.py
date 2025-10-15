"""Encrypted file token storage backend (fallback)."""

import json
import os
from pathlib import Path
from typing import Optional

from ..oauth_config import OAuthTokens
from ..secure_token_store import SecureTokenStore, TokenStorageError


class EncryptedFileTokenStore(SecureTokenStore):
    """Token storage using encrypted files (fallback when OS keyring unavailable)."""

    def __init__(
        self, token_dir: Optional[Path] = None, password: Optional[str] = None
    ):
        """
        Initialize encrypted file token store.

        Args:
            token_dir: Directory to store encrypted tokens (default: ~/.mcp_cli/tokens)
            password: Encryption password (default: prompt user or use env var)

        Raises:
            TokenStorageError: If encryption setup fails
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

            self.Fernet = Fernet
            self.PBKDF2HMAC = PBKDF2HMAC
            self.hashes = hashes
        except ImportError:
            raise TokenStorageError(
                "cryptography library not installed. "
                "Install with: pip install cryptography"
            )

        if token_dir is None:
            token_dir = Path.home() / ".mcp_cli" / "tokens"

        self.token_dir = token_dir
        self.token_dir.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions on token directory
        os.chmod(self.token_dir, 0o700)

        # Initialize encryption key
        self._initialize_key(password)

    def _initialize_key(self, password: Optional[str] = None) -> None:
        """Initialize or load encryption key."""
        salt_file = self.token_dir / ".salt"

        # Get or create salt
        if salt_file.exists():
            with open(salt_file, "rb") as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(salt_file, "wb") as f:
                f.write(salt)
            os.chmod(salt_file, 0o600)

        # Get password from parameter, environment, or prompt
        if password is None:
            password = os.getenv("MCP_CLI_ENCRYPTION_KEY")

        if password is None:
            # Prompt for password
            from getpass import getpass

            password = getpass("Enter encryption password for token storage: ")

        # Derive key from password using PBKDF2
        kdf = self.PBKDF2HMAC(
            algorithm=self.hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = kdf.derive(password.encode())

        # Store base64-encoded key for Fernet
        import base64

        self.key = base64.urlsafe_b64encode(key)
        self.fernet = self.Fernet(self.key)

    def _get_token_path(self, server_name: str) -> Path:
        """Get path to encrypted token file."""
        safe_name = self._sanitize_name(server_name)
        return self.token_dir / f"{safe_name}.enc"

    def store_token(self, server_name: str, tokens: OAuthTokens) -> None:
        """Store tokens in encrypted file."""
        try:
            token_path = self._get_token_path(server_name)

            # Add issued_at timestamp if not present
            if tokens.issued_at is None:
                import time

                tokens.issued_at = time.time()

            # Serialize to JSON
            token_json = json.dumps(tokens.to_dict())

            # Encrypt
            encrypted_data = self.fernet.encrypt(token_json.encode())

            # Write to file
            with open(token_path, "wb") as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(token_path, 0o600)
        except Exception as e:
            raise TokenStorageError(f"Failed to store encrypted token: {e}")

    def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
        """Retrieve tokens from encrypted file."""
        try:
            token_path = self._get_token_path(server_name)

            if not token_path.exists():
                return None

            # Read encrypted data
            with open(token_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt
            decrypted_data = self.fernet.decrypt(encrypted_data)

            # Parse JSON
            token_data = json.loads(decrypted_data.decode())
            return OAuthTokens.from_dict(token_data)
        except json.JSONDecodeError as e:
            raise TokenStorageError(f"Failed to parse token data: {e}")
        except Exception as e:
            raise TokenStorageError(f"Failed to retrieve encrypted token: {e}")

    def delete_token(self, server_name: str) -> bool:
        """Delete encrypted token file."""
        try:
            token_path = self._get_token_path(server_name)

            if not token_path.exists():
                return False

            token_path.unlink()
            return True
        except Exception as e:
            raise TokenStorageError(f"Failed to delete encrypted token: {e}")

    def has_token(self, server_name: str) -> bool:
        """Check if encrypted token file exists."""
        token_path = self._get_token_path(server_name)
        return token_path.exists()

    # Raw storage methods for generic tokens

    def _store_raw(self, key: str, value: str) -> None:
        """Store raw string value in encrypted file."""
        try:
            safe_key = self._sanitize_name(key)
            file_path = self.token_dir / f"{safe_key}.enc"

            # Encrypt value
            encrypted_data = self.fernet.encrypt(value.encode())

            # Write to file
            with open(file_path, "wb") as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            import os

            os.chmod(file_path, 0o600)
        except Exception as e:
            raise TokenStorageError(f"Failed to store encrypted value: {e}")

    def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw string value from encrypted file."""
        try:
            safe_key = self._sanitize_name(key)
            file_path = self.token_dir / f"{safe_key}.enc"

            if not file_path.exists():
                return None

            # Read encrypted data
            with open(file_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt
            decrypted_data = self.fernet.decrypt(encrypted_data)
            result: str = decrypted_data.decode()
            return result
        except Exception as e:
            raise TokenStorageError(f"Failed to retrieve encrypted value: {e}")

    def _delete_raw(self, key: str) -> bool:
        """Delete encrypted file."""
        try:
            safe_key = self._sanitize_name(key)
            file_path = self.token_dir / f"{safe_key}.enc"

            if not file_path.exists():
                return False

            file_path.unlink()
            return True
        except Exception as e:
            raise TokenStorageError(f"Failed to delete encrypted value: {e}")
