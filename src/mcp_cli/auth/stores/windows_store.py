"""Windows Credential Manager token storage backend."""

import json
import platform
from typing import Optional

from ..oauth_config import OAuthTokens
from ..secure_token_store import SecureTokenStore, TokenStorageError


class CredentialManagerTokenStore(SecureTokenStore):
    """Token storage using Windows Credential Manager."""

    SERVICE_NAME = "mcp-cli-oauth"

    def __init__(self):
        """Initialize Windows Credential Manager token store."""
        if platform.system() != "Windows":
            raise TokenStorageError(
                "Windows Credential Manager is only available on Windows"
            )

        try:
            import keyring

            self.keyring = keyring
        except ImportError:
            raise TokenStorageError(
                "keyring library not installed. Install with: pip install keyring"
            )

    def store_token(self, server_name: str, tokens: OAuthTokens) -> None:
        """Store tokens in Windows Credential Manager."""
        try:
            safe_name = self._sanitize_name(server_name)

            # Add issued_at timestamp if not present
            if tokens.issued_at is None:
                import time

                tokens.issued_at = time.time()

            # Serialize tokens to JSON
            token_json = json.dumps(tokens.to_dict())

            # Store in Credential Manager
            self.keyring.set_password(self.SERVICE_NAME, safe_name, token_json)
        except Exception as e:
            raise TokenStorageError(f"Failed to store token in Credential Manager: {e}")

    def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
        """Retrieve tokens from Windows Credential Manager."""
        try:
            safe_name = self._sanitize_name(server_name)

            # Retrieve from Credential Manager
            token_json = self.keyring.get_password(self.SERVICE_NAME, safe_name)

            if token_json is None:
                return None

            # Deserialize from JSON
            token_data = json.loads(token_json)
            return OAuthTokens.from_dict(token_data)
        except json.JSONDecodeError as e:
            raise TokenStorageError(f"Failed to parse token data: {e}")
        except Exception as e:
            raise TokenStorageError(
                f"Failed to retrieve token from Credential Manager: {e}"
            )

    def delete_token(self, server_name: str) -> bool:
        """Delete tokens from Windows Credential Manager."""
        try:
            safe_name = self._sanitize_name(server_name)

            # Check if token exists
            if self.keyring.get_password(self.SERVICE_NAME, safe_name) is None:
                return False

            # Delete from Credential Manager
            self.keyring.delete_password(self.SERVICE_NAME, safe_name)
            return True
        except Exception as e:
            raise TokenStorageError(
                f"Failed to delete token from Credential Manager: {e}"
            )

    def has_token(self, server_name: str) -> bool:
        """Check if tokens exist in Windows Credential Manager."""
        try:
            safe_name = self._sanitize_name(server_name)
            return self.keyring.get_password(self.SERVICE_NAME, safe_name) is not None
        except Exception:
            return False

    # Raw storage methods for generic tokens

    def _store_raw(self, key: str, value: str) -> None:
        """Store raw string value in Credential Manager."""
        try:
            safe_key = self._sanitize_name(key)
            self.keyring.set_password(self.SERVICE_NAME, safe_key, value)
        except Exception as e:
            raise TokenStorageError(f"Failed to store value in Credential Manager: {e}")

    def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw string value from Credential Manager."""
        try:
            safe_key = self._sanitize_name(key)
            result = self.keyring.get_password(self.SERVICE_NAME, safe_key)
            return str(result) if result is not None else None
        except Exception as e:
            raise TokenStorageError(
                f"Failed to retrieve value from Credential Manager: {e}"
            )

    def _delete_raw(self, key: str) -> bool:
        """Delete raw value from Credential Manager."""
        try:
            safe_key = self._sanitize_name(key)
            if self.keyring.get_password(self.SERVICE_NAME, safe_key) is None:
                return False
            self.keyring.delete_password(self.SERVICE_NAME, safe_key)
            return True
        except Exception as e:
            raise TokenStorageError(
                f"Failed to delete value from Credential Manager: {e}"
            )
