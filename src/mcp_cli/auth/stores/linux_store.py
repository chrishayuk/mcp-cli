"""Linux Secret Service API token storage backend."""

import json
import platform
from typing import Optional

from ..oauth_config import OAuthTokens
from ..secure_token_store import SecureTokenStore, TokenStorageError


class SecretServiceTokenStore(SecureTokenStore):
    """Token storage using Linux Secret Service API (GNOME Keyring/KWallet)."""

    SERVICE_NAME = "mcp-cli-oauth"

    def __init__(self):
        """Initialize Secret Service token store."""
        if platform.system() != "Linux":
            raise TokenStorageError("Secret Service storage is only available on Linux")

        try:
            import keyring

            self.keyring = keyring
        except ImportError:
            raise TokenStorageError(
                "keyring library not installed. Install with: pip install keyring"
            )

        # Verify that a keyring backend is available
        backend = keyring.get_keyring()
        if "fail" in backend.__class__.__name__.lower():
            raise TokenStorageError(
                "No keyring backend available. Install one of: "
                "gnome-keyring, kwallet, or secretstorage"
            )

    def store_token(self, server_name: str, tokens: OAuthTokens) -> None:
        """Store tokens in Secret Service."""
        try:
            safe_name = self._sanitize_name(server_name)

            # Add issued_at timestamp if not present
            if tokens.issued_at is None:
                import time

                tokens.issued_at = time.time()

            # Serialize tokens to JSON
            token_json = json.dumps(tokens.to_dict())

            # Store in Secret Service
            self.keyring.set_password(self.SERVICE_NAME, safe_name, token_json)
        except Exception as e:
            raise TokenStorageError(f"Failed to store token in Secret Service: {e}")

    def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
        """Retrieve tokens from Secret Service."""
        try:
            safe_name = self._sanitize_name(server_name)

            # Retrieve from Secret Service
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
                f"Failed to retrieve token from Secret Service: {e}"
            )

    def delete_token(self, server_name: str) -> bool:
        """Delete tokens from Secret Service."""
        try:
            safe_name = self._sanitize_name(server_name)

            # Check if token exists
            if self.keyring.get_password(self.SERVICE_NAME, safe_name) is None:
                return False

            # Delete from Secret Service
            self.keyring.delete_password(self.SERVICE_NAME, safe_name)
            return True
        except Exception as e:
            raise TokenStorageError(f"Failed to delete token from Secret Service: {e}")

    def has_token(self, server_name: str) -> bool:
        """Check if tokens exist in Secret Service."""
        try:
            safe_name = self._sanitize_name(server_name)
            return self.keyring.get_password(self.SERVICE_NAME, safe_name) is not None
        except Exception:
            return False

    # Raw storage methods for generic tokens

    def _store_raw(self, key: str, value: str) -> None:
        """Store raw string value in Secret Service."""
        try:
            safe_key = self._sanitize_name(key)
            self.keyring.set_password(self.SERVICE_NAME, safe_key, value)
        except Exception as e:
            raise TokenStorageError(f"Failed to store value in Secret Service: {e}")

    def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw string value from Secret Service."""
        try:
            safe_key = self._sanitize_name(key)
            result = self.keyring.get_password(self.SERVICE_NAME, safe_key)
            return str(result) if result is not None else None
        except Exception as e:
            raise TokenStorageError(
                f"Failed to retrieve value from Secret Service: {e}"
            )

    def _delete_raw(self, key: str) -> bool:
        """Delete raw value from Secret Service."""
        try:
            safe_key = self._sanitize_name(key)
            if self.keyring.get_password(self.SERVICE_NAME, safe_key) is None:
                return False
            self.keyring.delete_password(self.SERVICE_NAME, safe_key)
            return True
        except Exception as e:
            raise TokenStorageError(f"Failed to delete value from Secret Service: {e}")
