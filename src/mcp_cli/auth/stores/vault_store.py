"""HashiCorp Vault token storage backend."""

import json
import os
from typing import Optional

from ..oauth_config import OAuthTokens
from ..secure_token_store import SecureTokenStore, TokenStorageError


class VaultTokenStore(SecureTokenStore):
    """Token storage using HashiCorp Vault."""

    def __init__(
        self,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        mount_point: str = "secret",
        path_prefix: str = "mcp-cli/oauth",
        namespace: Optional[str] = None,
    ):
        """
        Initialize Vault token store.

        Args:
            vault_url: Vault server URL (default: VAULT_ADDR env var)
            vault_token: Vault authentication token (default: VAULT_TOKEN env var)
            mount_point: KV secrets engine mount point (default: "secret")
            path_prefix: Path prefix for storing tokens (default: "mcp-cli/oauth")
            namespace: Vault namespace (optional, for Vault Enterprise)

        Raises:
            TokenStorageError: If Vault configuration is invalid
        """
        try:
            import hvac

            self.hvac = hvac
        except ImportError:
            raise TokenStorageError(
                "hvac library not installed. Install with: pip install hvac"
            )

        # Get configuration from parameters or environment
        self.vault_url = vault_url or os.getenv("VAULT_ADDR")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.mount_point = mount_point
        self.path_prefix = path_prefix.strip("/")
        self.namespace = namespace

        if not self.vault_url:
            raise TokenStorageError(
                "Vault URL not provided. Set VAULT_ADDR environment variable "
                "or pass vault_url parameter"
            )

        if not self.vault_token:
            raise TokenStorageError(
                "Vault token not provided. Set VAULT_TOKEN environment variable "
                "or pass vault_token parameter"
            )

        # Initialize Vault client
        try:
            self.client = hvac.Client(
                url=self.vault_url,
                token=self.vault_token,
                namespace=self.namespace,
            )

            # Verify authentication
            if not self.client.is_authenticated():
                raise TokenStorageError("Vault authentication failed")
        except Exception as e:
            raise TokenStorageError(f"Failed to initialize Vault client: {e}")

    def _get_vault_path(self, server_name: str) -> str:
        """Get full Vault path for a server."""
        safe_name = self._sanitize_name(server_name)
        return f"{self.path_prefix}/{safe_name}"

    def store_token(self, server_name: str, tokens: OAuthTokens) -> None:
        """Store tokens in HashiCorp Vault."""
        try:
            path = self._get_vault_path(server_name)

            # Add issued_at timestamp if not present
            if tokens.issued_at is None:
                import time

                tokens.issued_at = time.time()

            # Prepare token data
            token_data = tokens.to_dict()

            # Store in Vault (KV v2)
            try:
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=token_data,
                    mount_point=self.mount_point,
                )
            except self.hvac.exceptions.InvalidPath:
                # Fallback to KV v1 if v2 is not available
                self.client.secrets.kv.v1.create_or_update_secret(
                    path=path,
                    secret=token_data,
                    mount_point=self.mount_point,
                )
        except Exception as e:
            raise TokenStorageError(f"Failed to store token in Vault: {e}")

    def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
        """Retrieve tokens from HashiCorp Vault."""
        try:
            path = self._get_vault_path(server_name)

            # Read from Vault (KV v2)
            try:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.mount_point,
                )
                token_data = response["data"]["data"]
            except self.hvac.exceptions.InvalidPath:
                # Fallback to KV v1 if v2 is not available
                try:
                    response = self.client.secrets.kv.v1.read_secret(
                        path=path,
                        mount_point=self.mount_point,
                    )
                    token_data = response["data"]
                except self.hvac.exceptions.InvalidPath:
                    # Secret doesn't exist
                    return None

            return OAuthTokens.from_dict(token_data)
        except json.JSONDecodeError as e:
            raise TokenStorageError(f"Failed to parse token data: {e}")
        except Exception as e:
            raise TokenStorageError(f"Failed to retrieve token from Vault: {e}")

    def delete_token(self, server_name: str) -> bool:
        """Delete tokens from HashiCorp Vault."""
        try:
            path = self._get_vault_path(server_name)

            # Check if token exists
            if not self.has_token(server_name):
                return False

            # Delete from Vault (KV v2)
            try:
                self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path,
                    mount_point=self.mount_point,
                )
            except self.hvac.exceptions.InvalidPath:
                # Fallback to KV v1 if v2 is not available
                self.client.secrets.kv.v1.delete_secret(
                    path=path,
                    mount_point=self.mount_point,
                )

            return True
        except Exception as e:
            raise TokenStorageError(f"Failed to delete token from Vault: {e}")

    def has_token(self, server_name: str) -> bool:
        """Check if tokens exist in HashiCorp Vault."""
        try:
            path = self._get_vault_path(server_name)

            # Try to read (KV v2)
            try:
                self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.mount_point,
                )
                return True
            except self.hvac.exceptions.InvalidPath:
                # Try KV v1
                try:
                    self.client.secrets.kv.v1.read_secret(
                        path=path,
                        mount_point=self.mount_point,
                    )
                    return True
                except self.hvac.exceptions.InvalidPath:
                    return False
        except Exception:
            return False

    # Raw storage methods for generic tokens

    def _store_raw(self, key: str, value: str) -> None:
        """Store raw string value in Vault."""
        try:
            path = f"{self.path_prefix}/{self._sanitize_name(key)}"
            data = {"value": value}

            # Store in Vault (KV v2)
            try:
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=self.mount_point,
                )
            except self.hvac.exceptions.InvalidPath:
                # Fallback to KV v1
                self.client.secrets.kv.v1.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=self.mount_point,
                )
        except Exception as e:
            raise TokenStorageError(f"Failed to store value in Vault: {e}")

    def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw string value from Vault."""
        try:
            path = f"{self.path_prefix}/{self._sanitize_name(key)}"

            # Read from Vault (KV v2)
            try:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.mount_point,
                )
                result = response["data"]["data"].get("value")
                return str(result) if result is not None else None
            except self.hvac.exceptions.InvalidPath:
                # Fallback to KV v1
                try:
                    response = self.client.secrets.kv.v1.read_secret(
                        path=path,
                        mount_point=self.mount_point,
                    )
                    result = response["data"].get("value")
                    return str(result) if result is not None else None
                except self.hvac.exceptions.InvalidPath:
                    return None
        except Exception as e:
            raise TokenStorageError(f"Failed to retrieve value from Vault: {e}")

    def _delete_raw(self, key: str) -> bool:
        """Delete raw value from Vault."""
        try:
            path = f"{self.path_prefix}/{self._sanitize_name(key)}"

            # Check if exists first
            if not self._retrieve_raw(key):
                return False

            # Delete from Vault (KV v2)
            try:
                self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path,
                    mount_point=self.mount_point,
                )
            except self.hvac.exceptions.InvalidPath:
                # Fallback to KV v1
                self.client.secrets.kv.v1.delete_secret(
                    path=path,
                    mount_point=self.mount_point,
                )

            return True
        except Exception as e:
            raise TokenStorageError(f"Failed to delete value from Vault: {e}")
