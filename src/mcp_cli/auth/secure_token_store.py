"""Abstract interface for secure token storage."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .oauth_config import OAuthTokens


class SecureTokenStore(ABC):
    """Abstract base class for secure token storage backends."""

    @abstractmethod
    def store_token(self, server_name: str, tokens: OAuthTokens) -> None:
        """
        Store OAuth tokens securely for a server.

        Args:
            server_name: Name of the MCP server
            tokens: OAuth tokens to store

        Raises:
            TokenStorageError: If storage fails
        """
        pass

    @abstractmethod
    def retrieve_token(self, server_name: str) -> Optional[OAuthTokens]:
        """
        Retrieve OAuth tokens for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            OAuth tokens if found, None otherwise

        Raises:
            TokenStorageError: If retrieval fails
        """
        pass

    @abstractmethod
    def delete_token(self, server_name: str) -> bool:
        """
        Delete OAuth tokens for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            True if tokens were deleted, False if they didn't exist

        Raises:
            TokenStorageError: If deletion fails
        """
        pass

    @abstractmethod
    def has_token(self, server_name: str) -> bool:
        """
        Check if OAuth tokens exist for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            True if tokens exist (regardless of expiration)
        """
        pass

    # Generic token storage (for bearer tokens, API keys, etc.)

    def store_generic(self, key: str, value: str, namespace: str = "generic") -> None:
        """
        Store a generic token/secret.

        Args:
            key: Identifier for the token
            value: Token value to store
            namespace: Namespace for organization (default: "generic")

        Raises:
            TokenStorageError: If storage fails
        """
        # Default implementation using JSON serialization
        # Subclasses can override for more efficient storage
        from .token_types import StoredToken, TokenType

        stored = StoredToken(
            token_type=TokenType.BEARER,
            name=key,
            data={"token": value},
            metadata={"namespace": namespace},
        )

        import json

        full_key = f"{namespace}:{key}"
        self._store_raw(full_key, json.dumps(stored.to_dict()))

    def retrieve_generic(self, key: str, namespace: str = "generic") -> Optional[str]:
        """
        Retrieve a generic token/secret.

        Args:
            key: Identifier for the token
            namespace: Namespace for organization (default: "generic")

        Returns:
            Token value if found, None otherwise

        Raises:
            TokenStorageError: If retrieval fails
        """
        import json

        from .token_types import StoredToken

        full_key = f"{namespace}:{key}"
        raw_data = self._retrieve_raw(full_key)

        if raw_data is None:
            return None

        try:
            stored = StoredToken.from_dict(json.loads(raw_data))
            return stored.data.get("token")
        except (json.JSONDecodeError, KeyError):
            return None

    def delete_generic(self, key: str, namespace: str = "generic") -> bool:
        """
        Delete a generic token/secret.

        Args:
            key: Identifier for the token
            namespace: Namespace for organization

        Returns:
            True if token was deleted, False if it didn't exist

        Raises:
            TokenStorageError: If deletion fails
        """
        full_key = f"{namespace}:{key}"
        return self._delete_raw(full_key)

    def list_keys(self, namespace: Optional[str] = None) -> List[str]:
        """
        List all stored token keys.

        Args:
            namespace: Filter by namespace (None = all namespaces)

        Returns:
            List of token identifiers (without namespace prefix)

        Raises:
            TokenStorageError: If listing fails
        """
        # Default implementation - subclasses should override if they can do better
        # This is a basic implementation that may not work for all backends
        return []

    def clear_all(self, namespace: Optional[str] = None) -> int:
        """
        Clear all tokens (optionally filtered by namespace).

        Args:
            namespace: Clear only tokens in this namespace (None = all)

        Returns:
            Number of tokens deleted

        Raises:
            TokenStorageError: If clearing fails
        """
        keys = self.list_keys(namespace)
        count = 0

        for key in keys:
            if self.delete_generic(key, namespace or "generic"):
                count += 1

        return count

    # Abstract methods for raw storage (subclasses must implement)

    @abstractmethod
    def _store_raw(self, key: str, value: str) -> None:
        """Store raw string value (internal use only)."""
        pass

    @abstractmethod
    def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw string value (internal use only)."""
        pass

    @abstractmethod
    def _delete_raw(self, key: str) -> bool:
        """Delete raw value (internal use only)."""
        pass

    def _sanitize_name(self, server_name: str) -> str:
        """Sanitize server name for use as identifier."""
        return "".join(c if c.isalnum() or c in "-_:" else "_" for c in server_name)


class TokenStorageError(Exception):
    """Exception raised when token storage operations fail."""

    pass
