"""Token registry for tracking stored tokens."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .token_types import TokenType


class TokenRegistry:
    """
    Registry for tracking tokens stored in secure storage.

    This maintains a lightweight index of tokens (metadata only, no values)
    to enable listing functionality across all storage backends.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize token registry.

        Args:
            registry_path: Path to registry file (default: ~/.mcp_cli/token_registry.json)
        """
        if registry_path is None:
            registry_path = Path.home() / ".mcp_cli" / "token_registry.json"

        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._entries: Dict[str, Dict] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[Any, Any]]:
        """Load registry from file."""
        if not self.registry_path.exists():
            return {}

        try:
            with open(self.registry_path, "r") as f:
                data: Dict[str, Dict[Any, Any]] = json.load(f)
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_registry(self) -> None:
        """Save registry to file."""
        with open(self.registry_path, "w") as f:
            json.dump(self._entries, f, indent=2)

        # Set restrictive permissions
        import os

        os.chmod(self.registry_path, 0o600)

    def _make_key(self, namespace: str, name: str) -> str:
        """Create registry key from namespace and name."""
        return f"{namespace}:{name}"

    def register(
        self,
        name: str,
        token_type: TokenType,
        namespace: str = "generic",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Register a token in the index.

        Args:
            name: Token name
            token_type: Type of token
            namespace: Storage namespace
            metadata: Additional metadata (no sensitive data)
        """
        import time

        key = self._make_key(namespace, name)
        self._entries[key] = {
            "name": name,
            "type": token_type.value
            if isinstance(token_type, TokenType)
            else token_type,
            "namespace": namespace,
            "registered_at": time.time(),
            "metadata": metadata or {},
        }
        self._save_registry()

    def unregister(self, name: str, namespace: str = "generic") -> bool:
        """
        Unregister a token from the index.

        Args:
            name: Token name
            namespace: Storage namespace

        Returns:
            True if token was unregistered, False if not found
        """
        key = self._make_key(namespace, name)
        if key in self._entries:
            del self._entries[key]
            self._save_registry()
            return True
        return False

    def list_tokens(
        self,
        namespace: Optional[str] = None,
        token_type: Optional[TokenType] = None,
    ) -> List[Dict]:
        """
        List registered tokens.

        Args:
            namespace: Filter by namespace (None = all)
            token_type: Filter by token type (None = all)

        Returns:
            List of token metadata dictionaries
        """
        results = []

        for key, entry in self._entries.items():
            # Apply filters
            if namespace and entry.get("namespace") != namespace:
                continue

            if token_type and entry.get("type") != token_type.value:
                continue

            results.append(entry)

        # Sort by registered_at (most recent first)
        results.sort(key=lambda x: x.get("registered_at", 0), reverse=True)

        return results

    def get_entry(self, name: str, namespace: str = "generic") -> Optional[Dict]:
        """
        Get registry entry for a token.

        Args:
            name: Token name
            namespace: Storage namespace

        Returns:
            Token metadata if found, None otherwise
        """
        key = self._make_key(namespace, name)
        return self._entries.get(key)

    def has_token(self, name: str, namespace: str = "generic") -> bool:
        """
        Check if token is registered.

        Args:
            name: Token name
            namespace: Storage namespace

        Returns:
            True if token is registered
        """
        key = self._make_key(namespace, name)
        return key in self._entries

    def clear_namespace(self, namespace: str) -> int:
        """
        Clear all tokens from a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of tokens cleared
        """
        keys_to_remove = [
            key
            for key, entry in self._entries.items()
            if entry.get("namespace") == namespace
        ]

        for key in keys_to_remove:
            del self._entries[key]

        if keys_to_remove:
            self._save_registry()

        return len(keys_to_remove)

    def clear_all(self) -> int:
        """
        Clear all registered tokens.

        Returns:
            Number of tokens cleared
        """
        count = len(self._entries)
        self._entries = {}
        self._save_registry()
        return count

    def update_metadata(
        self,
        name: str,
        namespace: str,
        metadata: Dict,
    ) -> bool:
        """
        Update metadata for a token.

        Args:
            name: Token name
            namespace: Storage namespace
            metadata: New metadata (merged with existing)

        Returns:
            True if token was found and updated
        """
        key = self._make_key(namespace, name)
        if key in self._entries:
            existing_metadata = self._entries[key].get("metadata", {})
            existing_metadata.update(metadata)
            self._entries[key]["metadata"] = existing_metadata
            self._save_registry()
            return True
        return False
