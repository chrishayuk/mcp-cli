"""Tests for TokenRegistry."""

import json

import pytest

from mcp_cli.auth.token_registry import TokenRegistry
from mcp_cli.auth.token_types import TokenType


@pytest.fixture
def temp_registry_path(tmp_path):
    """Provide temporary registry path."""
    return tmp_path / "token_registry.json"


@pytest.fixture
def registry(temp_registry_path):
    """Provide TokenRegistry instance with temp path."""
    return TokenRegistry(registry_path=temp_registry_path)


class TestTokenRegistry:
    """Test TokenRegistry functionality."""

    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates parent directory."""
        registry_path = tmp_path / "subdir" / "registry.json"
        registry = TokenRegistry(registry_path=registry_path)

        assert registry_path.parent.exists()
        assert registry.registry_path == registry_path

    def test_register_token(self, registry, temp_registry_path):
        """Test registering a token."""
        registry.register(
            name="test-token",
            token_type=TokenType.BEARER,
            namespace="test",
            metadata={"foo": "bar"},
        )

        # Check registry file was created
        assert temp_registry_path.exists()

        # Check file permissions
        import stat

        mode = temp_registry_path.stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

        # Check content
        with open(temp_registry_path) as f:
            data = json.load(f)

        assert "test:test-token" in data
        entry = data["test:test-token"]
        assert entry["name"] == "test-token"
        assert entry["type"] == "bearer"
        assert entry["namespace"] == "test"
        assert entry["metadata"]["foo"] == "bar"
        assert "registered_at" in entry

    def test_unregister_token(self, registry):
        """Test unregistering a token."""
        registry.register("test-token", TokenType.BEARER, "test")

        # Unregister existing token
        result = registry.unregister("test-token", "test")
        assert result is True

        # Unregister non-existent token
        result = registry.unregister("non-existent", "test")
        assert result is False

    def test_list_tokens_empty(self, registry):
        """Test listing tokens when registry is empty."""
        tokens = registry.list_tokens()
        assert tokens == []

    def test_list_tokens(self, registry):
        """Test listing all tokens."""
        registry.register("token1", TokenType.BEARER, "ns1")
        registry.register("token2", TokenType.API_KEY, "ns2")
        registry.register("token3", TokenType.BEARER, "ns1")

        tokens = registry.list_tokens()
        assert len(tokens) == 3

        # Check they're sorted by registered_at (most recent first)
        names = [t["name"] for t in tokens]
        assert names == ["token3", "token2", "token1"]

    def test_list_tokens_by_namespace(self, registry):
        """Test filtering tokens by namespace."""
        registry.register("token1", TokenType.BEARER, "ns1")
        registry.register("token2", TokenType.API_KEY, "ns2")
        registry.register("token3", TokenType.BEARER, "ns1")

        tokens = registry.list_tokens(namespace="ns1")
        assert len(tokens) == 2
        assert all(t["namespace"] == "ns1" for t in tokens)

    def test_list_tokens_by_type(self, registry):
        """Test filtering tokens by type."""
        registry.register("token1", TokenType.BEARER, "ns1")
        registry.register("token2", TokenType.API_KEY, "ns2")
        registry.register("token3", TokenType.BEARER, "ns3")

        tokens = registry.list_tokens(token_type=TokenType.BEARER)
        assert len(tokens) == 2
        assert all(t["type"] == "bearer" for t in tokens)

    def test_get_entry(self, registry):
        """Test getting a specific entry."""
        registry.register(
            "test-token", TokenType.BEARER, "test", metadata={"key": "value"}
        )

        entry = registry.get_entry("test-token", "test")
        assert entry is not None
        assert entry["name"] == "test-token"
        assert entry["type"] == "bearer"
        assert entry["metadata"]["key"] == "value"

        # Non-existent entry
        entry = registry.get_entry("non-existent", "test")
        assert entry is None

    def test_has_token(self, registry):
        """Test checking if token exists."""
        registry.register("test-token", TokenType.BEARER, "test")

        assert registry.has_token("test-token", "test") is True
        assert registry.has_token("non-existent", "test") is False

    def test_clear_namespace(self, registry):
        """Test clearing all tokens in a namespace."""
        registry.register("token1", TokenType.BEARER, "ns1")
        registry.register("token2", TokenType.API_KEY, "ns2")
        registry.register("token3", TokenType.BEARER, "ns1")

        count = registry.clear_namespace("ns1")
        assert count == 2

        # Check ns1 tokens are gone
        tokens = registry.list_tokens(namespace="ns1")
        assert len(tokens) == 0

        # Check ns2 token still exists
        tokens = registry.list_tokens(namespace="ns2")
        assert len(tokens) == 1

    def test_clear_all(self, registry):
        """Test clearing all tokens."""
        registry.register("token1", TokenType.BEARER, "ns1")
        registry.register("token2", TokenType.API_KEY, "ns2")
        registry.register("token3", TokenType.BEARER, "ns3")

        count = registry.clear_all()
        assert count == 3

        # Check all tokens are gone
        tokens = registry.list_tokens()
        assert len(tokens) == 0

    def test_update_metadata(self, registry):
        """Test updating token metadata."""
        registry.register(
            "test-token", TokenType.BEARER, "test", metadata={"key1": "value1"}
        )

        # Update metadata
        result = registry.update_metadata("test-token", "test", {"key2": "value2"})
        assert result is True

        # Check metadata was merged
        entry = registry.get_entry("test-token", "test")
        assert entry["metadata"]["key1"] == "value1"
        assert entry["metadata"]["key2"] == "value2"

        # Update non-existent token
        result = registry.update_metadata("non-existent", "test", {})
        assert result is False

    def test_persistence(self, temp_registry_path):
        """Test that registry persists across instances."""
        # Create first registry and add token
        registry1 = TokenRegistry(registry_path=temp_registry_path)
        registry1.register("test-token", TokenType.BEARER, "test")

        # Create second registry and verify token exists
        registry2 = TokenRegistry(registry_path=temp_registry_path)
        assert registry2.has_token("test-token", "test") is True

        entry = registry2.get_entry("test-token", "test")
        assert entry is not None
        assert entry["name"] == "test-token"

    def test_corrupted_registry_file(self, temp_registry_path):
        """Test handling of corrupted registry file."""
        # Write invalid JSON
        with open(temp_registry_path, "w") as f:
            f.write("invalid json {")

        # Should load with empty registry
        registry = TokenRegistry(registry_path=temp_registry_path)
        tokens = registry.list_tokens()
        assert tokens == []

    def test_make_key(self, registry):
        """Test key generation."""
        key = registry._make_key("test-ns", "test-name")
        assert key == "test-ns:test-name"

    def test_register_with_string_token_type(self, registry):
        """Test registering with string token type."""
        registry.register("test-token", "bearer", "test")

        entry = registry.get_entry("test-token", "test")
        assert entry is not None
        assert entry["type"] == "bearer"
