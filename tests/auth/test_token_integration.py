"""Integration tests for complete token management workflow."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_cli.auth.oauth_config import OAuthTokens
from mcp_cli.auth.token_manager import TokenManager
from mcp_cli.auth.token_store_factory import TokenStoreBackend
from mcp_cli.auth.token_types import TokenType
from mcp_cli.commands.actions.token import (
    token_clear_action_async,
    token_delete_action_async,
    token_get_action_async,
    token_list_action_async,
    token_set_action_async,
)


@pytest.fixture
def integration_env(tmp_path):
    """Set up integration test environment."""
    token_dir = tmp_path / "tokens"

    # Create manager
    manager = TokenManager(
        token_dir=token_dir,
        backend=TokenStoreBackend.ENCRYPTED_FILE,
        password="test-password",
    )

    # Mock _get_token_manager to return this instance
    with patch(
        "mcp_cli.commands.actions.token._get_token_manager", return_value=manager
    ):
        yield {
            "manager": manager,
            "token_dir": token_dir,
        }


class TestFullTokenWorkflow:
    """Test complete token workflow from set to delete."""

    @pytest.mark.asyncio
    async def test_bearer_token_lifecycle(self, integration_env):
        """Test complete lifecycle of a bearer token."""
        manager = integration_env["manager"]

        # 1. Set bearer token
        with patch("mcp_cli.commands.actions.token.output"):
            await token_set_action_async(
                name="my-api", token_type="bearer", value="token123", namespace="bearer"
            )

        # 2. Verify token exists in storage
        retrieved = manager.token_store.retrieve_generic("my-api", "bearer")
        assert retrieved is not None

        # 3. Verify token registered
        assert manager.registry.has_token("my-api", "bearer")

        # 4. List tokens
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async()
            mock_output.print_table.assert_called()

        # 5. Get token info
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_get_action_async(name="my-api", namespace="bearer")
            mock_output.info.assert_called()

        # 6. Delete token
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_delete_action_async(name="my-api", namespace="bearer")
            mock_output.success.assert_called()

        # 7. Verify token removed
        assert not manager.registry.has_token("my-api", "bearer")
        assert manager.token_store.retrieve_generic("my-api", "bearer") is None

    @pytest.mark.asyncio
    async def test_api_key_lifecycle(self, integration_env):
        """Test complete lifecycle of an API key."""
        manager = integration_env["manager"]

        # 1. Set API key
        with patch("mcp_cli.commands.actions.token.output"):
            await token_set_action_async(
                name="openai",
                token_type="api-key",
                value="sk-123",
                provider="openai",
                namespace="api-key",
            )

        # 2. Verify stored
        assert manager.registry.has_token("openai", "api-key")

        # 3. Get info
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_get_action_async(name="openai", namespace="api-key")
            mock_output.info.assert_called()

        # 4. Delete
        with patch("mcp_cli.commands.actions.token.output"):
            await token_delete_action_async(name="openai", namespace="api-key")

        # 5. Verify removed
        assert not manager.registry.has_token("openai", "api-key")

    @pytest.mark.asyncio
    async def test_oauth_token_lifecycle(self, integration_env):
        """Test complete lifecycle of OAuth token."""
        manager = integration_env["manager"]

        # 1. Store OAuth token directly via manager
        oauth_tokens = OAuthTokens(
            access_token="access123",
            refresh_token="refresh456",
            expires_in=3600,
            token_type="Bearer",
        )
        manager.save_tokens("test-server", oauth_tokens)

        # 2. Verify stored
        assert manager.has_valid_tokens("test-server")
        assert manager.registry.has_token("test-server", "oauth")

        # 3. List should show it
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async()
            mock_output.print_table.assert_called()

        # 4. Delete via command
        with patch("mcp_cli.commands.actions.token.output"):
            await token_delete_action_async(name="test-server", oauth=True)

        # 5. Verify removed
        assert not manager.has_valid_tokens("test-server")
        assert not manager.registry.has_token("test-server", "oauth")


class TestMultipleTokenTypes:
    """Test managing multiple token types simultaneously."""

    @pytest.mark.asyncio
    async def test_mixed_token_types(self, integration_env):
        """Test storing and managing multiple token types."""
        manager = integration_env["manager"]

        # Store different token types
        with patch("mcp_cli.commands.actions.token.output"):
            # Bearer token
            await token_set_action_async(
                name="api1", token_type="bearer", value="bearer123", namespace="bearer"
            )

            # API key
            await token_set_action_async(
                name="openai",
                token_type="api-key",
                value="sk-123",
                provider="openai",
                namespace="api-key",
            )

            # Generic token
            await token_set_action_async(
                name="custom", token_type="generic", value="custom123", namespace="custom"
            )

        # OAuth token
        oauth_tokens = OAuthTokens(
            access_token="oauth123", expires_in=3600, token_type="Bearer"
        )
        manager.save_tokens("notion", oauth_tokens)

        # List all tokens
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async()
            mock_output.print_table.assert_called()

        # Verify all exist
        assert manager.registry.has_token("api1", "bearer")
        assert manager.registry.has_token("openai", "api-key")
        assert manager.registry.has_token("custom", "custom")
        assert manager.registry.has_token("notion", "oauth")

    @pytest.mark.asyncio
    async def test_filter_token_types(self, integration_env):
        """Test filtering tokens by type."""
        manager = integration_env["manager"]

        # Store different types
        with patch("mcp_cli.commands.actions.token.output"):
            await token_set_action_async(
                name="bearer1", token_type="bearer", value="value1", namespace="bearer"
            )
            await token_set_action_async(
                name="api1",
                token_type="api-key",
                value="key1",
                provider="provider1",
                namespace="api-key",
            )

        oauth_tokens = OAuthTokens(
            access_token="oauth1", expires_in=3600, token_type="Bearer"
        )
        manager.save_tokens("server1", oauth_tokens)

        # List only bearer
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(
                show_oauth=False, show_bearer=True, show_api_keys=False
            )
            mock_output.print_table.assert_called()

        # List only OAuth
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(
                show_oauth=True, show_bearer=False, show_api_keys=False
            )
            mock_output.print_table.assert_called()


class TestNamespaceOperations:
    """Test operations across namespaces."""

    @pytest.mark.asyncio
    async def test_multiple_namespaces(self, integration_env):
        """Test storing tokens in different namespaces."""
        with patch("mcp_cli.commands.actions.token.output"):
            # Store in different namespaces
            await token_set_action_async(
                name="token1", token_type="bearer", value="value1", namespace="ns1"
            )
            await token_set_action_async(
                name="token2", token_type="bearer", value="value2", namespace="ns2"
            )
            await token_set_action_async(
                name="token3", token_type="bearer", value="value3", namespace="ns1"
            )

        # List by namespace
        manager = integration_env["manager"]
        ns1_tokens = manager.registry.list_tokens(namespace="ns1")
        assert len(ns1_tokens) == 2

        ns2_tokens = manager.registry.list_tokens(namespace="ns2")
        assert len(ns2_tokens) == 1

    @pytest.mark.asyncio
    async def test_clear_namespace(self, integration_env):
        """Test clearing tokens in a namespace."""
        manager = integration_env["manager"]

        # Store tokens in different namespaces
        with patch("mcp_cli.commands.actions.token.output"):
            await token_set_action_async(
                name="token1", token_type="bearer", value="value1", namespace="ns1"
            )
            await token_set_action_async(
                name="token2", token_type="bearer", value="value2", namespace="ns2"
            )

        # Clear ns1
        with patch("mcp_cli.commands.actions.token.output"):
            await token_clear_action_async(namespace="ns1", force=True)

        # Verify ns1 cleared but ns2 remains
        assert len(manager.registry.list_tokens(namespace="ns1")) == 0
        assert len(manager.registry.list_tokens(namespace="ns2")) == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, integration_env):
        """Test clearing all tokens."""
        manager = integration_env["manager"]

        # Store multiple tokens
        with patch("mcp_cli.commands.actions.token.output"):
            await token_set_action_async(
                name="token1", token_type="bearer", value="value1", namespace="ns1"
            )
            await token_set_action_async(
                name="token2", token_type="bearer", value="value2", namespace="ns2"
            )

        # Clear all
        with patch("mcp_cli.commands.actions.token.output"):
            await token_clear_action_async(force=True)

        # Verify all cleared
        assert len(manager.registry.list_tokens()) == 0


class TestPersistenceAndRecovery:
    """Test persistence across sessions and error recovery."""

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, tmp_path):
        """Test tokens persist across manager instances."""
        token_dir = tmp_path / "tokens"
        registry_path = tmp_path / "registry.json"

        # First session - store token
        manager1 = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager", return_value=manager1
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                await token_set_action_async(
                    name="persistent",
                    token_type="bearer",
                    value="value123",
                    namespace="bearer",
                )

        # Second session - verify token exists
        manager2 = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        assert manager2.registry.has_token("persistent", "bearer")
        retrieved = manager2.token_store.retrieve_generic("persistent", "bearer")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_registry_recovery_from_corruption(self, tmp_path):
        """Test recovery when registry is corrupted."""
        token_dir = tmp_path / "tokens"

        # Create manager and store token
        manager1 = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        oauth_tokens = OAuthTokens(
            access_token="oauth123", expires_in=3600, token_type="Bearer"
        )
        manager1.save_tokens("server1", oauth_tokens)

        # Get the actual registry path being used
        registry_path = manager1.registry.registry_path

        # Corrupt registry
        with open(registry_path, "w") as f:
            f.write("invalid json {")

        # Create new manager - should handle gracefully
        manager2 = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        # Registry should be empty but functional
        assert len(manager2.registry.list_tokens()) == 0

        # Should still be able to access stored token directly
        retrieved = manager2.load_tokens("server1")
        assert retrieved is not None
        assert retrieved.access_token == "oauth123"

    @pytest.mark.asyncio
    async def test_file_permissions(self, tmp_path):
        """Test that files have correct permissions."""
        import stat

        token_dir = tmp_path / "tokens"

        manager = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager", return_value=manager
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                await token_set_action_async(
                    name="secure", token_type="bearer", value="value123", namespace="bearer"
                )

        # Check token directory permissions
        assert token_dir.exists()
        assert stat.S_IMODE(token_dir.stat().st_mode) == 0o700

        # Check registry file permissions (get actual path)
        registry_path = manager.registry.registry_path
        assert registry_path.exists()
        assert stat.S_IMODE(registry_path.stat().st_mode) == 0o600


class TestExpirationTracking:
    """Test token expiration tracking and display."""

    @pytest.mark.asyncio
    async def test_expired_token_warning(self, integration_env):
        """Test that expired tokens show warning."""
        import time

        manager = integration_env["manager"]

        # Store token with past expiration
        expires_at = time.time() - 3600  # Expired 1 hour ago
        manager.token_store.store_generic("expired", "value123", "bearer")
        manager.registry.register(
            "expired", TokenType.BEARER, "bearer", metadata={"expires_at": expires_at}
        )

        # List should show warning
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async()
            mock_output.print_table.assert_called()
            # Table data should include warning emoji for expired token

    @pytest.mark.asyncio
    async def test_future_expiration(self, integration_env):
        """Test token with future expiration."""
        import time

        manager = integration_env["manager"]

        # Store token with future expiration
        expires_at = time.time() + 3600  # Expires in 1 hour
        manager.token_store.store_generic("future", "value123", "bearer")
        manager.registry.register(
            "future", TokenType.BEARER, "bearer", metadata={"expires_at": expires_at}
        )

        # List should show expiration date without warning
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async()
            mock_output.print_table.assert_called()


class TestSecurityFeatures:
    """Test security-related features."""

    @pytest.mark.asyncio
    async def test_token_values_never_displayed(self, integration_env):
        """Test that token values are never displayed."""
        manager = integration_env["manager"]

        # Store token
        with patch("mcp_cli.commands.actions.token.output"):
            await token_set_action_async(
                name="secret", token_type="bearer", value="supersecret123", namespace="bearer"
            )

        # Get token info - should not show value
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_get_action_async(name="secret", namespace="bearer")

            # Check that output.info was called but never with the actual token value
            for call_args in mock_output.info.call_args_list:
                assert "supersecret123" not in str(call_args)

    @pytest.mark.asyncio
    async def test_secure_prompting(self, integration_env):
        """Test that token input uses secure prompting."""
        with patch("mcp_cli.commands.actions.token.output"):
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = "secret-value"

                # Set without providing value - should use getpass
                await token_set_action_async(name="secure", token_type="bearer")

                # Should have called getpass (hidden input)
                mock_getpass.assert_called_once()

    @pytest.mark.asyncio
    async def test_encryption_isolation(self, tmp_path):
        """Test that different passwords can't decrypt each other's data."""
        token_dir = tmp_path / "tokens"
        registry_path = tmp_path / "registry.json"

        # Store with password1
        manager1 = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="password1",
        )
        manager1.token_store._store_raw("test-key", "test-value")

        # Try to retrieve with password2
        manager2 = TokenManager(
            token_dir=token_dir,
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="password2",
        )

        from mcp_cli.auth.secure_token_store import TokenStorageError

        with pytest.raises(TokenStorageError):
            manager2.token_store._retrieve_raw("test-key")


class TestErrorHandling:
    """Test error handling throughout the workflow."""

    @pytest.mark.asyncio
    async def test_invalid_token_type(self, integration_env):
        """Test handling of invalid token type."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_set_action_async(
                name="test", token_type="invalid", value="value123"
            )

            mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, integration_env):
        """Test handling of missing required fields."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            # API key without provider
            await token_set_action_async(
                name="test", token_type="api-key", value="key123"
            )

            mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_empty_value(self, integration_env):
        """Test handling of empty token value."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = ""

                await token_set_action_async(name="test", token_type="bearer")

                mock_output.error.assert_called()
