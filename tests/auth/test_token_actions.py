"""Tests for token command actions."""

import json
from unittest.mock import patch

import pytest

from mcp_cli.auth import OAuthTokens
from mcp_cli.auth import TokenType
from mcp_cli.commands.actions.token import (
    token_backends_action_async,
    token_clear_action_async,
    token_delete_action_async,
    token_get_action_async,
    token_list_action_async,
    token_set_action_async,
)
from mcp_cli.commands.models import (
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
)


@pytest.fixture
def mock_token_manager(tmp_path):
    """Mock TokenManager with temporary storage."""
    with patch("mcp_cli.commands.actions.token._get_token_manager") as mock:
        from mcp_cli.auth import TokenManager
        from mcp_cli.auth import TokenStoreBackend

        manager = TokenManager(
            token_dir=tmp_path / "tokens",
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
            service_name="mcp-cli",
        )
        mock.return_value = manager
        yield manager


@pytest.fixture
def sample_oauth_tokens():
    """Sample OAuth tokens."""
    return OAuthTokens(
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        expires_in=3600,
        token_type="Bearer",
    )


class TestTokenListAction:
    """Test token list action."""

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_token_manager):
        """Test listing when no tokens exist."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(TokenListParams())
            # Should show info/warning messages
            # The function shows either warning about no tokens or info about token management
            assert mock_output.info.called or mock_output.warning.called

    @pytest.mark.asyncio
    async def test_list_with_oauth_tokens(
        self, mock_token_manager, sample_oauth_tokens
    ):
        """Test listing OAuth tokens."""
        # Store token
        mock_token_manager.save_tokens("test-server", sample_oauth_tokens)

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(TokenListParams())
            # Should print table
            mock_output.print_table.assert_called()

    @pytest.mark.asyncio
    async def test_list_with_bearer_tokens(self, mock_token_manager):
        """Test listing bearer tokens."""
        # Store bearer token
        mock_token_manager.token_store.store_generic("my-api", "token123", "bearer")
        mock_token_manager.registry.register("my-api", TokenType.BEARER, "bearer")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(TokenListParams())
            mock_output.print_table.assert_called()

    @pytest.mark.asyncio
    async def test_list_filter_by_namespace(self, mock_token_manager):
        """Test filtering tokens by namespace."""
        # Store tokens in different namespaces
        mock_token_manager.token_store.store_generic("token1", "value1", "ns1")
        mock_token_manager.registry.register("token1", TokenType.BEARER, "ns1")

        mock_token_manager.token_store.store_generic("token2", "value2", "ns2")
        mock_token_manager.registry.register("token2", TokenType.BEARER, "ns2")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(TokenListParams(namespace="ns1"))
            mock_output.print_table.assert_called()

    @pytest.mark.asyncio
    async def test_list_filter_by_type(self, mock_token_manager, sample_oauth_tokens):
        """Test filtering tokens by type."""
        # Store different token types
        mock_token_manager.save_tokens("server1", sample_oauth_tokens)
        mock_token_manager.token_store.store_generic("bearer1", "value1", "bearer")
        mock_token_manager.registry.register("bearer1", TokenType.BEARER, "bearer")

        # List only bearer tokens
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(
                TokenListParams(show_oauth=False, show_bearer=True)
            )
            mock_output.print_table.assert_called()

    @pytest.mark.asyncio
    async def test_list_shows_expiration(self, mock_token_manager):
        """Test that list shows expiration dates."""
        import time

        # Store token with expiration
        expires_at = time.time() + 3600
        mock_token_manager.token_store.store_generic("token1", "value1", "bearer")
        mock_token_manager.registry.register(
            "token1", TokenType.BEARER, "bearer", metadata={"expires_at": expires_at}
        )

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_list_action_async(TokenListParams())
            # Should format table with expiration column
            mock_output.print_table.assert_called()

    @pytest.mark.asyncio
    async def test_list_error_handling(self):
        """Test error handling in list action."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception):
                await token_list_action_async(TokenListParams())


class TestTokenSetAction:
    """Test token set action."""

    @pytest.mark.asyncio
    async def test_set_bearer_token(self, mock_token_manager):
        """Test storing bearer token."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_set_action_async(
                TokenSetParams(
                    name="my-token",
                    token_type="bearer",
                    value="token123",
                    namespace="bearer",
                )
            )

            # Should show success
            mock_output.success.assert_called()

            # Verify stored
            retrieved = mock_token_manager.token_store.retrieve_generic(
                "my-token", "bearer"
            )
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_set_api_key(self, mock_token_manager):
        """Test storing API key."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_set_action_async(
                TokenSetParams(
                    name="openai",
                    token_type="api-key",
                    value="sk-123",
                    provider="openai",
                    namespace="api-key",
                )
            )

            mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_set_api_key_without_provider(self, mock_token_manager):
        """Test that API key requires provider."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_set_action_async(
                TokenSetParams(name="openai", token_type="api-key", value="sk-123")
            )

            # Should show error
            mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_set_generic_token(self, mock_token_manager):
        """Test storing generic token."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_set_action_async(
                TokenSetParams(
                    name="my-token",
                    token_type="generic",
                    value="value123",
                    namespace="custom",
                )
            )

            mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_set_prompts_for_value(self, mock_token_manager):
        """Test that set prompts for value if not provided."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = "prompted-value"

                await token_set_action_async(
                    TokenSetParams(name="my-token", token_type="bearer")
                )

                # Should call getpass
                mock_getpass.assert_called()
                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_set_empty_value(self, mock_token_manager):
        """Test that empty value shows error."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = ""

                await token_set_action_async(
                    TokenSetParams(name="my-token", token_type="bearer")
                )

                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_set_unknown_type(self, mock_token_manager):
        """Test error handling for unknown token type."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_set_action_async(
                TokenSetParams(name="my-token", token_type="unknown", value="value123")
            )

            mock_output.error.assert_called()
            mock_output.hint.assert_called()


class TestTokenGetAction:
    """Test token get action."""

    @pytest.mark.asyncio
    async def test_get_existing_token(self, mock_token_manager):
        """Test getting token information."""
        # Store token
        from mcp_cli.auth import BearerToken

        bearer = BearerToken(token="token123")
        stored = bearer.to_stored_token("my-token")
        stored.metadata = {"namespace": "bearer"}
        mock_token_manager.token_store._store_raw(
            "bearer:my-token", json.dumps(stored.model_dump())
        )

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_get_action_async(name="my-token", namespace="bearer")

            # Should show token info
            mock_output.rule.assert_called()
            mock_output.info.assert_called()

    @pytest.mark.asyncio
    async def test_get_nonexistent_token(self, mock_token_manager):
        """Test getting token that doesn't exist."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_get_action_async(name="nonexistent", namespace="bearer")

            mock_output.warning.assert_called()

    @pytest.mark.asyncio
    async def test_get_with_different_namespace(self, mock_token_manager):
        """Test getting token from specific namespace."""
        from mcp_cli.auth import APIKeyToken

        api_key = APIKeyToken(provider="openai", key="sk-123")
        stored = api_key.to_stored_token("openai")
        stored.metadata = {"namespace": "api-key"}
        mock_token_manager.token_store._store_raw(
            "api-key:openai", json.dumps(stored.model_dump())
        )

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_get_action_async(name="openai", namespace="api-key")

            mock_output.info.assert_called()


class TestTokenDeleteAction:
    """Test token delete action."""

    @pytest.mark.asyncio
    async def test_delete_oauth_token(self, mock_token_manager, sample_oauth_tokens):
        """Test deleting OAuth token."""
        # Store OAuth token
        mock_token_manager.save_tokens("test-server", sample_oauth_tokens)

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_delete_action_async(
                TokenDeleteParams(name="test-server", oauth=True)
            )

            mock_output.success.assert_called()

            # Verify deleted
            assert not mock_token_manager.has_valid_tokens("test-server")

    @pytest.mark.asyncio
    async def test_delete_generic_token(self, mock_token_manager):
        """Test deleting generic token."""
        # Store token
        mock_token_manager.token_store.store_generic("my-token", "value123", "bearer")
        mock_token_manager.registry.register("my-token", TokenType.BEARER, "bearer")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_delete_action_async(
                TokenDeleteParams(name="my-token", namespace="bearer")
            )

            mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_delete_with_namespace_search(self, mock_token_manager):
        """Test deleting token without specifying namespace."""
        # Store token in bearer namespace
        mock_token_manager.token_store.store_generic("my-token", "value123", "bearer")
        mock_token_manager.registry.register("my-token", TokenType.BEARER, "bearer")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            # Delete without namespace - should search common namespaces
            await token_delete_action_async(TokenDeleteParams(name="my-token"))

            mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_token(self, mock_token_manager):
        """Test deleting token that doesn't exist."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_delete_action_async(TokenDeleteParams(name="nonexistent"))

            mock_output.warning.assert_called()


class TestTokenClearAction:
    """Test token clear action."""

    @pytest.mark.asyncio
    async def test_clear_namespace_with_confirmation(self, mock_token_manager):
        """Test clearing tokens in a namespace."""
        # Store tokens
        mock_token_manager.token_store.store_generic("token1", "value1", "ns1")
        mock_token_manager.registry.register("token1", TokenType.BEARER, "ns1")
        mock_token_manager.token_store.store_generic("token2", "value2", "ns1")
        mock_token_manager.registry.register("token2", TokenType.BEARER, "ns1")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            with patch("chuk_term.ui.prompts.confirm", return_value=True):
                await token_clear_action_async(TokenClearParams(namespace="ns1"))

                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_clear_all_with_confirmation(self, mock_token_manager):
        """Test clearing all tokens."""
        # Store tokens
        mock_token_manager.token_store.store_generic("token1", "value1", "ns1")
        mock_token_manager.registry.register("token1", TokenType.BEARER, "ns1")
        mock_token_manager.token_store.store_generic("token2", "value2", "ns2")
        mock_token_manager.registry.register("token2", TokenType.BEARER, "ns2")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            with patch("chuk_term.ui.prompts.confirm", return_value=True):
                await token_clear_action_async(TokenClearParams())

                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_clear_cancelled(self, mock_token_manager):
        """Test canceling clear operation."""
        # Store token
        mock_token_manager.token_store.store_generic("token1", "value1", "ns1")
        mock_token_manager.registry.register("token1", TokenType.BEARER, "ns1")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            with patch("chuk_term.ui.prompts.confirm", return_value=False):
                await token_clear_action_async(TokenClearParams(namespace="ns1"))

                mock_output.warning.assert_called_with("Cancelled")

    @pytest.mark.asyncio
    async def test_clear_with_force(self, mock_token_manager):
        """Test clearing with force flag."""
        # Store token
        mock_token_manager.token_store.store_generic("token1", "value1", "ns1")
        mock_token_manager.registry.register("token1", TokenType.BEARER, "ns1")

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            # Force should not prompt
            await token_clear_action_async(
                TokenClearParams(namespace="ns1", force=True)
            )

            mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_clear_empty_namespace(self, mock_token_manager):
        """Test clearing when no tokens exist."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_clear_action_async(
                TokenClearParams(namespace="empty", force=True)
            )

            mock_output.warning.assert_called()


class TestTokenBackendsAction:
    """Test token backends action."""

    @pytest.mark.asyncio
    async def test_list_backends(self):
        """Test listing available backends."""
        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_backends_action_async()

            # Should print table of backends
            mock_output.print_table.assert_called()
            mock_output.info.assert_called()

    @pytest.mark.asyncio
    async def test_backends_shows_detected(self):
        """Test that detected backend is indicated."""

        with patch("mcp_cli.commands.actions.token.output") as mock_output:
            await token_backends_action_async()

            # Should show which backend is auto-detected
            mock_output.info.assert_called()


class TestErrorHandling:
    """Test error handling in token actions."""

    @pytest.mark.asyncio
    async def test_list_error(self):
        """Test list action error handling."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                with pytest.raises(Exception):
                    await token_list_action_async(TokenListParams())
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_set_error(self):
        """Test set action error handling."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                with pytest.raises(Exception):
                    await token_set_action_async(
                        TokenSetParams(
                            name="test", token_type="bearer", value="value123"
                        )
                    )
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_get_error(self):
        """Test get action error handling."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                with pytest.raises(Exception):
                    await token_get_action_async(name="test")
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_delete_error(self):
        """Test delete action error handling."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                with pytest.raises(Exception):
                    await token_delete_action_async(TokenDeleteParams(name="test"))
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_clear_error(self):
        """Test clear action error handling."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                with pytest.raises(Exception):
                    await token_clear_action_async(TokenClearParams(force=True))
                mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_backends_error(self):
        """Test backends action error handling."""
        with patch(
            "mcp_cli.commands.actions.token.TokenStoreFactory.get_available_backends",
            side_effect=Exception("Test error"),
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                with pytest.raises(Exception):
                    await token_backends_action_async()
                mock_output.error.assert_called()
