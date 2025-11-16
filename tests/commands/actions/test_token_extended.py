"""Extended tests for token action to reach 90%+ coverage."""

import pytest
import time
from unittest.mock import Mock, patch

from mcp_cli.commands.actions.token import (
    _get_token_manager,
    token_list_action_async,
    token_set_action_async,
    token_get_action_async,
    token_delete_action_async,
    token_set_provider_action_async,
    token_get_provider_action_async,
    token_delete_provider_action_async,
    token_clear_action_async,
    token_backends_action_async,
)
from mcp_cli.commands.models import (
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
    TokenProviderParams,
)
from mcp_cli.auth import TokenType, TokenStoreBackend


class TestGetTokenManager:
    """Tests for _get_token_manager function."""

    def test_get_token_manager_with_env_override(self):
        """Test getting token manager with environment variable override."""
        with patch.dict("os.environ", {"MCP_CLI_TOKEN_BACKEND": "keychain"}):
            with patch("mcp_cli.commands.actions.token.TokenManager") as mock_tm:
                _get_token_manager()

                # Verify TokenManager was called with keychain backend
                call_args = mock_tm.call_args
                assert call_args[1]["backend"] == TokenStoreBackend.KEYCHAIN

    def test_get_token_manager_with_invalid_env_override(self):
        """Test invalid environment variable falls back to config."""
        with patch.dict("os.environ", {"MCP_CLI_TOKEN_BACKEND": "invalid-backend"}):
            with patch("mcp_cli.commands.actions.token.get_config") as mock_config:
                mock_config.return_value.token_store_backend = "encrypted"
                with patch("mcp_cli.commands.actions.token.TokenManager") as mock_tm:
                    _get_token_manager()

                    # Should fall back to config
                    call_args = mock_tm.call_args
                    assert call_args[1]["backend"] == TokenStoreBackend.ENCRYPTED_FILE

    def test_get_token_manager_from_config(self):
        """Test getting token manager from config when no env override."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("mcp_cli.commands.actions.token.get_config") as mock_config:
                mock_config.return_value.token_store_backend = "encrypted"
                with patch("mcp_cli.commands.actions.token.TokenManager") as mock_tm:
                    _get_token_manager()

                    call_args = mock_tm.call_args
                    assert call_args[1]["backend"] == TokenStoreBackend.ENCRYPTED_FILE

    def test_get_token_manager_config_exception_fallback(self):
        """Test config exception falls back to AUTO."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "mcp_cli.commands.actions.token.get_config",
                side_effect=Exception("Config error"),
            ):
                with patch("mcp_cli.commands.actions.token.TokenManager") as mock_tm:
                    _get_token_manager()

                    call_args = mock_tm.call_args
                    assert call_args[1]["backend"] == TokenStoreBackend.AUTO


class TestTokenListExtended:
    """Extended tests for token_list_action_async."""

    @pytest.mark.asyncio
    async def test_list_provider_tokens(self):
        """Test listing provider tokens."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list_prov:
                mock_manager = Mock()
                mock_registry = Mock()
                mock_registry.list_tokens = Mock(return_value=[])
                mock_manager.registry = mock_registry
                mock_get_tm.return_value = mock_manager

                # Return provider tokens with env override
                mock_list_prov.return_value = {
                    "openai": {
                        "env_var": "OPENAI_API_KEY",
                        "in_env": True,
                    },
                    "anthropic": {
                        "env_var": "ANTHROPIC_API_KEY",
                        "in_env": False,
                    },
                }

                params = TokenListParams(show_providers=True, show_oauth=False)
                await token_list_action_async(params)

                mock_list_prov.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_registry_tokens_bearer(self):
        """Test listing bearer tokens from registry."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_registry = Mock()

            # Return bearer tokens
            mock_registry.list_tokens.return_value = [
                {
                    "type": "bearer",
                    "name": "test-token",
                    "namespace": "generic",
                    "registered_at": time.time(),
                    "metadata": {"expires_at": time.time() + 3600},
                }
            ]
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                show_bearer=True, show_api_keys=False, show_providers=False
            )
            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_registry_tokens_api_key(self):
        """Test listing API key tokens from registry."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_registry = Mock()

            # Return API key tokens
            mock_registry.list_tokens.return_value = [
                {
                    "type": "api-key",
                    "name": "test-api-key",
                    "namespace": "generic",
                    "registered_at": time.time(),
                    "metadata": {"provider": "custom-provider"},
                }
            ]
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                show_api_keys=True, show_bearer=False, show_providers=False
            )
            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_registry_tokens_expired(self):
        """Test listing expired tokens from registry."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_registry = Mock()

            # Return expired token
            mock_registry.list_tokens.return_value = [
                {
                    "type": "bearer",
                    "name": "expired-token",
                    "namespace": "generic",
                    "registered_at": time.time() - 7200,
                    "metadata": {
                        "expires_at": time.time() - 3600
                    },  # Expired 1 hour ago
                }
            ]
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(show_bearer=True, show_providers=False)
            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_no_tokens(self):
        """Test listing when no tokens found."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens", return_value={}
            ):
                mock_manager = Mock()
                mock_registry = Mock()
                mock_registry.list_tokens.return_value = []
                mock_manager.registry = mock_registry
                mock_get_tm.return_value = mock_manager

                params = TokenListParams(show_providers=True)
                await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_error_handling(self):
        """Test error handling in token list."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenListParams()

            with pytest.raises(Exception, match="Test error"):
                await token_list_action_async(params)


class TestTokenSet:
    """Tests for token_set_action_async."""

    @pytest.mark.asyncio
    async def test_set_bearer_token_with_value(self):
        """Test setting bearer token with provided value."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_registry = Mock()
            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenSetParams(
                name="test-bearer",
                token_type="bearer",
                value="bearer-token-123",
                namespace="generic",
            )

            await token_set_action_async(params)

            # Verify store and registry were called
            mock_store._store_raw.assert_called_once()
            mock_registry.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_bearer_token_prompt_for_value(self):
        """Test setting bearer token with prompted value."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch("getpass.getpass", return_value="prompted-token"):
                mock_manager = Mock()
                mock_store = Mock()
                mock_registry = Mock()
                mock_manager.token_store = mock_store
                mock_manager.registry = mock_registry
                mock_get_tm.return_value = mock_manager

                params = TokenSetParams(
                    name="test-bearer",
                    token_type="bearer",
                    value=None,  # Will be prompted
                    namespace="generic",
                )

                await token_set_action_async(params)

                mock_store._store_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_bearer_token_empty_value(self):
        """Test setting bearer token with empty value."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch("getpass.getpass", return_value=""):
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                params = TokenSetParams(
                    name="test-bearer",
                    token_type="bearer",
                    value=None,
                    namespace="generic",
                )

                await token_set_action_async(params)

                # Should not store anything

    @pytest.mark.asyncio
    async def test_set_api_key_token(self):
        """Test setting API key token."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_registry = Mock()
            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenSetParams(
                name="test-api-key",
                token_type="api-key",
                value="api-key-123",
                provider="custom-provider",
                namespace="generic",
            )

            await token_set_action_async(params)

            mock_store._store_raw.assert_called_once()
            mock_registry.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_api_key_token_no_provider(self):
        """Test setting API key token without provider."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_get_tm.return_value = mock_manager

            params = TokenSetParams(
                name="test-api-key",
                token_type="api-key",
                value="api-key-123",
                provider=None,
                namespace="generic",
            )

            await token_set_action_async(params)

    @pytest.mark.asyncio
    async def test_set_generic_token(self):
        """Test setting generic token."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_registry = Mock()
            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenSetParams(
                name="test-generic",
                token_type="generic",
                value="generic-token-123",
                namespace="custom",
            )

            await token_set_action_async(params)

            mock_store.store_generic.assert_called_once()
            mock_registry.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_unknown_token_type(self):
        """Test setting unknown token type."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_get_tm.return_value = mock_manager

            params = TokenSetParams(
                name="test-unknown",
                token_type="unknown-type",
                value="token-123",
                namespace="generic",
            )

            await token_set_action_async(params)

    @pytest.mark.asyncio
    async def test_set_error_handling(self):
        """Test error handling in token set."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenSetParams(
                name="test",
                token_type="bearer",
                value="token",
                namespace="generic",
            )

            with pytest.raises(Exception, match="Test error"):
                await token_set_action_async(params)


class TestTokenGet:
    """Tests for token_get_action_async."""

    @pytest.mark.asyncio
    async def test_get_token_success(self):
        """Test getting token successfully."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            from mcp_cli.auth import StoredToken

            mock_manager = Mock()
            mock_store = Mock()

            # Create a stored token
            stored = StoredToken(
                name="test-token",
                token_type=TokenType.BEARER,
                data={"token": "encrypted-value"},
                metadata={},
            )

            mock_store._retrieve_raw.return_value = stored.model_dump_json()
            mock_manager.token_store = mock_store
            mock_get_tm.return_value = mock_manager

            with patch("mcp_cli.commands.actions.token.output"):
                await token_get_action_async("test-token", "generic")

            mock_store._retrieve_raw.assert_called_once_with("generic:test-token")

    @pytest.mark.asyncio
    async def test_get_token_not_found(self):
        """Test getting token that doesn't exist."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_store._retrieve_raw.return_value = None
            mock_manager.token_store = mock_store
            mock_get_tm.return_value = mock_manager

            await token_get_action_async("missing-token", "generic")

    @pytest.mark.asyncio
    async def test_get_token_parse_error(self):
        """Test getting token with parse error."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_store._retrieve_raw.return_value = "invalid json"
            mock_manager.token_store = mock_store
            mock_get_tm.return_value = mock_manager

            await token_get_action_async("test-token", "generic")

    @pytest.mark.asyncio
    async def test_get_token_error_handling(self):
        """Test error handling in token get."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception, match="Test error"):
                await token_get_action_async("test-token", "generic")


class TestTokenDelete:
    """Tests for token_delete_action_async."""

    @pytest.mark.asyncio
    async def test_delete_oauth_token(self):
        """Test deleting OAuth token."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_manager.delete_tokens.return_value = True
            mock_get_tm.return_value = mock_manager

            params = TokenDeleteParams(name="test-server", oauth=True)
            await token_delete_action_async(params)

            mock_manager.delete_tokens.assert_called_once_with("test-server")

    @pytest.mark.asyncio
    async def test_delete_oauth_token_not_found(self):
        """Test deleting OAuth token that doesn't exist."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_manager.delete_tokens.return_value = False
            mock_get_tm.return_value = mock_manager

            params = TokenDeleteParams(name="test-server", oauth=True)
            await token_delete_action_async(params)

    @pytest.mark.asyncio
    async def test_delete_generic_token_with_namespace(self):
        """Test deleting generic token with specific namespace."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_store.delete_generic.return_value = True
            mock_registry = Mock()
            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenDeleteParams(
                name="test-token", namespace="custom", oauth=False
            )
            await token_delete_action_async(params)

            mock_store.delete_generic.assert_called_once_with("test-token", "custom")
            mock_registry.unregister.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_generic_token_search_all_namespaces(self):
        """Test deleting generic token by searching all namespaces."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()

            # Not found in first namespace, found in second
            mock_store.delete_generic.side_effect = [False, True]
            mock_registry = Mock()
            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenDeleteParams(name="test-token", namespace=None, oauth=False)
            await token_delete_action_async(params)

    @pytest.mark.asyncio
    async def test_delete_token_not_found(self):
        """Test deleting token that doesn't exist."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_store.delete_generic.return_value = False
            mock_manager.token_store = mock_store
            mock_get_tm.return_value = mock_manager

            params = TokenDeleteParams(
                name="missing-token", namespace="generic", oauth=False
            )
            await token_delete_action_async(params)

    @pytest.mark.asyncio
    async def test_delete_error_handling(self):
        """Test error handling in token delete."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenDeleteParams(name="test-token")

            with pytest.raises(Exception, match="Test error"):
                await token_delete_action_async(params)


class TestTokenSetProvider:
    """Tests for token_set_provider_action_async."""

    @pytest.mark.asyncio
    async def test_set_provider_token_with_value(self):
        """Test setting provider token with provided value."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=True
            ) as mock_set:
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name",
                    return_value="OPENAI_API_KEY",
                ):
                    with patch.dict("os.environ", {}, clear=True):
                        mock_manager = Mock()
                        mock_get_tm.return_value = mock_manager

                        params = TokenProviderParams(
                            provider="openai", api_key="sk-test123"
                        )
                        await token_set_provider_action_async(params)

                        mock_set.assert_called_once_with(
                            "openai", "sk-test123", mock_manager
                        )

    @pytest.mark.asyncio
    async def test_set_provider_token_prompt_for_value(self):
        """Test setting provider token with prompted value."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=True
            ):
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name",
                    return_value="ANTHROPIC_API_KEY",
                ):
                    with patch("getpass.getpass", return_value="prompted-key"):
                        with patch.dict("os.environ", {}, clear=True):
                            mock_manager = Mock()
                            mock_get_tm.return_value = mock_manager

                            params = TokenProviderParams(
                                provider="anthropic", api_key=None
                            )
                            await token_set_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_set_provider_token_empty_value(self):
        """Test setting provider token with empty value."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch("getpass.getpass", return_value=""):
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                params = TokenProviderParams(provider="openai", api_key=None)
                await token_set_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_set_provider_token_with_env_var_set(self):
        """Test setting provider token when env var is also set."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=True
            ):
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name",
                    return_value="OPENAI_API_KEY",
                ):
                    with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
                        mock_manager = Mock()
                        mock_get_tm.return_value = mock_manager

                        params = TokenProviderParams(
                            provider="openai", api_key="sk-test123"
                        )
                        await token_set_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_set_provider_token_failure(self):
        """Test provider token set failure."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=False
            ):
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                params = TokenProviderParams(provider="openai", api_key="sk-test123")
                await token_set_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_set_provider_error_handling(self):
        """Test error handling in set provider."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenProviderParams(provider="openai", api_key="sk-test123")

            with pytest.raises(Exception, match="Test error"):
                await token_set_provider_action_async(params)


class TestTokenGetProvider:
    """Tests for token_get_provider_action_async."""

    @pytest.mark.asyncio
    async def test_get_provider_token_configured(self):
        """Test getting provider token that is configured."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status"
            ) as mock_check:
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                mock_check.return_value = {
                    "has_token": True,
                    "source": "storage",
                    "env_var": "OPENAI_API_KEY",
                    "in_env": False,
                    "in_storage": True,
                }

                params = TokenProviderParams(provider="openai")
                await token_get_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_get_provider_token_not_configured(self):
        """Test getting provider token that is not configured."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status"
            ) as mock_check:
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                mock_check.return_value = {
                    "has_token": False,
                    "source": None,
                    "env_var": "ANTHROPIC_API_KEY",
                    "in_env": False,
                    "in_storage": False,
                }

                params = TokenProviderParams(provider="anthropic")
                await token_get_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_get_provider_error_handling(self):
        """Test error handling in get provider."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenProviderParams(provider="openai")

            with pytest.raises(Exception, match="Test error"):
                await token_get_provider_action_async(params)


class TestTokenDeleteProvider:
    """Tests for token_delete_provider_action_async."""

    @pytest.mark.asyncio
    async def test_delete_provider_token_configured(self):
        """Test deleting provider token that is configured."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status"
            ) as mock_check:
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                mock_check.return_value = {
                    "has_token": True,
                    "source": "storage",
                    "env_var": "OPENAI_API_KEY",
                    "in_env": False,
                    "in_storage": True,
                }

                params = TokenProviderParams(provider="openai")
                await token_delete_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_delete_provider_token_not_configured(self):
        """Test deleting provider token that is not configured."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status"
            ) as mock_check:
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                mock_check.return_value = {
                    "has_token": False,
                    "source": None,
                    "env_var": "OPENAI_API_KEY",
                    "in_env": False,
                    "in_storage": False,
                }

                params = TokenProviderParams(provider="openai")
                await token_delete_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_delete_provider_error_handling(self):
        """Test error handling in delete provider."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenProviderParams(provider="openai")

            with pytest.raises(Exception, match="Test error"):
                await token_delete_provider_action_async(params)


class TestTokenClear:
    """Tests for token_clear_action_async."""

    @pytest.mark.asyncio
    async def test_clear_tokens_with_force(self):
        """Test clearing tokens with force flag."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_registry = Mock()

            mock_registry.list_tokens.return_value = [
                {"name": "token1", "namespace": "generic"},
                {"name": "token2", "namespace": "generic"},
            ]
            mock_store.delete_generic.return_value = True

            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenClearParams(namespace="generic", force=True)
            await token_clear_action_async(params)

            # Should delete both tokens
            assert mock_store.delete_generic.call_count == 2
            mock_registry.clear_namespace.assert_called_once_with("generic")

    @pytest.mark.asyncio
    async def test_clear_all_tokens_with_force(self):
        """Test clearing all tokens with force flag."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_registry = Mock()

            mock_registry.list_tokens.return_value = [
                {"name": "token1", "namespace": "generic"},
            ]
            mock_store.delete_generic.return_value = True

            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenClearParams(namespace=None, force=True)
            await token_clear_action_async(params)

            mock_registry.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_tokens_cancelled(self):
        """Test clearing tokens when user cancels."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            with patch("chuk_term.ui.prompts.confirm", return_value=False):
                mock_manager = Mock()
                mock_get_tm.return_value = mock_manager

                params = TokenClearParams(namespace="generic", force=False)
                await token_clear_action_async(params)

    @pytest.mark.asyncio
    async def test_clear_tokens_no_tokens(self):
        """Test clearing tokens when none exist."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_registry = Mock()
            mock_registry.list_tokens.return_value = []
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenClearParams(namespace="generic", force=True)
            await token_clear_action_async(params)

    @pytest.mark.asyncio
    async def test_clear_tokens_partial_deletion(self):
        """Test clearing tokens when some deletions fail."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = Mock()
            mock_store = Mock()
            mock_registry = Mock()

            mock_registry.list_tokens.return_value = [
                {"name": "token1", "namespace": "generic"},
                {"name": "token2", "namespace": "generic"},
            ]
            # First deletion succeeds, second fails
            mock_store.delete_generic.side_effect = [True, False]

            mock_manager.token_store = mock_store
            mock_manager.registry = mock_registry
            mock_get_tm.return_value = mock_manager

            params = TokenClearParams(namespace="generic", force=True)
            await token_clear_action_async(params)

    @pytest.mark.asyncio
    async def test_clear_error_handling(self):
        """Test error handling in token clear."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            side_effect=Exception("Test error"),
        ):
            params = TokenClearParams(force=True)

            with pytest.raises(Exception, match="Test error"):
                await token_clear_action_async(params)


class TestTokenBackends:
    """Tests for token_backends_action_async."""

    @pytest.mark.asyncio
    async def test_backends_list_with_override(self):
        """Test listing backends with CLI override."""
        with patch.dict("os.environ", {"MCP_CLI_TOKEN_BACKEND": "keychain"}):
            with patch(
                "mcp_cli.commands.actions.token.TokenStoreFactory"
            ) as mock_factory:
                mock_factory.get_available_backends.return_value = [
                    TokenStoreBackend.KEYCHAIN,
                    TokenStoreBackend.ENCRYPTED_FILE,
                ]
                mock_factory._detect_backend.return_value = TokenStoreBackend.KEYCHAIN

                await token_backends_action_async()

    @pytest.mark.asyncio
    async def test_backends_list_invalid_override(self):
        """Test listing backends with invalid CLI override."""
        with patch.dict("os.environ", {"MCP_CLI_TOKEN_BACKEND": "invalid"}):
            with patch(
                "mcp_cli.commands.actions.token.TokenStoreFactory"
            ) as mock_factory:
                mock_factory.get_available_backends.return_value = [
                    TokenStoreBackend.ENCRYPTED_FILE,
                ]
                mock_factory._detect_backend.return_value = (
                    TokenStoreBackend.ENCRYPTED_FILE
                )

                await token_backends_action_async()

    @pytest.mark.asyncio
    async def test_backends_list_no_override(self):
        """Test listing backends without CLI override."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "mcp_cli.commands.actions.token.TokenStoreFactory"
            ) as mock_factory:
                mock_factory.get_available_backends.return_value = [
                    TokenStoreBackend.KEYCHAIN,
                ]
                mock_factory._detect_backend.return_value = TokenStoreBackend.KEYCHAIN

                await token_backends_action_async()

    @pytest.mark.asyncio
    async def test_backends_error_handling(self):
        """Test error handling in backends list."""
        with patch(
            "mcp_cli.commands.actions.token.TokenStoreFactory.get_available_backends",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception, match="Test error"):
                await token_backends_action_async()
