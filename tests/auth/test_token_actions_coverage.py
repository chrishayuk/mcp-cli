"""Additional tests for token actions to improve coverage."""

import pytest
from unittest.mock import patch, MagicMock
import time

from mcp_cli.commands.actions.token import (
    token_list_action_async,
    token_set_action_async,
    _get_token_manager,
)
from mcp_cli.commands.models import (
    TokenListParams,
    TokenSetParams,
)
from mcp_cli.auth import TokenType


class TestGetTokenManager:
    """Test _get_token_manager helper function."""

    def test_get_token_manager_with_config(self):
        """Test getting token manager with valid config."""
        mock_config = MagicMock()
        mock_config.token_store_backend = "encrypted_file"

        with patch(
            "mcp_cli.commands.actions.token.get_config", return_value=mock_config
        ):
            with patch("mcp_cli.commands.actions.token.TokenManager") as mock_tm:
                _get_token_manager()
                mock_tm.assert_called_once()

    def test_get_token_manager_with_config_error(self):
        """Test getting token manager when config raises exception."""
        with patch(
            "mcp_cli.commands.actions.token.get_config",
            side_effect=Exception("Config error"),
        ):
            with patch("mcp_cli.commands.actions.token.TokenManager") as mock_tm:
                _get_token_manager()
                # Should fall back to AUTO backend
                mock_tm.assert_called_once()


class TestTokenListActionCoverage:
    """Additional coverage tests for token_list_action_async."""

    @pytest.fixture
    def mock_token_manager(self, tmp_path):
        """Mock TokenManager with temporary storage."""
        from mcp_cli.auth import TokenManager, TokenStoreBackend

        manager = TokenManager(
            token_dir=tmp_path / "tokens",
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
            service_name="mcp-cli",
        )
        return manager

    @pytest.mark.asyncio
    async def test_list_with_provider_tokens(self, mock_token_manager):
        """Test listing with show_providers=True and provider tokens."""
        # Mock provider tokens
        mock_provider_tokens = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "in_env": False,
            },
            "anthropic": {
                "env_var": "ANTHROPIC_API_KEY",
                "in_env": True,
            },
        }

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens",
                return_value=mock_provider_tokens,
            ):
                with patch("mcp_cli.commands.actions.token.output"):
                    params = TokenListParams(show_providers=True)
                    await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_with_show_oauth(self, mock_token_manager):
        """Test listing with show_oauth=True."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                params = TokenListParams(show_oauth=True)
                await token_list_action_async(params)
                # Should show OAuth info message
                assert mock_output.info.called

    @pytest.mark.asyncio
    async def test_list_skip_provider_namespace(self, mock_token_manager):
        """Test that provider namespace is skipped when show_providers is True."""
        # Register a token in provider namespace
        mock_token_manager.registry.register(
            "test-token", TokenType.API_KEY, "provider"
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens", return_value={}
            ):
                with patch("mcp_cli.commands.actions.token.output"):
                    params = TokenListParams(show_providers=True)
                    await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_filter_bearer_tokens(self, mock_token_manager):
        """Test filtering bearer tokens when show_bearer=False."""
        # Register bearer token
        mock_token_manager.registry.register("bearer-token", TokenType.BEARER, "test")

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                params = TokenListParams(show_bearer=False)
                await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_filter_api_key_tokens(self, mock_token_manager):
        """Test filtering API key tokens when show_api_keys=False."""
        # Register API key token
        mock_token_manager.registry.register("api-key-token", TokenType.API_KEY, "test")

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                params = TokenListParams(show_api_keys=False)
                await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_with_expired_token(self, mock_token_manager):
        """Test listing token with expired timestamp."""
        # Register token with expired metadata
        mock_token_manager.registry.register(
            "expired-token",
            TokenType.BEARER,
            "test",
            metadata={"expires_at": time.time() - 3600},  # 1 hour ago
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                params = TokenListParams()
                await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_with_provider_metadata(self, mock_token_manager):
        """Test listing token with provider in metadata."""
        # Register token with provider metadata
        mock_token_manager.registry.register(
            "provider-token",
            TokenType.API_KEY,
            "test",
            metadata={"provider": "custom-provider"},
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                params = TokenListParams()
                await token_list_action_async(params)


class TestTokenSetActionCoverage:
    """Additional coverage tests for token_set_action_async."""

    @pytest.fixture
    def mock_token_manager(self, tmp_path):
        """Mock TokenManager."""
        from mcp_cli.auth import TokenManager, TokenStoreBackend

        manager = TokenManager(
            token_dir=tmp_path / "tokens",
            backend=TokenStoreBackend.ENCRYPTED_FILE,
            password="test-password",
            service_name="mcp-cli",
        )
        return manager

    @pytest.mark.asyncio
    async def test_set_generic_token_with_all_params(self, mock_token_manager):
        """Test setting generic token with all parameters."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                params = TokenSetParams(
                    name="my-custom-token",
                    value="token-value",
                    token_type="bearer",
                    namespace="custom",
                )
                await token_set_action_async(params)
