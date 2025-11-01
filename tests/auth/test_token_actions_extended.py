"""Extended tests for token actions to reach 90%+ coverage."""

import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.actions.token import (
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
    TokenSetParams,
    TokenDeleteParams,
    TokenProviderParams,
    TokenClearParams,
)


class TestTokenSetExtended:
    """Extended tests for token_set_action_async."""

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
    async def test_set_bearer_token_with_expiration(self, mock_token_manager):
        """Test setting bearer token that has expires_at."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output"):
                # BearerToken with expires_in will have expires_at
                params = TokenSetParams(
                    name="bearer-with-exp",
                    value="bearer-value",
                    token_type="bearer",
                    namespace="test",
                )
                await token_set_action_async(params)


class TestTokenGetExtended:
    """Extended tests for token_get_action_async."""

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
    async def test_get_token_parsing_error(self, mock_token_manager):
        """Test when token data can't be parsed."""
        # Mock store to return invalid JSON
        mock_token_manager.token_store._retrieve_raw = MagicMock(
            return_value="invalid json"
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                await token_get_action_async("test-token", "test")
                # Should show warning about parsing
                assert any(
                    "Could not parse" in str(call)
                    for call in mock_output.warning.call_args_list
                )


class TestTokenDeleteExtended:
    """Extended tests for token_delete_action_async."""

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
    async def test_delete_oauth_token_found(self, mock_token_manager):
        """Test deleting OAuth token that exists."""
        mock_token_manager.delete_tokens = MagicMock(return_value=True)

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                params = TokenDeleteParams(name="server-name", oauth=True)
                await token_delete_action_async(params)
                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_delete_oauth_token_not_found(self, mock_token_manager):
        """Test deleting OAuth token that doesn't exist."""
        mock_token_manager.delete_tokens = MagicMock(return_value=False)

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                params = TokenDeleteParams(name="server-name", oauth=True)
                await token_delete_action_async(params)
                mock_output.warning.assert_called()


class TestTokenProviderSetExtended:
    """Extended tests for token_set_provider_action_async."""

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
    async def test_set_provider_with_getpass(self, mock_token_manager):
        """Test setting provider token using getpass."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=True
            ):
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name",
                    return_value="OPENAI_API_KEY",
                ):
                    with patch("getpass.getpass", return_value="secret-key"):
                        with patch("mcp_cli.commands.actions.token.output"):
                            params = TokenProviderParams(provider="openai")
                            await token_set_provider_action_async(params)

    @pytest.mark.asyncio
    async def test_set_provider_empty_api_key(self, mock_token_manager):
        """Test setting provider token with empty API key."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("getpass.getpass", return_value=""):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenProviderParams(provider="openai")
                    await token_set_provider_action_async(params)
                    mock_output.error.assert_called()

    @pytest.mark.asyncio
    async def test_set_provider_with_env_var_set(self, mock_token_manager):
        """Test setting provider token when env var is also set."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=True
            ):
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name",
                    return_value="OPENAI_API_KEY",
                ):
                    with patch("os.environ.get", return_value="env-value"):
                        with patch(
                            "mcp_cli.commands.actions.token.output"
                        ) as mock_output:
                            params = TokenProviderParams(
                                provider="openai", api_key="test-key"
                            )
                            await token_set_provider_action_async(params)
                            # Should show warning about env var precedence
                            assert mock_output.warning.called

    @pytest.mark.asyncio
    async def test_set_provider_failure(self, mock_token_manager):
        """Test when setting provider token fails."""
        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token", return_value=False
            ):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenProviderParams(provider="openai", api_key="test-key")
                    await token_set_provider_action_async(params)
                    mock_output.error.assert_called()


class TestTokenProviderGetExtended:
    """Extended tests for token_get_provider_action_async."""

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
    async def test_get_provider_with_token(self, mock_token_manager):
        """Test getting provider info when token exists."""
        mock_status = {
            "has_token": True,
            "source": "storage",
            "env_var": "OPENAI_API_KEY",
            "in_env": False,
            "in_storage": True,
        }

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status",
                return_value=mock_status,
            ):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenProviderParams(provider="openai")
                    await token_get_provider_action_async(params)
                    mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_get_provider_without_token(self, mock_token_manager):
        """Test getting provider info when no token exists."""
        mock_status = {
            "has_token": False,
            "source": None,
            "env_var": "OPENAI_API_KEY",
            "in_env": False,
            "in_storage": False,
        }

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status",
                return_value=mock_status,
            ):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenProviderParams(provider="openai")
                    await token_get_provider_action_async(params)
                    # Should show instructions on how to set
                    assert any(
                        "To set API key" in str(call)
                        for call in mock_output.info.call_args_list
                    )


class TestTokenProviderDeleteExtended:
    """Extended tests for token_delete_provider_action_async."""

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
    async def test_delete_provider_with_token(self, mock_token_manager):
        """Test delete provider display when token exists."""
        mock_status = {
            "has_token": True,
            "source": "storage",
            "env_var": "OPENAI_API_KEY",
            "in_env": False,
            "in_storage": True,
        }

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status",
                return_value=mock_status,
            ):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenProviderParams(provider="openai")
                    await token_delete_provider_action_async(params)
                    mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_delete_provider_without_token(self, mock_token_manager):
        """Test delete provider display when no token exists."""
        mock_status = {
            "has_token": False,
            "source": None,
            "env_var": "OPENAI_API_KEY",
            "in_env": False,
            "in_storage": False,
        }

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status",
                return_value=mock_status,
            ):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenProviderParams(provider="openai")
                    await token_delete_provider_action_async(params)
                    # Should show instructions
                    assert any(
                        "To set API key" in str(call)
                        for call in mock_output.info.call_args_list
                    )


class TestTokenClearExtended:
    """Extended tests for token_clear_action_async."""

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
    async def test_clear_with_no_tokens(self, mock_token_manager):
        """Test clearing when no tokens exist."""
        mock_token_manager.registry.list_tokens = MagicMock(return_value=[])

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                params = TokenClearParams(force=True)
                await token_clear_action_async(params)
                mock_output.warning.assert_called()

    @pytest.mark.asyncio
    async def test_clear_cancelled(self, mock_token_manager):
        """Test clearing when user cancels."""
        mock_token_manager.registry.list_tokens = MagicMock(
            return_value=[{"name": "token1", "namespace": "test"}]
        )

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("chuk_term.ui.prompts.confirm", return_value=False):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    params = TokenClearParams()
                    await token_clear_action_async(params)
                    mock_output.warning.assert_called_with("Cancelled")

    @pytest.mark.asyncio
    async def test_clear_no_tokens_deleted(self, mock_token_manager):
        """Test clearing when no tokens can be deleted."""
        mock_token_manager.registry.list_tokens = MagicMock(
            return_value=[{"name": "token1", "namespace": "test"}]
        )
        mock_token_manager.token_store.delete_generic = MagicMock(return_value=False)

        with patch(
            "mcp_cli.commands.actions.token._get_token_manager",
            return_value=mock_token_manager,
        ):
            with patch("mcp_cli.commands.actions.token.output") as mock_output:
                params = TokenClearParams(force=True)
                await token_clear_action_async(params)
                # Should show warning
                assert any(
                    "No tokens" in str(call)
                    for call in mock_output.warning.call_args_list
                )


class TestTokenBackends:
    """Tests for token_backends_action_async."""

    @pytest.mark.asyncio
    async def test_backends_listing(self):
        """Test listing token storage backends."""
        from mcp_cli.auth import TokenStoreBackend

        mock_available = [TokenStoreBackend.ENCRYPTED_FILE, TokenStoreBackend.KEYCHAIN]
        mock_detected = TokenStoreBackend.ENCRYPTED_FILE

        with patch(
            "mcp_cli.commands.actions.token.TokenStoreFactory.get_available_backends",
            return_value=mock_available,
        ):
            with patch(
                "mcp_cli.commands.actions.token.TokenStoreFactory._detect_backend",
                return_value=mock_detected,
            ):
                with patch("mcp_cli.commands.actions.token.output") as mock_output:
                    await token_backends_action_async()
                    mock_output.print_table.assert_called()
