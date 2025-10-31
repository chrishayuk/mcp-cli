"""Tests for token action with new server_names parameter."""

import pytest
from unittest.mock import Mock, patch
import time

from mcp_cli.commands.models import TokenListParams
from mcp_cli.commands.actions.token import token_list_action_async
from mcp_cli.auth import OAuthTokens


def create_mock_manager(load_tokens_return=None):
    """Helper to create a properly mocked token manager."""
    mock_manager = Mock()
    mock_manager.load_tokens = Mock(return_value=load_tokens_return)
    mock_registry = Mock()
    mock_registry.list_tokens = Mock(return_value=[])  # No other tokens by default
    mock_manager.registry = mock_registry
    return mock_manager


class TestTokenListOAuthWithServers:
    """Test OAuth token listing with server_names."""

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_with_servers(self):
        """Test listing OAuth tokens when servers provided."""
        mock_tokens = OAuthTokens(
            access_token="test-token",
            token_type="bearer",
            expires_in=3600,
            issued_at=time.time(),
        )

        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager(mock_tokens)
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion", "github"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

            # Verify load_tokens was called for each server
            assert mock_manager.load_tokens.call_count == 2

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_with_expiration(self):
        """Test OAuth token listing shows expiration correctly."""
        issued_at = time.time() - 1800  # 30 minutes ago
        mock_tokens = OAuthTokens(
            access_token="test-token",
            token_type="bearer",
            expires_in=3600,
            issued_at=issued_at,
        )

        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager(mock_tokens)
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_expired(self):
        """Test OAuth token listing shows expired tokens."""
        issued_at = time.time() - 7200  # 2 hours ago
        mock_tokens = OAuthTokens(
            access_token="test-token",
            token_type="bearer",
            expires_in=3600,  # 1 hour expiry
            issued_at=issued_at,
        )

        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager(mock_tokens)
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_without_issued_at(self):
        """Test OAuth token listing when issued_at is None."""
        mock_tokens = OAuthTokens(
            access_token="test-token",
            token_type="bearer",
            expires_in=3600,
            issued_at=None,
        )

        with (
            patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm,
            patch("time.time", return_value=1234567890.0),
        ):
            mock_manager = create_mock_manager(mock_tokens)
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_no_expiration(self):
        """Test OAuth token listing when token has no expiration."""
        mock_tokens = OAuthTokens(
            access_token="test-token",
            token_type="bearer",
            expires_in=None,
            issued_at=time.time(),
        )

        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager(mock_tokens)
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_no_servers(self):
        """Test OAuth token listing when no servers configured."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager()
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=[],  # No servers
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_server_no_tokens(self):
        """Test OAuth token listing when server has no tokens."""
        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager(None)  # No tokens
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)

    @pytest.mark.asyncio
    async def test_list_oauth_tokens_with_registered_at(self):
        """Test OAuth token listing shows created date."""
        issued_at = time.time()
        mock_tokens = OAuthTokens(
            access_token="test-token",
            token_type="bearer",
            expires_in=3600,
            issued_at=issued_at,
        )

        with patch("mcp_cli.commands.actions.token._get_token_manager") as mock_get_tm:
            mock_manager = create_mock_manager(mock_tokens)
            mock_get_tm.return_value = mock_manager

            params = TokenListParams(
                server_names=["notion"],
                show_oauth=True,
                show_providers=False,
            )

            await token_list_action_async(params)
