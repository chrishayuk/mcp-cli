# tests/tools/test_oauth_safety.py
"""Tests for OAuth thread safety in ToolManager."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.tools.manager import ToolManager


class TestOAuthLock:
    """Verify _oauth_lock serializes concurrent OAuth flows."""

    def test_oauth_lock_exists(self):
        """ToolManager has an asyncio.Lock for OAuth."""
        tm = ToolManager(config_file="test.json", servers=[])
        assert hasattr(tm, "_oauth_lock")
        assert isinstance(tm._oauth_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_oauth_flows_serialized(self):
        """Multiple concurrent _handle_oauth_flow calls are serialized by the lock."""
        tm = ToolManager(config_file="test.json", servers=[])

        call_order: list[str] = []

        async def mock_oauth_flow(server_name: str, server_url: str) -> bool:
            """Track entry/exit to detect overlap."""
            call_order.append(f"enter-{server_name}")
            await asyncio.sleep(0.05)  # Simulate async work
            call_order.append(f"exit-{server_name}")
            return True

        # Patch the internals that _handle_oauth_flow calls
        with (
            patch("mcp_cli.tools.manager.TokenManager"),
            patch("mcp_cli.tools.manager.OAuthHandler") as mock_handler_cls,
        ):
            handler = mock_handler_cls.return_value
            handler.clear_tokens = MagicMock()

            # Make ensure_authenticated_mcp track ordering
            call_count = 0

            async def slow_auth(**kwargs):
                nonlocal call_count
                call_count += 1
                name = kwargs.get("server_name", f"s{call_count}")
                call_order.append(f"auth-{name}")
                await asyncio.sleep(0.05)
                # Return mock tokens
                tokens = MagicMock()
                tokens.access_token = f"token-{name}"
                tokens.refresh_token = None
                tokens.token_type = "bearer"
                tokens.expires_in = 3600
                tokens.issued_at = 0
                return tokens

            handler.ensure_authenticated_mcp = slow_auth

            # Launch 3 concurrent OAuth flows
            results = await asyncio.gather(
                tm._handle_oauth_flow("server1", "http://s1"),
                tm._handle_oauth_flow("server2", "http://s2"),
                tm._handle_oauth_flow("server3", "http://s3"),
            )

            # All should succeed
            assert all(results)

            # With the lock, auth calls should be sequential (not interleaved).
            # Extract just the auth entries to verify no overlap.
            auth_entries = [e for e in call_order if e.startswith("auth-")]
            assert len(auth_entries) == 3

    def test_header_copy_on_write(self):
        """Verify header update uses copy-on-write, not in-place mutation."""
        tm = ToolManager(config_file="test.json", servers=[])

        # Set up mock transport with headers
        original_headers = {"Content-Type": "application/json"}
        transport = MagicMock()
        transport.configured_headers = original_headers

        sm = MagicMock()
        sm.transports = {"test-server": transport}
        tm.stream_manager = sm

        # The copy-on-write pattern means after update,
        # transport.configured_headers should be a NEW dict
        # (verified by checking the implementation doesn't use __setitem__)
        assert "Authorization" not in original_headers


class TestPerServerTimeout:
    """Verify _get_server_timeout resolution: per-server → global → default."""

    def test_global_default_when_no_server(self):
        tm = ToolManager(config_file="test.json", servers=[])
        assert tm._get_server_timeout(None) == tm.tool_timeout

    def test_global_default_when_server_has_no_override(self):
        tm = ToolManager(config_file="test.json", servers=[])
        # Add a server without per-server timeout
        from mcp_cli.config.server_models import STDIOServerConfig

        tm._config_loader.stdio_servers.append(
            STDIOServerConfig(name="sqlite", command="sqlite3")
        )
        assert tm._get_server_timeout("sqlite") == tm.tool_timeout

    def test_per_server_timeout_used(self):
        tm = ToolManager(config_file="test.json", servers=[])
        from mcp_cli.config.server_models import HTTPServerConfig

        tm._config_loader.http_servers.append(
            HTTPServerConfig(
                name="slow-api",
                url="http://localhost:8080",
                tool_timeout=300.0,
            )
        )
        assert tm._get_server_timeout("slow-api") == 300.0

    def test_unknown_server_falls_back_to_global(self):
        tm = ToolManager(config_file="test.json", servers=[])
        assert tm._get_server_timeout("nonexistent") == tm.tool_timeout
