# tests/tools/test_tool_manager_extended.py
"""Extended tests for ToolManager to achieve >90% coverage.

Covers missing lines: 74, 168-182, 302-304, 358, 432-433, 440,
476-485, 497-570, 623-632, 648-676, 895, 898, 1010-1012.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.tools.manager import (
    ToolManager,
    _is_oauth_error,
    OAUTH_ERROR_PATTERNS,
)
from mcp_cli.tools.models import ToolInfo, TransportType


# ────────────────────────────────────────────────────────────────────
# _is_oauth_error helper (line 74: empty string branch)
# ────────────────────────────────────────────────────────────────────


class TestIsOAuthError:
    """Test the _is_oauth_error helper function."""

    def test_empty_string_returns_false(self):
        """Line 74: empty error message returns False."""
        assert _is_oauth_error("") is False

    def test_none_returns_false(self):
        """Line 74: None-ish error message returns False."""
        assert _is_oauth_error(None) is False  # type: ignore[arg-type]

    def test_oauth_patterns_detected(self):
        """Various OAuth error patterns are detected."""
        for pattern in OAUTH_ERROR_PATTERNS:
            assert _is_oauth_error(f"Error: {pattern}") is True

    def test_non_oauth_error_returns_false(self):
        """Non-OAuth errors return False."""
        assert _is_oauth_error("Connection timed out") is False
        assert _is_oauth_error("File not found") is False

    def test_case_insensitive_detection(self):
        """Detection is case-insensitive."""
        assert _is_oauth_error("REQUIRES OAUTH AUTHORIZATION") is True
        assert _is_oauth_error("Unauthorized") is True


# ────────────────────────────────────────────────────────────────────
# initialize() - lines 168-182
# Full initialization path with successful config and stream manager
# ────────────────────────────────────────────────────────────────────


class TestToolManagerInitializeFull:
    """Test the full initialize() flow with config detection and stream manager."""

    @pytest.mark.asyncio
    async def test_initialize_success_with_registry_and_processor(self, tmp_path):
        """Lines 168-182: Successful init sets registry and processor from stream_manager."""
        config = {"mcpServers": {"test_http": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["test_http"])

        mock_registry = MagicMock()
        mock_processor = MagicMock()

        with patch("mcp_cli.tools.manager.StreamManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.initialize_with_http_streamable = AsyncMock()
            mock_sm.registry = mock_registry
            mock_sm.processor = mock_processor
            mock_sm.enable_middleware = MagicMock()
            MockSM.return_value = mock_sm

            with patch("chuk_term.ui.output"):
                result = await tm.initialize()

        assert result is True
        assert tm._registry is mock_registry
        assert tm.processor is mock_processor

    @pytest.mark.asyncio
    async def test_initialize_success_no_registry_attr(self, tmp_path):
        """Lines 175-178: stream_manager without registry/processor attributes."""
        config = {"mcpServers": {"test_http": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["test_http"])

        with patch("mcp_cli.tools.manager.StreamManager") as MockSM:
            mock_sm = MagicMock(
                spec=[
                    "initialize_with_http_streamable",
                    "enable_middleware",
                ]
            )
            mock_sm.initialize_with_http_streamable = AsyncMock()
            mock_sm.enable_middleware = MagicMock()
            MockSM.return_value = mock_sm

            with patch("chuk_term.ui.output"):
                result = await tm.initialize()

        assert result is True
        # registry and processor should not be set since stream_manager lacks those attrs
        assert tm._registry is None
        assert tm.processor is None

    @pytest.mark.asyncio
    async def test_initialize_stream_manager_returns_false(self, tmp_path):
        """Line 173: success is False when _initialize_stream_manager fails."""
        config = {"mcpServers": {"test_http": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["test_http"])

        # Make _initialize_stream_manager raise to return False
        with patch.object(
            tm, "_initialize_stream_manager", new_callable=AsyncMock, return_value=False
        ):
            with patch("chuk_term.ui.output"):
                result = await tm.initialize()

        assert result is False


# ────────────────────────────────────────────────────────────────────
# _initialize_stream_manager exception handling - lines 302-304
# ────────────────────────────────────────────────────────────────────


class TestInitializeStreamManagerException:
    """Test _initialize_stream_manager when outer exception occurs."""

    @pytest.mark.asyncio
    async def test_stream_manager_init_outer_exception(self, tmp_path):
        """Lines 302-304: Exception in _initialize_stream_manager returns False."""
        config = {"mcpServers": {"test_http": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["test_http"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        # Patch create_oauth_refresh_callback to raise after StreamManager is created
        with patch("mcp_cli.tools.manager.StreamManager") as MockSM:
            MockSM.return_value = MagicMock()
            with patch.object(
                tm._config_loader,
                "create_oauth_refresh_callback",
                side_effect=RuntimeError("callback creation failed"),
            ):
                result = await tm._initialize_stream_manager("stdio")

        assert result is False


# ────────────────────────────────────────────────────────────────────
# get_all_tools with no stream_manager and no registry - line 358
# ────────────────────────────────────────────────────────────────────


class TestGetAllToolsNoStreamManager:
    """Test get_all_tools returns [] when stream_manager is None."""

    @pytest.mark.asyncio
    async def test_get_all_tools_no_stream_manager_no_registry(self):
        """Line 358: stream_manager is None and _registry is None -> []."""
        tm = ToolManager(config_file="test.json", servers=[])
        tm.stream_manager = None
        tm._registry = None

        result = await tm.get_all_tools()
        assert result == []


# ────────────────────────────────────────────────────────────────────
# format_tool_response edge cases - lines 432-433, 440
# ────────────────────────────────────────────────────────────────────


class TestFormatToolResponseEdgeCases:
    """Test format_tool_response with edge-case inputs."""

    def test_format_text_blocks_model_validate_fails(self):
        """Lines 432-433: TextContent.model_validate fails, falls through to dict check."""
        # Items that have type=text but with extra invalid fields that cause
        # model_validate to raise. Use a non-dict item mixed with text items
        # so model_validate fails on the comprehension.
        payload = [{"type": "text"}]  # missing 'text' field entirely
        result = ToolManager.format_tool_response(payload)
        # Should fall through - either model_validate raises or text_blocks is empty
        # Then hits line 440 (all items have type=text) and returns joined text
        assert result is not None

    def test_format_mixed_text_and_non_text_items(self):
        """Line 440: list of dicts all with type=text but using fallback path."""
        # This tests the branch at line 440 where all items have type=text
        # but TextContent model_validate failed earlier
        payload = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        result = ToolManager.format_tool_response(payload)
        assert "hello" in result
        assert "world" in result

    def test_format_list_with_non_dict_items(self):
        """List items that are not dicts go to json.dumps."""
        payload = [1, 2, 3]
        result = ToolManager.format_tool_response(payload)
        assert json.loads(result) == [1, 2, 3]

    def test_format_empty_text_in_text_blocks(self):
        """Line 440: text blocks with empty text field."""
        payload = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "data"},
        ]
        result = ToolManager.format_tool_response(payload)
        assert "data" in result


# ────────────────────────────────────────────────────────────────────
# _get_server_url - lines 476-485
# ────────────────────────────────────────────────────────────────────


class TestGetServerUrl:
    """Test _get_server_url method."""

    def test_get_server_url_http_found(self, tmp_path):
        """Lines 476-478: HTTP server URL is found."""
        config = {"mcpServers": {"my_http": {"url": "https://http.example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["my_http"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        url = tm._get_server_url("my_http")
        assert url == "https://http.example.com/mcp"

    def test_get_server_url_sse_found(self, tmp_path):
        """Lines 481-483: SSE server URL is found."""
        config = {
            "mcpServers": {
                "my_sse": {
                    "url": "https://sse.example.com",
                    "transport": "sse",
                }
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["my_sse"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        url = tm._get_server_url("my_sse")
        assert url == "https://sse.example.com"

    def test_get_server_url_not_found(self):
        """Line 485: no matching server returns None."""
        tm = ToolManager(config_file="test.json", servers=[])
        url = tm._get_server_url("nonexistent")
        assert url is None

    def test_get_server_url_stdio_not_found(self, tmp_path):
        """STDIO servers don't have URLs, so they return None."""
        config = {
            "mcpServers": {"my_stdio": {"command": "python", "args": ["-m", "server"]}}
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["my_stdio"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        url = tm._get_server_url("my_stdio")
        assert url is None


# ────────────────────────────────────────────────────────────────────
# _handle_oauth_flow - lines 497-570
# ────────────────────────────────────────────────────────────────────


class TestHandleOAuthFlow:
    """Test _handle_oauth_flow method."""

    @pytest.mark.asyncio
    async def test_handle_oauth_flow_success_with_transport_update(self):
        """Lines 497-555: Successful OAuth flow stores tokens and updates transport."""
        tm = ToolManager(config_file="test.json", servers=[])

        # Mock stream_manager with transports
        mock_transport = MagicMock()
        mock_transport.configured_headers = {}

        mock_sm = MagicMock()
        mock_sm.transports = {"test_server": mock_transport}
        tm.stream_manager = mock_sm

        mock_tokens = MagicMock()
        mock_tokens.access_token = "new_access_token"
        mock_tokens.refresh_token = "new_refresh_token"
        mock_tokens.token_type = "Bearer"
        mock_tokens.expires_in = 3600
        mock_tokens.issued_at = 1234567890

        with (
            patch("mcp_cli.tools.manager.TokenManager") as MockTM,
            patch("mcp_cli.tools.manager.OAuthHandler") as MockOH,
            patch("chuk_term.ui.output"),
        ):
            mock_token_manager = MagicMock()
            mock_token_store = MagicMock()
            mock_token_manager.token_store = mock_token_store
            MockTM.return_value = mock_token_manager

            mock_oauth = MockOH.return_value
            mock_oauth.clear_tokens = MagicMock()
            mock_oauth.ensure_authenticated_mcp = AsyncMock(return_value=mock_tokens)

            result = await tm._handle_oauth_flow("test_server", "https://example.com")

        assert result is True
        # Transport headers should be updated
        assert (
            mock_transport.configured_headers["Authorization"]
            == "Bearer new_access_token"
        )
        # Token store should be called
        mock_token_store._store_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_oauth_flow_no_tokens(self):
        """Lines 556-560: OAuth flow returns no valid tokens."""
        tm = ToolManager(config_file="test.json", servers=[])

        with (
            patch("mcp_cli.tools.manager.TokenManager") as MockTM,
            patch("mcp_cli.tools.manager.OAuthHandler") as MockOH,
            patch("chuk_term.ui.output"),
        ):
            mock_token_manager = MagicMock()
            MockTM.return_value = mock_token_manager

            mock_oauth = MockOH.return_value
            mock_oauth.clear_tokens = MagicMock()
            mock_oauth.ensure_authenticated_mcp = AsyncMock(return_value=None)

            result = await tm._handle_oauth_flow("test_server", "https://example.com")

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_oauth_flow_tokens_no_access_token(self):
        """Lines 556-560: tokens returned but no access_token."""
        tm = ToolManager(config_file="test.json", servers=[])

        mock_tokens = MagicMock()
        mock_tokens.access_token = None

        with (
            patch("mcp_cli.tools.manager.TokenManager") as MockTM,
            patch("mcp_cli.tools.manager.OAuthHandler") as MockOH,
            patch("chuk_term.ui.output"),
        ):
            mock_token_manager = MagicMock()
            MockTM.return_value = mock_token_manager

            mock_oauth = MockOH.return_value
            mock_oauth.clear_tokens = MagicMock()
            mock_oauth.ensure_authenticated_mcp = AsyncMock(return_value=mock_tokens)

            result = await tm._handle_oauth_flow("test_server", "https://example.com")

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_oauth_flow_exception(self):
        """Lines 562-570: Exception during OAuth flow."""
        tm = ToolManager(config_file="test.json", servers=[])

        with (
            patch(
                "mcp_cli.tools.manager.TokenManager",
                side_effect=RuntimeError("Token error"),
            ),
            patch("chuk_term.ui.output"),
        ):
            result = await tm._handle_oauth_flow("test_server", "https://example.com")

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_oauth_flow_exception_no_output_module(self):
        """Lines 564-569: Exception during OAuth flow, chuk_term not importable."""
        tm = ToolManager(config_file="test.json", servers=[])

        with (
            patch(
                "mcp_cli.tools.manager.TokenManager",
                side_effect=RuntimeError("Token error"),
            ),
            patch("chuk_term.ui.output", side_effect=ImportError("no module")),
        ):
            result = await tm._handle_oauth_flow("test_server", "https://example.com")

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_oauth_flow_no_transport_attr(self):
        """Lines 545-551: stream_manager exists but has no transports attribute."""
        tm = ToolManager(config_file="test.json", servers=[])

        mock_sm = MagicMock(spec=[])  # No transports attribute
        tm.stream_manager = mock_sm

        mock_tokens = MagicMock()
        mock_tokens.access_token = "token123"
        mock_tokens.refresh_token = "refresh456"
        mock_tokens.token_type = "Bearer"
        mock_tokens.expires_in = 3600
        mock_tokens.issued_at = 1234567890

        with (
            patch("mcp_cli.tools.manager.TokenManager") as MockTM,
            patch("mcp_cli.tools.manager.OAuthHandler") as MockOH,
            patch("chuk_term.ui.output"),
        ):
            mock_token_manager = MagicMock()
            mock_token_store = MagicMock()
            mock_token_manager.token_store = mock_token_store
            MockTM.return_value = mock_token_manager

            mock_oauth = MockOH.return_value
            mock_oauth.clear_tokens = MagicMock()
            mock_oauth.ensure_authenticated_mcp = AsyncMock(return_value=mock_tokens)

            result = await tm._handle_oauth_flow("test_server", "https://example.com")

        assert result is True


# ────────────────────────────────────────────────────────────────────
# execute_tool - OAuth error in result (lines 623-632)
# ────────────────────────────────────────────────────────────────────


class TestExecuteToolOAuthInResult:
    """Test execute_tool OAuth error detection in successful tool result."""

    @pytest.mark.asyncio
    async def test_execute_tool_result_contains_oauth_error(self, tmp_path):
        """Lines 622-638: OAuth error detected in result string, triggers re-auth."""
        config = {"mcpServers": {"http_srv": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["http_srv"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        mock_sm = MagicMock()
        # First call returns OAuth error, second call returns success
        mock_sm.call_tool = AsyncMock(
            side_effect=[
                {"error": "requires OAuth authorization"},
                {"data": "success"},
            ]
        )
        tm.stream_manager = mock_sm

        # Mock get_server_for_tool to return the server name
        tm.get_server_for_tool = AsyncMock(return_value="http_srv")

        # Mock _handle_oauth_flow to succeed
        tm._handle_oauth_flow = AsyncMock(return_value=True)

        result = await tm.execute_tool("my_tool", {"arg": "val"})

        assert result.success is True
        assert result.result == {"data": "success"}
        tm._handle_oauth_flow.assert_called_once_with(
            "http_srv", "https://example.com/mcp"
        )

    @pytest.mark.asyncio
    async def test_execute_tool_result_oauth_error_no_server_url(self, tmp_path):
        """Lines 627-628: OAuth error but server URL not found."""
        config = {"mcpServers": {}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=[])

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(
            return_value={"error": "requires OAuth authorization"}
        )
        tm.stream_manager = mock_sm
        tm.get_server_for_tool = AsyncMock(return_value="some_server")

        result = await tm.execute_tool("my_tool", {})

        # Should return success (the result was technically returned, just contained oauth text)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_result_oauth_error_no_server_name(self):
        """Lines 626: OAuth error but server name not found."""
        tm = ToolManager(config_file="test.json", servers=[])

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(return_value="requires OAuth authorization")
        tm.stream_manager = mock_sm
        tm.get_server_for_tool = AsyncMock(return_value=None)

        result = await tm.execute_tool("my_tool", {})

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_result_oauth_retry_flag_prevents_loop(self):
        """OAuth error with _oauth_retry=True does not trigger re-auth."""
        tm = ToolManager(config_file="test.json", servers=[])

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(return_value="requires OAuth authorization")
        tm.stream_manager = mock_sm

        result = await tm.execute_tool("my_tool", {}, _oauth_retry=True)

        # Should just return the result without attempting OAuth
        assert result.success is True


# ────────────────────────────────────────────────────────────────────
# execute_tool - OAuth error in exception (lines 648-676)
# ────────────────────────────────────────────────────────────────────


class TestExecuteToolOAuthInException:
    """Test execute_tool OAuth error detection in exceptions."""

    @pytest.mark.asyncio
    async def test_execute_tool_exception_oauth_error_retry_success(self, tmp_path):
        """Lines 648-666: OAuth error in exception, re-auth succeeds, retry succeeds."""
        config = {"mcpServers": {"http_srv": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["http_srv"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        mock_sm = MagicMock()
        # First call raises OAuth error, second call succeeds
        mock_sm.call_tool = AsyncMock(
            side_effect=[
                RuntimeError("requires OAuth authorization"),
                {"data": "success_after_oauth"},
            ]
        )
        tm.stream_manager = mock_sm
        tm.get_server_for_tool = AsyncMock(return_value="http_srv")
        tm._handle_oauth_flow = AsyncMock(return_value=True)

        result = await tm.execute_tool("my_tool", {"arg": "val"})

        assert result.success is True
        assert result.result == {"data": "success_after_oauth"}

    @pytest.mark.asyncio
    async def test_execute_tool_exception_oauth_error_auth_fails(self, tmp_path):
        """Lines 667-672: OAuth error in exception, re-auth fails."""
        config = {"mcpServers": {"http_srv": {"url": "https://example.com/mcp"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["http_srv"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(
            side_effect=RuntimeError("requires OAuth authorization")
        )
        tm.stream_manager = mock_sm
        tm.get_server_for_tool = AsyncMock(return_value="http_srv")
        tm._handle_oauth_flow = AsyncMock(return_value=False)

        result = await tm.execute_tool("my_tool", {})

        assert result.success is False
        assert "OAuth authentication failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_exception_oauth_error_no_server_url(self, tmp_path):
        """Lines 673-674: OAuth error in exception but no server URL found."""
        config = {"mcpServers": {}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=[])

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(side_effect=RuntimeError("unauthorized"))
        tm.stream_manager = mock_sm
        tm.get_server_for_tool = AsyncMock(return_value="some_server")

        result = await tm.execute_tool("my_tool", {})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_tool_exception_oauth_error_no_server_name(self):
        """Lines 675-676: OAuth error in exception but server name not found."""
        tm = ToolManager(config_file="test.json", servers=[])

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(side_effect=RuntimeError("unauthorized"))
        tm.stream_manager = mock_sm
        tm.get_server_for_tool = AsyncMock(return_value=None)

        result = await tm.execute_tool("my_tool", {})

        assert result.success is False
        assert "unauthorized" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_exception_oauth_retry_flag(self):
        """OAuth error in exception with _oauth_retry=True skips re-auth."""
        tm = ToolManager(config_file="test.json", servers=[])

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(side_effect=RuntimeError("unauthorized"))
        tm.stream_manager = mock_sm

        result = await tm.execute_tool("my_tool", {}, _oauth_retry=True)

        assert result.success is False
        assert "unauthorized" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_exception_oauth_with_namespace(self, tmp_path):
        """OAuth retry uses the provided namespace instead of looking it up."""
        config = {"mcpServers": {"ns_server": {"url": "https://ns.example.com"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["ns_server"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        mock_sm = MagicMock()
        mock_sm.call_tool = AsyncMock(
            side_effect=[
                RuntimeError("requires OAuth authorization"),
                {"result": "ok"},
            ]
        )
        tm.stream_manager = mock_sm
        tm._handle_oauth_flow = AsyncMock(return_value=True)

        result = await tm.execute_tool("my_tool", {}, namespace="ns_server")

        assert result.success is True
        tm._handle_oauth_flow.assert_called_once_with(
            "ns_server", "https://ns.example.com"
        )


# ────────────────────────────────────────────────────────────────────
# validate_single_tool - lines 895, 898
# ────────────────────────────────────────────────────────────────────


class TestValidateSingleToolCoverage:
    """Test validate_single_tool return paths for valid/invalid tools."""

    @pytest.mark.asyncio
    async def test_validate_single_tool_valid_returns_true_none(self):
        """Line 895: valid tool returns (True, None)."""
        tm = ToolManager(config_file="test.json", servers=[])
        tool = ToolInfo(
            name="good_tool",
            namespace="ns",
            description="A well-described tool",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string", "description": "An input"}},
            },
        )
        tm.get_all_tools = AsyncMock(return_value=[tool])

        # Mock filter_tools to return the tool as valid
        tm.tool_filter.filter_tools = MagicMock(
            return_value=([{"function": {"name": "good_tool"}}], [])
        )

        valid, error = await tm.validate_single_tool("good_tool")

        assert valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_single_tool_invalid_returns_error(self):
        """Line 898: neither valid nor invalid returns 'Tool validation failed'."""
        tm = ToolManager(config_file="test.json", servers=[])
        tool = ToolInfo(
            name="tool_x",
            namespace="ns",
            description="test",
            parameters={},
        )
        tm.get_all_tools = AsyncMock(return_value=[tool])

        # Mock filter_tools to return empty for both valid and invalid
        tm.tool_filter.filter_tools = MagicMock(return_value=([], []))

        valid, error = await tm.validate_single_tool("tool_x")

        assert valid is False
        assert error == "Tool validation failed"

    @pytest.mark.asyncio
    async def test_validate_single_tool_invalid_with_error_msg(self):
        """Line 897: invalid tool returns error from invalid_tools list."""
        tm = ToolManager(config_file="test.json", servers=[])
        tool = ToolInfo(
            name="bad_tool",
            namespace="ns",
            description="test",
            parameters={},
        )
        tm.get_all_tools = AsyncMock(return_value=[tool])

        # Mock filter_tools to return invalid with error
        tm.tool_filter.filter_tools = MagicMock(
            return_value=([], [{"name": "bad_tool", "error": "Missing parameters"}])
        )

        valid, error = await tm.validate_single_tool("bad_tool")

        assert valid is False
        assert error == "Missing parameters"


# ────────────────────────────────────────────────────────────────────
# get_server_info exception - lines 1010-1012
# ────────────────────────────────────────────────────────────────────


class TestGetServerInfoExceptionCoverage:
    """Test get_server_info exception handling."""

    @pytest.mark.asyncio
    async def test_get_server_info_exception_in_server_iteration(self, tmp_path):
        """Lines 1010-1012: exception during server info gathering returns []."""
        config = {
            "mcpServers": {
                "http_server": {"url": "https://example.com"},
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(config_file=str(config_file), servers=["http_server"])
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        # Create a stream_manager mock with tool_to_server_map that raises
        mock_sm = MagicMock()
        # Make tool_to_server_map.values() raise an exception
        bad_map = MagicMock()
        bad_map.values.side_effect = RuntimeError("map error")
        mock_sm.tool_to_server_map = bad_map
        tm.stream_manager = mock_sm

        result = await tm.get_server_info()

        assert result == []


# ────────────────────────────────────────────────────────────────────
# initialize() with empty config (warning path) - line 165-166
# ────────────────────────────────────────────────────────────────────


class TestInitializeEmptyConfig:
    """Test initialize when config loader returns empty dict."""

    @pytest.mark.asyncio
    async def test_initialize_empty_config_calls_setup_empty(self, tmp_path):
        """Lines 164-166: empty config triggers warning and _setup_empty_toolset."""
        # Create an empty config file
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        tm = ToolManager(config_file=str(config_file), servers=[])

        with patch("chuk_term.ui.output"):
            result = await tm.initialize()

        assert result is True
        assert tm.stream_manager is None


# ────────────────────────────────────────────────────────────────────
# Middleware enablement during init
# ────────────────────────────────────────────────────────────────────


class TestMiddlewareEnablement:
    """Test middleware is enabled during _initialize_stream_manager."""

    @pytest.mark.asyncio
    async def test_middleware_enabled_during_init(self, tmp_path):
        """Lines 294-298: middleware is enabled after server init."""
        config = {"mcpServers": {"test": {"url": "https://example.com"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(
            config_file=str(config_file),
            servers=["test"],
            middleware_enabled=True,
        )
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        with patch("mcp_cli.tools.manager.StreamManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.initialize_with_http_streamable = AsyncMock()
            mock_sm.enable_middleware = MagicMock()
            MockSM.return_value = mock_sm

            result = await tm._initialize_stream_manager("stdio")

        assert result is True
        mock_sm.enable_middleware.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_disabled_during_init(self, tmp_path):
        """Middleware is NOT enabled when middleware_enabled=False."""
        config = {"mcpServers": {"test": {"url": "https://example.com"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(
            config_file=str(config_file),
            servers=["test"],
            middleware_enabled=False,
        )
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        with patch("mcp_cli.tools.manager.StreamManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.initialize_with_http_streamable = AsyncMock()
            mock_sm.enable_middleware = MagicMock()
            MockSM.return_value = mock_sm

            result = await tm._initialize_stream_manager("stdio")

        assert result is True
        mock_sm.enable_middleware.assert_not_called()


# ────────────────────────────────────────────────────────────────────
# get_server_info full path with mixed server types
# ────────────────────────────────────────────────────────────────────


class TestGetServerInfoFull:
    """Test get_server_info with all transport types."""

    @pytest.mark.asyncio
    async def test_get_server_info_all_transport_types(self, tmp_path):
        """Cover the full server info generation with HTTP, SSE, and STDIO servers."""
        config = {
            "mcpServers": {
                "http_srv": {"url": "https://http.example.com"},
                "sse_srv": {"url": "https://sse.example.com", "transport": "sse"},
                "stdio_srv": {
                    "command": "python",
                    "args": ["-m", "srv"],
                    "env": {"KEY": "val"},
                },
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        tm = ToolManager(
            config_file=str(config_file),
            servers=["http_srv", "sse_srv", "stdio_srv"],
        )
        tm._config_loader.load()
        tm._config_loader.detect_server_types(tm._config_loader._config_cache)

        mock_sm = MagicMock()
        mock_sm.tool_to_server_map = {
            "tool_a": "http_srv",
            "tool_b": "sse_srv",
            "tool_c": "stdio_srv",
        }
        tm.stream_manager = mock_sm

        result = await tm.get_server_info()

        assert len(result) == 3

        # Check HTTP server
        http_info = next(s for s in result if s.name == "http_srv")
        assert http_info.transport == TransportType.HTTP
        assert http_info.url == "https://http.example.com"
        assert http_info.tool_count == 1

        # Check SSE server
        sse_info = next(s for s in result if s.name == "sse_srv")
        assert sse_info.transport == TransportType.SSE
        assert sse_info.url == "https://sse.example.com"

        # Check STDIO server
        stdio_info = next(s for s in result if s.name == "stdio_srv")
        assert stdio_info.transport == TransportType.STDIO
        assert stdio_info.command == "python"
        assert stdio_info.args == ["-m", "srv"]
        assert stdio_info.env == {"KEY": "val"}
