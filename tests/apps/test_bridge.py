# tests/apps/test_bridge.py
"""Tests for MCP Apps bridge (Python-side protocol handler)."""

from __future__ import annotations

import asyncio
import json

import pytest

from mcp_cli.apps.bridge import AppBridge
from mcp_cli.apps.models import AppInfo, AppState


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeToolResult:
    def __init__(self, success: bool, result=None, error=None):
        self.success = success
        self.result = result
        self.error = error


class FakeToolManager:
    """Minimal ToolManager stub for bridge tests."""

    def __init__(self):
        self.executed_tools: list[tuple[str, dict, str | None]] = []
        self.read_resources: list[tuple[str, str | None]] = []
        self._next_result = FakeToolResult(True, result="ok")
        self._next_resource = {
            "contents": [{"uri": "ui://test", "text": "<html></html>"}]
        }
        self._raise_on_execute: Exception | None = None
        self._execute_delay: float = 0

    async def execute_tool(self, name, arguments, namespace=None):
        if self._execute_delay:
            await asyncio.sleep(self._execute_delay)
        self.executed_tools.append((name, arguments, namespace))
        if self._raise_on_execute:
            raise self._raise_on_execute
        return self._next_result

    async def read_resource(self, uri, server_name=None):
        self.read_resources.append((uri, server_name))
        return self._next_resource


class FakeWs:
    """Minimal WebSocket stub."""

    def __init__(self):
        self.sent: list[str] = []
        self.closed = False
        self._raise_on_send = False

    async def send(self, msg: str) -> None:
        if self._raise_on_send:
            raise ConnectionError("ws closed")
        self.sent.append(msg)

    async def close(self) -> None:
        self.closed = True


def _make_bridge() -> tuple[AppBridge, FakeToolManager]:
    tm = FakeToolManager()
    info = AppInfo(
        tool_name="test-tool",
        resource_uri="ui://test-tool/app.html",
        server_name="test-server",
        port=9470,
    )
    bridge = AppBridge(info, tm)
    return bridge, tm


# ── Tests ──────────────────────────────────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_tools_call_success(self):
        bridge, tm = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get-time", "arguments": {}},
            }
        )

        resp = await bridge.handle_message(msg)
        assert resp is not None
        parsed = json.loads(resp)
        assert parsed["id"] == 1
        assert "result" in parsed
        assert tm.executed_tools == [("get-time", {}, "test-server")]

    @pytest.mark.asyncio
    async def test_tools_call_failure(self):
        bridge, tm = _make_bridge()
        tm._next_result = FakeToolResult(False, error="timeout")

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "broken-tool", "arguments": {"x": 1}},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 2
        assert "error" in parsed
        assert parsed["error"]["message"] == "timeout"

    @pytest.mark.asyncio
    async def test_resources_read(self):
        bridge, tm = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/read",
                "params": {"uri": "ui://test/data.json"},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 3
        assert "result" in parsed
        assert tm.read_resources == [("ui://test/data.json", "test-server")]

    @pytest.mark.asyncio
    async def test_ui_message(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "ui/message",
                "params": {"content": {"type": "text", "text": "hello"}},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 4
        assert parsed["result"] == {}

    @pytest.mark.asyncio
    async def test_model_context_update(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "ui/update-model-context",
                "params": {"content": [{"type": "text", "text": "user picked red"}]},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 5
        assert parsed["result"] == {}
        assert bridge.model_context is not None

    @pytest.mark.asyncio
    async def test_initialized_notification(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ui/notifications/initialized",
                "params": {},
            }
        )

        resp = await bridge.handle_message(msg)
        assert resp is None
        assert bridge.app_info.state == AppState.READY

    @pytest.mark.asyncio
    async def test_unknown_request(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "unknown/method",
                "params": {},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_unknown_notification_ignored(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "unknown/notification",
                "params": {},
            }
        )

        resp = await bridge.handle_message(msg)
        assert resp is None

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        bridge, _ = _make_bridge()
        resp = await bridge.handle_message("not json at all")
        assert resp is None

    @pytest.mark.asyncio
    async def test_tool_call_with_exception(self):
        """execute_tool raises an exception."""
        bridge, tm = _make_bridge()
        tm._raise_on_execute = RuntimeError("server crashed")
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "crash-tool", "arguments": {}},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32000
        assert "server crashed" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_tool_name_rejected(self):
        """Tool names with special chars are rejected."""
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"name": "rm -rf /", "arguments": {}},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_empty_tool_name_rejected(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {"name": "", "arguments": {}},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_resource_read_exception(self):
        """read_resource raises an exception."""
        bridge, tm = _make_bridge()
        tm._next_resource = None  # will cause AttributeError

        # Override to raise
        async def _raise_resource(uri, server_name=None):
            raise RuntimeError("network error")

        tm.read_resource = _raise_resource

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "resources/read",
                "params": {"uri": "ui://bad/resource"},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32000
        assert "network error" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_teardown_notification_sets_closed(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ui/notifications/teardown",
                "params": {},
            }
        )
        resp = await bridge.handle_message(msg)
        assert resp is None
        assert bridge.app_info.state == AppState.CLOSED


class TestWebSocketLifecycle:
    def test_set_ws_resets_state(self):
        bridge, _ = _make_bridge()
        bridge.app_info.state = AppState.READY
        ws = FakeWs()
        bridge.set_ws(ws)
        assert bridge.app_info.state == AppState.INITIALIZING

    def test_set_ws_closes_old(self):
        bridge, _ = _make_bridge()
        old_ws = FakeWs()
        bridge.set_ws(old_ws)
        new_ws = FakeWs()
        bridge.set_ws(new_ws)
        # Old WS close is scheduled via ensure_future, verify the new one is set
        assert bridge._ws is new_ws

    @pytest.mark.asyncio
    async def test_push_tool_result_queued_when_no_ws(self):
        bridge, _ = _make_bridge()
        # No WS set
        await bridge.push_tool_result("test result")
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_push_tool_result_sent_when_ws(self):
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)
        await bridge.push_tool_result("test result")
        assert len(ws.sent) == 1
        parsed = json.loads(ws.sent[0])
        assert parsed["method"] == "ui/notifications/tool-result"

    @pytest.mark.asyncio
    async def test_push_tool_result_queued_on_send_failure(self):
        bridge, _ = _make_bridge()
        ws = FakeWs()
        ws._raise_on_send = True
        bridge.set_ws(ws)
        await bridge.push_tool_result("test result")
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_drain_pending_sends_queued(self):
        bridge, _ = _make_bridge()
        # Queue a notification while WS is down
        await bridge.push_tool_result("queued result")
        assert len(bridge._pending_notifications) == 1

        # Now connect WS and drain
        ws = FakeWs()
        bridge.set_ws(ws)
        await bridge.drain_pending()
        assert len(ws.sent) == 1
        assert len(bridge._pending_notifications) == 0

    @pytest.mark.asyncio
    async def test_drain_pending_requeues_on_failure(self):
        bridge, _ = _make_bridge()
        await bridge.push_tool_result("queued result")

        ws = FakeWs()
        ws._raise_on_send = True
        bridge.set_ws(ws)
        await bridge.drain_pending()
        # Should still be in queue since send failed
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_initial_tool_result_deferred_until_initialized(self):
        """Initial tool result should not be sent immediately but after app initialized."""
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)

        # Store initial tool result
        bridge.set_initial_tool_result("initial data")
        assert bridge._initial_tool_result == "initial data"
        # Nothing sent yet
        assert len(ws.sent) == 0

        # Simulate app sending ui/notifications/initialized
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "ui/notifications/initialized", "params": {}}
        )
        await bridge.handle_message(msg)

        # Give the ensure_future a chance to run
        await asyncio.sleep(0)

        # Now the tool result should have been pushed
        assert len(ws.sent) == 1
        parsed = json.loads(ws.sent[0])
        assert parsed["method"] == "ui/notifications/tool-result"
        # And the stored result should be cleared
        assert bridge._initial_tool_result is None

    @pytest.mark.asyncio
    async def test_initial_tool_result_not_sent_twice(self):
        """Ensure deferred tool result is only sent once even on re-init."""
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)
        bridge.set_initial_tool_result("initial data")

        # First initialized
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "ui/notifications/initialized", "params": {}}
        )
        await bridge.handle_message(msg)
        await asyncio.sleep(0)
        assert len(ws.sent) == 1

        # Second initialized (e.g., after reconnect)
        await bridge.handle_message(msg)
        await asyncio.sleep(0)
        # Should still be just 1 message
        assert len(ws.sent) == 1


class TestExtractStructuredContent:
    """Test the spec-compliant structuredContent extraction from text blocks."""

    def test_hoists_structured_content_from_text_block(self):
        """Per MCP spec: servers include structuredContent as JSON in text blocks."""
        inner = json.dumps(
            {
                "content": [{"type": "text", "text": "Temperature: 72°F"}],
                "structuredContent": {"temperature": 72, "conditions": "sunny"},
            }
        )
        out = {
            "content": [{"type": "text", "text": inner}],
        }
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == {"temperature": 72, "conditions": "sunny"}
        assert result["content"] == [{"type": "text", "text": "Temperature: 72°F"}]

    def test_skips_if_already_present(self):
        out = {
            "content": [{"type": "text", "text": "{}"}],
            "structuredContent": {"existing": True},
        }
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == {"existing": True}

    def test_skips_non_json_text(self):
        out = {"content": [{"type": "text", "text": "plain text"}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_skips_multiple_content_blocks(self):
        out = {
            "content": [
                {"type": "text", "text": "{}"},
                {"type": "text", "text": "{}"},
            ]
        }
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_skips_json_without_structured_content(self):
        out = {"content": [{"type": "text", "text": json.dumps({"key": "value"})}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_keeps_original_content_when_inner_content_missing(self):
        """When JSON has structuredContent but no content array."""
        inner = json.dumps({"structuredContent": {"type": "markdown", "data": {}}})
        out = {"content": [{"type": "text", "text": inner}]}
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == {"type": "markdown", "data": {}}
        # Original text block preserved since no inner content array
        assert result["content"] == [{"type": "text", "text": inner}]


class TestFormatToolResult:
    def test_string_result(self):
        result = AppBridge._format_tool_result("hello world")
        assert result == {"content": [{"type": "text", "text": "hello world"}]}

    def test_dict_with_content(self):
        original = {"content": [{"type": "text", "text": "foo"}]}
        result = AppBridge._format_tool_result(original)
        assert result == original

    def test_raw_dict(self):
        result = AppBridge._format_tool_result({"key": "value"})
        assert result["content"][0]["type"] == "text"
        assert '"key"' in result["content"][0]["text"]

    def test_numeric_result(self):
        result = AppBridge._format_tool_result(42)
        assert result == {"content": [{"type": "text", "text": "42"}]}

    def test_format_extracts_structured_content_from_dict(self):
        """End-to-end: dict result with embedded JSON text containing structuredContent."""
        inner_json = json.dumps(
            {
                "content": [{"type": "text", "text": "Chart data"}],
                "structuredContent": {"type": "chart", "data": {"values": [1, 2, 3]}},
            }
        )
        result = AppBridge._format_tool_result(
            {
                "content": [{"type": "text", "text": inner_json}],
            }
        )
        assert result["structuredContent"] == {
            "type": "chart",
            "data": {"values": [1, 2, 3]},
        }
        assert result["content"] == [{"type": "text", "text": "Chart data"}]

    def test_format_extracts_structured_content_from_pydantic(self):
        """End-to-end: Pydantic-like model with embedded JSON text."""

        class FakeContent:
            def __init__(self, type, text):
                self.type = type
                self.text = text

            def model_dump(self):
                return {"type": self.type, "text": self.text}

        class FakePydanticResult:
            def __init__(self):
                inner_json = json.dumps(
                    {
                        "content": [{"type": "text", "text": "Map"}],
                        "structuredContent": {"type": "geojson", "data": {}},
                    }
                )
                self.content = [FakeContent("text", inner_json)]
                self.isError = False

        result = AppBridge._format_tool_result(FakePydanticResult())
        assert result["structuredContent"] == {"type": "geojson", "data": {}}
        assert result["content"] == [{"type": "text", "text": "Map"}]


class TestHelpers:
    def test_to_serializable_circular_reference(self):
        """Circular references should not cause infinite recursion."""
        d: dict = {"key": "value"}
        d["self"] = d  # circular!
        result = AppBridge._to_serializable(d)
        assert result["key"] == "value"
        assert result["self"] == "<circular>"

    def test_safe_json_dumps_normal(self):
        result = AppBridge._safe_json_dumps({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_safe_json_dumps_fallback(self):
        """Non-serializable objects should fall back to _to_serializable."""

        class Custom:
            def __init__(self):
                self.x = 42

        result = AppBridge._safe_json_dumps({"obj": Custom()})
        parsed = json.loads(result)
        assert parsed["obj"]["x"] == 42
