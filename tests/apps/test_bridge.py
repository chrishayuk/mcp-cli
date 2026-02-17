# tests/apps/test_bridge.py
"""Tests for MCP Apps bridge (Python-side protocol handler)."""

from __future__ import annotations

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
        self._next_resource = {"contents": [{"uri": "ui://test", "text": "<html></html>"}]}

    async def execute_tool(self, name, arguments, namespace=None):
        self.executed_tools.append((name, arguments, namespace))
        return self._next_result

    async def read_resource(self, uri, server_name=None):
        self.read_resources.append((uri, server_name))
        return self._next_resource


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
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get-time", "arguments": {}},
        })

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

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "broken-tool", "arguments": {"x": 1}},
        })

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 2
        assert "error" in parsed
        assert parsed["error"]["message"] == "timeout"

    @pytest.mark.asyncio
    async def test_resources_read(self):
        bridge, tm = _make_bridge()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "ui://test/data.json"},
        })

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 3
        assert "result" in parsed
        assert tm.read_resources == [("ui://test/data.json", "test-server")]

    @pytest.mark.asyncio
    async def test_ui_message(self):
        bridge, _ = _make_bridge()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "ui/message",
            "params": {"content": {"type": "text", "text": "hello"}},
        })

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 4
        assert parsed["result"] == {}

    @pytest.mark.asyncio
    async def test_model_context_update(self):
        bridge, _ = _make_bridge()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "ui/update-model-context",
            "params": {"content": [{"type": "text", "text": "user picked red"}]},
        })

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 5
        assert parsed["result"] == {}
        assert bridge.model_context is not None

    @pytest.mark.asyncio
    async def test_initialized_notification(self):
        bridge, _ = _make_bridge()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "ui/notifications/initialized",
            "params": {},
        })

        resp = await bridge.handle_message(msg)
        assert resp is None
        assert bridge.app_info.state == AppState.READY

    @pytest.mark.asyncio
    async def test_unknown_request(self):
        bridge, _ = _make_bridge()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "unknown/method",
            "params": {},
        })

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_unknown_notification_ignored(self):
        bridge, _ = _make_bridge()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "unknown/notification",
            "params": {},
        })

        resp = await bridge.handle_message(msg)
        assert resp is None

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        bridge, _ = _make_bridge()
        resp = await bridge.handle_message("not json at all")
        assert resp is None


class TestExtractStructuredContent:
    """Test the spec-compliant structuredContent extraction from text blocks."""

    def test_hoists_structured_content_from_text_block(self):
        """Per MCP spec: servers include structuredContent as JSON in text blocks."""
        inner = json.dumps({
            "content": [{"type": "text", "text": "Temperature: 72°F"}],
            "structuredContent": {"temperature": 72, "conditions": "sunny"},
        })
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
        out = {
            "content": [{"type": "text", "text": json.dumps({"key": "value"})}]
        }
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
        inner_json = json.dumps({
            "content": [{"type": "text", "text": "Chart data"}],
            "structuredContent": {"type": "chart", "data": {"values": [1, 2, 3]}},
        })
        result = AppBridge._format_tool_result({
            "content": [{"type": "text", "text": inner_json}],
        })
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

        class FakeToolResult:
            def __init__(self):
                inner_json = json.dumps({
                    "content": [{"type": "text", "text": "Map"}],
                    "structuredContent": {"type": "geojson", "data": {}},
                })
                self.content = [FakeContent("text", inner_json)]
                self.isError = False

        result = AppBridge._format_tool_result(FakeToolResult())
        assert result["structuredContent"] == {"type": "geojson", "data": {}}
        assert result["content"] == [{"type": "text", "text": "Map"}]
