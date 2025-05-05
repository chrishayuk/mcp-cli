"""
End‑to‑end tests for mcp_cli.llm.tools_handler utilities.

Covered:
* handle_tool_call      – StreamManager branch
* convert_to_openai_tools – description propagation
"""

from __future__ import annotations

import json
from typing import Dict, List

import pytest
from unittest.mock import AsyncMock

from mcp_cli.llm.tools_handler import (
    handle_tool_call,
    convert_to_openai_tools,
)


# --------------------------------------------------------------------------- #
# ── Dummy StreamManager stub ──────────────────────────────────────────────── #
# --------------------------------------------------------------------------- #


class DummyStreamManager:
    """
    Minimal stub with just the methods handle_tool_call relies on.
    """

    def __init__(self):
        self.tool_to_server_map = {
            "testTool": "DummySrv",
            "streamingTool": "DummySrv",
        }

    def get_server_for_tool(self, name: str) -> str:
        return self.tool_to_server_map.get(name, "Unknown")

    async def call_tool(self, tool_name: str, arguments: Dict):
        # patched by AsyncMock in individual tests
        return {"isError": False, "content": {"ok": True}}


# --------------------------------------------------------------------------- #
# ── Fixtures ──────────────────────────────────────────────────────────────── #
# --------------------------------------------------------------------------- #

@pytest.fixture
def sm() -> DummyStreamManager:
    return DummyStreamManager()


@pytest.fixture
def convo() -> List[Dict]:
    # minimal conversation context
    return [
        {"role": "system", "content": "system message"},
        {"role": "user", "content": "user prompt"},
    ]


@pytest.fixture
def ok_response():
    return {"isError": False, "content": {"result": "good"}}


@pytest.fixture
def error_response():
    return {"isError": True, "error": "boom", "content": "Error: boom"}


@pytest.fixture
def streaming_response():
    return {
        "isError": False,
        "content": [
            {"type": "text", "text": "chunk 1"},
            {"type": "text", "text": "chunk 2"},
            {"type": "text", "text": "chunk 3"},
        ],
    }


# --------------------------------------------------------------------------- #
# ── handle_tool_call tests (StreamManager branch) ─────────────────────────── #
# --------------------------------------------------------------------------- #

class TestHandleToolCall:
    @pytest.mark.asyncio
    async def test_openai_style_success(self, sm, convo, ok_response):
        sm.call_tool = AsyncMock(return_value=ok_response)

        tool_call = {
            "id": "call123",
            "type": "function",
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "v"}),
            },
        }

        await handle_tool_call(tool_call, convo, stream_manager=sm)

        sm.call_tool.assert_awaited_once_with("testTool", {"param": "v"})
        assert [m["role"] for m in convo][-2:] == ["assistant", "tool"]

        # content is pretty‑printed JSON – parse to compare
        assert json.loads(convo[-1]["content"]) == {"result": "good"}

    @pytest.mark.asyncio
    async def test_object_style_success(self, sm, convo, ok_response):
        sm.call_tool = AsyncMock(return_value=ok_response)

        class Fn:
            name = "testTool"
            arguments = json.dumps({"x": 1})

        class Call:
            id = "c"
            type = "function"
            function = Fn()

        await handle_tool_call(Call(), convo, stream_manager=sm)

        sm.call_tool.assert_awaited_once_with("testTool", {"x": 1})
        assert convo[-1]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_error_response(self, sm, convo, error_response):
        sm.call_tool = AsyncMock(return_value=error_response)

        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"a": 1}),
            }
        }

        await handle_tool_call(tool_call, convo, stream_manager=sm)

        sm.call_tool.assert_awaited_once()
        assert [m["role"] for m in convo][-1] == "tool"
        assert "Error: boom" in convo[-1]["content"]

    @pytest.mark.asyncio
    async def test_streaming_response(self, sm, convo, streaming_response, monkeypatch):
        sm.call_tool = AsyncMock(return_value=streaming_response)

        # Monkey‑patch formatter so we know what to expect
        monkeypatch.setattr(
            "mcp_cli.llm.tools_handler.format_tool_response",
            lambda content: "\n".join(chunk["text"] for chunk in content),
        )

        tool_call = {
            "function": {
                "name": "streamingTool",
                "arguments": "{}",
            }
        }

        await handle_tool_call(tool_call, convo, stream_manager=sm)

        sm.call_tool.assert_awaited_once()
        assert convo[-1]["content"] == "chunk 1\nchunk 2\nchunk 3"

    @pytest.mark.asyncio
    async def test_call_tool_exception(self, sm, convo):
        sm.call_tool = AsyncMock(side_effect=Exception("fail"))

        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": "{}",
            }
        }

        await handle_tool_call(tool_call, convo, stream_manager=sm)

        # Exception path: call attempted, but conversation unchanged
        sm.call_tool.assert_awaited_once()
        assert len(convo) == 2

    @pytest.mark.asyncio
    async def test_bad_json_arguments(self, sm, convo, ok_response):
        sm.call_tool = AsyncMock(return_value=ok_response)

        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": "not-json",
            }
        }

        await handle_tool_call(tool_call, convo, stream_manager=sm)

        sm.call_tool.assert_awaited_once_with("testTool", {})
        assert [m["role"] for m in convo][-2:] == ["assistant", "tool"]


# --------------------------------------------------------------------------- #
# ── convert_to_openai_tools quick check ───────────────────────────────────── #
# --------------------------------------------------------------------------- #

def test_convert_to_openai_tools_description():
    mcp_tools = [
        {
            "name": "demo",
            "description": "test description",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]
    out = convert_to_openai_tools(mcp_tools)
    assert out[0]["function"]["description"] == "test description"
