"""
Unit-tests for:

* mcp_cli.llm.tools_handler.handle_tool_call
* mcp_cli.llm.tools_handler.convert_to_openai_tools
"""
import json
from typing import List, Dict

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_cli.llm.tools_handler import handle_tool_call, convert_to_openai_tools
from mcp_cli.tools.models import ToolCallResult
from mcp_cli.tools.manager import ToolManager


# --------------------------------------------------------------------------- #
# Helper fixtures                                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tool_call_dict() -> Dict:
    """OpenAI-style tool-call represented as a *dict*."""
    return {
        "id": "call_test123",
        "function": {
            "name": "test_tool",
            "arguments": json.dumps({"param": "value"}),
        },
    }


@pytest.fixture()
def tool_call_obj():
    """OpenAI-style tool-call represented as an *object* with attributes."""
    class _Fn:
        name = "test_tool"
        arguments = json.dumps({"param": "value"})

    class _Call:
        id = "call_test123"
        function = _Fn()

    return _Call()


# --------------------------------------------------------------------------- #
# handle_tool_call – StreamManager branch                                     #
# --------------------------------------------------------------------------- #

class TestHandleToolCall_StreamManager:
    """Exercising the legacy StreamManager path."""

    @pytest.mark.asyncio
    async def test_success(self, tool_call_dict):
        convo: List[Dict] = []

        sm = MagicMock()
        sm.call_tool = AsyncMock(
            return_value={"isError": False, "content": "Tool execution successful"}
        )
        sm.get_server_for_tool.return_value = "fake_srv"

        await handle_tool_call(
            tool_call=tool_call_dict,
            conversation_history=convo,
            stream_manager=sm,
        )

        sm.call_tool.assert_awaited_once_with("test_tool", {"param": "value"})
        assert [m["role"] for m in convo] == ["assistant", "tool"]
        assert convo[1]["content"] == "Tool execution successful"

    @pytest.mark.asyncio
    async def test_error(self, tool_call_dict):
        convo: List[Dict] = []

        sm = MagicMock()
        sm.call_tool = AsyncMock(
            return_value={"isError": True, "error": "Boom!", "content": ""}
        )
        sm.get_server_for_tool.return_value = "fake_srv"

        await handle_tool_call(
            tool_call=tool_call_dict,
            conversation_history=convo,
            stream_manager=sm,
        )

        assert [m["role"] for m in convo] == ["assistant", "tool"]
        assert "Error: Boom!" in convo[1]["content"]


# --------------------------------------------------------------------------- #
# handle_tool_call – ToolManager branch                                       #
# --------------------------------------------------------------------------- #

class DummyToolManager(ToolManager):
    """Lightweight stub that still passes `isinstance(..., ToolManager)`."""

    def __init__(self):  # Bypass heavy real initialiser
        pass


class TestHandleToolCall_ToolManager:
    """Exercising the preferred ToolManager path."""

    @pytest.mark.asyncio
    async def test_success(self, tool_call_dict):
        convo: List[Dict] = []

        tm = DummyToolManager()
        tm.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool",
                success=True,
                result="Tool execution successful",
                error=None,
            )
        )
        tm.get_server_for_tool = MagicMock(return_value="fake_srv")

        await handle_tool_call(
            tool_call=tool_call_dict,
            conversation_history=convo,
            tool_manager=tm,
        )

        tm.execute_tool.assert_awaited_once_with("test_tool", {"param": "value"})
        assert [m["role"] for m in convo] == ["assistant", "tool"]
        assert convo[1]["content"] == "Tool execution successful"

    @pytest.mark.asyncio
    async def test_failure(self, tool_call_obj):
        convo: List[Dict] = []

        tm = DummyToolManager()
        tm.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool",
                success=False,
                result=None,
                error="Exploded",
            )
        )
        tm.get_server_for_tool = MagicMock(return_value="fake_srv")

        await handle_tool_call(
            tool_call=tool_call_obj,
            conversation_history=convo,
            tool_manager=tm,
        )

        assert [m["role"] for m in convo] == ["assistant", "tool"]
        assert "Error: Exploded" in convo[1]["content"]


# --------------------------------------------------------------------------- #
# convert_to_openai_tools                                                     #
# --------------------------------------------------------------------------- #

class TestConvertToOpenAITools:
    """Behavioural tests for convert_to_openai_tools."""

    def test_basic_conversion_with_descriptions(self):
        mcp_tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"p": {"type": "string"}},
                },
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"q": {"type": "number"}},
                },
            },
        ]

        openai_tools = convert_to_openai_tools(mcp_tools)

        assert [t["function"]["name"] for t in openai_tools] == ["tool1", "tool2"]
        assert openai_tools[0]["function"]["parameters"] == mcp_tools[0]["inputSchema"]
        assert openai_tools[0]["function"]["description"] == "First tool"

    def test_missing_schema(self):
        mcp = [{"name": "noschema", "description": "no schema"}]

        res = convert_to_openai_tools(mcp)
        assert res[0]["function"]["parameters"] == {}
        assert res[0]["function"]["description"] == "no schema"

    def test_no_description_field(self):
        mcp = [{"name": "nodesc"}]
        res = convert_to_openai_tools(mcp)
        assert res[0]["function"]["description"] == ""

    def test_complex_schema_preserved(self):
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "arr": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["s"],
        }
        mcp = [
            {"name": "complex", "description": "c", "inputSchema": schema},
        ]
        res = convert_to_openai_tools(mcp)
        assert res[0]["function"]["parameters"] == schema

    def test_namespaced_tools_preserved(self):
        mcp = [
            {"name": "Srv1_toolA", "description": "A", "inputSchema": {}},
            {"name": "Srv2_toolB", "description": "B", "inputSchema": {}},
        ]
        res = convert_to_openai_tools(mcp)
        assert [t["function"]["name"] for t in res] == ["Srv1_toolA", "Srv2_toolB"]

    def test_idempotent_when_already_openai_format(self):
        openai_fmt = [
            {
                "type": "function",
                "function": {
                    "name": "already",
                    "description": "done",
                    "parameters": {},
                },
            }
        ]
        assert convert_to_openai_tools(openai_fmt) == openai_fmt
