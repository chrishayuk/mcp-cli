"""
Integration tests for tools-handler components that use a mocked
StreamManager.
"""
from __future__ import annotations

import json
from typing import Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.llm.tools_handler import (
    convert_to_openai_tools,
    handle_tool_call,
    format_tool_response,
)

pytest_plugins = ["pytest_asyncio"]


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def sm() -> MagicMock:
    """Mocked StreamManager providing only the methods used by handle_tool_call."""
    m = MagicMock(spec=["call_tool", "get_server_for_tool", "get_internal_tools"])
    m.call_tool = AsyncMock()
    m.get_server_for_tool.return_value = "test_server"
    m.get_internal_tools.return_value = []
    return m


@pytest.fixture
def convo() -> List[Dict]:
    return [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]


def _last_posargs(mock) -> tuple:
    """Return the positional args of the last await of a mock."""
    return mock.call_args.args if mock.call_args else ()


# --------------------------------------------------------------------------- #
# complex response + formatting                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
@patch("mcp_cli.llm.tools_handler.format_tool_response")
async def test_complex_response_integration(mock_fmt, sm, convo):
    complex_resp = {
        "isError": False,
        "content": [
            {
                "type": "result",
                "rows": [
                    {"id": 1, "name": "Item 1", "value": 100},
                    {"id": 2, "name": "Item 2", "value": 200},
                ],
            },
            {"type": "error", "message": "partial", "code": "WARN"},
        ],
    }
    sm.call_tool.return_value = complex_resp
    mock_fmt.side_effect = format_tool_response

    tc = {
        "id": "call_123",
        "function": {"name": "complexTool", "arguments": json.dumps({"param": "v"})},
    }

    await handle_tool_call(tc, convo, stream_manager=sm)

    assert _last_posargs(sm.call_tool) == ("complexTool", {"param": "v"})
    mock_fmt.assert_called_once_with(complex_resp["content"])

    assert [m["role"] for m in convo][-2:] == ["assistant", "tool"]
    assert convo[-1]["name"] == "complexTool"


# --------------------------------------------------------------------------- #
# full workflow with convert_to_openai_tools                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_full_workflow(sm, convo):
    sm.call_tool.return_value = {"isError": False, "content": {"result": "Success"}}
    sm.get_internal_tools.return_value = [
        {
            "name": "Server1_testTool",
            "description": "Test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param": {"type": "string"}},
            },
        }
    ]

    oa_tools = convert_to_openai_tools(sm.get_internal_tools())
    assert oa_tools[0]["function"]["name"] == "Server1_testTool"
    assert oa_tools[0]["function"]["description"] == "Test tool"

    tc = {
        "id": "call_abc",
        "function": {
            "name": "Server1_testTool",
            "arguments": json.dumps({"param": "x"}),
        },
    }

    await handle_tool_call(tc, convo, stream_manager=sm)
    assert _last_posargs(sm.call_tool) == ("Server1_testTool", {"param": "x"})
    assert convo[-1]["name"] == "Server1_testTool"


# --------------------------------------------------------------------------- #
# multiple sequential calls                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_multiple_tool_calls(sm, convo):
    sm.call_tool.side_effect = [
        {"isError": False, "content": {"result": "first"}},
        {"isError": False, "content": {"result": "second"}},
    ]

    tc1 = {
        "id": "call_1",
        "function": {"name": "Server1_tool1", "arguments": json.dumps({"p": 1})},
    }
    tc2 = {
        "id": "call_2",
        "function": {"name": "Server2_tool2", "arguments": json.dumps({"p": 2})},
    }

    await handle_tool_call(tc1, convo, stream_manager=sm)
    await handle_tool_call(tc2, convo, stream_manager=sm)

    assert sm.call_tool.call_count == 2
    assert [m["role"] for m in convo][-4:] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    # assistant stub for the SECOND call is convo[-2]
    assert convo[-2]["tool_calls"][0]["function"]["name"] == "Server2_tool2"
    assert convo[-1]["name"] == "Server2_tool2"


# --------------------------------------------------------------------------- #
# error flow                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tool_error_flow(sm, convo):
    sm.call_tool.return_value = {
        "isError": True,
        "error": "DB fail",
        "content": "Error: DB fail",
    }

    tc = {
        "id": "call_err",
        "function": {
            "name": "Server1_dbQuery",
            "arguments": json.dumps({"query": "SELECT"}),
        },
    }

    await handle_tool_call(tc, convo, stream_manager=sm)

    assert sm.call_tool.await_count == 1
    assert convo[-1]["role"] == "tool"
    assert "Error" in convo[-1]["content"]
    assert "DB fail" in convo[-1]["content"]
