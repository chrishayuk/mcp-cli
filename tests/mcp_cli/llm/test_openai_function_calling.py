"""
Tests that verify compatibility with OpenAI-style function-calling.

Updated for the 2025-05 implementation of mcp_cli.llm.tools_handler.
"""
from __future__ import annotations

import json
from typing import Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_cli.llm.tools_handler import (
    handle_tool_call,
    convert_to_openai_tools,
)

pytest_plugins = ["pytest_asyncio"]


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _last_call_posargs(mock) -> tuple:
    """Return the positional args of the most recent call to *mock*."""
    return mock.call_args.args if mock.call_args else ()


@pytest.fixture
def mock_stream_manager():
    sm = MagicMock()
    sm.call_tool = AsyncMock()
    sm.get_server_for_tool = MagicMock(return_value="test_server")
    return sm


@pytest.fixture
def convo() -> List[Dict]:
    return [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "What's the weather like in Paris today?"},
    ]


# --------------------------------------------------------------------------- #
# single tool-call                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_openai_style_tool_call(mock_stream_manager, convo):
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": {"temperature": 14, "unit": "celsius"},
    }

    tool_call = {
        "id": "call_12345xyz",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Paris, France\"}",
        },
    }

    await handle_tool_call(tool_call, convo, stream_manager=mock_stream_manager)

    # called exactly once, positional args
    assert _last_call_posargs(mock_stream_manager.call_tool) == (
        "get_weather",
        {"location": "Paris, France"},
    )

    # two new messages added
    assert [m["role"] for m in convo][-2:] == ["assistant", "tool"]

    tool_call_msg, tool_resp_msg = convo[-2], convo[-1]
    assert tool_call_msg["tool_calls"][0]["id"] == "call_12345xyz"
    assert tool_resp_msg["name"] == "get_weather"
    assert tool_resp_msg["tool_call_id"] == "call_12345xyz"


# --------------------------------------------------------------------------- #
# multiple tool-calls                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_multiple_tool_calls(mock_stream_manager, convo):
    mock_stream_manager.call_tool.side_effect = [
        {"isError": False, "content": {"temperature": 14, "unit": "celsius"}},
        {"isError": False, "content": {"temperature": 18, "unit": "celsius"}},
        {"isError": False, "content": "Email sent successfully"},
    ]

    tool_calls = [
        {
            "id": "call_12345xyz",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris, France\"}",
            },
        },
        {
            "id": "call_67890abc",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Bogotá, Colombia\"}",
            },
        },
        {
            "id": "call_99999def",
            "type": "function",
            "function": {
                "name": "send_email",
                "arguments": "{\"to\":\"bob@email.com\",\"body\":\"Hi bob\"}",
            },
        },
    ]

    for tc in tool_calls:
        await handle_tool_call(tc, convo, stream_manager=mock_stream_manager)

    # three invocations
    assert mock_stream_manager.call_tool.call_count == 3
    assert len(convo) == 2 + 3 * 2  # original 2 + 3*(assistant+tool)

    # check each pair
    for i, tc in enumerate(tool_calls):
        call_msg = convo[2 + i * 2]
        resp_msg = convo[3 + i * 2]
        assert call_msg["tool_calls"][0]["id"] == tc["id"]
        assert resp_msg["name"] == tc["function"]["name"]


# --------------------------------------------------------------------------- #
# convert_to_openai_tools                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_convert_to_openai_tools_format():
    mcp_tools = [
        {
            "name": "get_weather",
            "description": "Retrieves current weather.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "units"],
            },
        }
    ]

    out = convert_to_openai_tools(mcp_tools)

    fn = out[0]["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Retrieves current weather."
    assert fn["parameters"]["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]


# --------------------------------------------------------------------------- #
# complex nested arguments                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_handling_complex_arguments(mock_stream_manager, convo):
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": "Action completed successfully",
    }

    big_args = {
        "user": {"id": 1, "prefs": {"theme": "dark"}},
        "items": [{"id": 1}, {"id": 2}],
    }

    tool_call = {
        "id": "call_complex",
        "type": "function",
        "function": {
            "name": "complex_action",
            "arguments": json.dumps(big_args),
        },
    }

    await handle_tool_call(tool_call, convo, stream_manager=mock_stream_manager)

    assert _last_call_posargs(mock_stream_manager.call_tool) == (
        "complex_action",
        big_args,
    )


# --------------------------------------------------------------------------- #
# namespaced tools                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_namespaced_tools_compatibility():
    mcp_tools = [
        {
            "name": "Server1_get_weather",
            "description": "Weather server 1",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "Server2_get_weather",
            "description": "Weather server 2",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    oa_tools = convert_to_openai_tools(mcp_tools)
    assert [t["function"]["name"] for t in oa_tools] == [
        "Server1_get_weather",
        "Server2_get_weather",
    ]

    sm = MagicMock()
    sm.call_tool = AsyncMock(
        return_value={"isError": False, "content": {"temperature": 14}}
    )
    sm.get_server_for_tool = MagicMock(return_value="Server1")

    convo = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "What's the weather?"},
    ]

    tc = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "Server1_get_weather",
            "arguments": "{\"location\":\"Paris\"}",
        },
    }

    await handle_tool_call(tc, convo, stream_manager=sm)

    assert _last_call_posargs(sm.call_tool) == ("Server1_get_weather", {"location": "Paris"})


# --------------------------------------------------------------------------- #
# error handling                                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_openai_error_handling(mock_stream_manager, convo):
    mock_stream_manager.call_tool.return_value = {
        "isError": True,
        "error": "Tool execution failed",
        "content": "Error: Tool execution failed",
    }

    tc = {
        "id": "call_12345xyz",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Invalid\"}",
        },
    }

    await handle_tool_call(tc, convo, stream_manager=mock_stream_manager)

    assert mock_stream_manager.call_tool.await_count == 1
    # assistant + tool error appended
    assert [m["role"] for m in convo][-1] == "tool"
    assert "Error" in convo[-1]["content"]
