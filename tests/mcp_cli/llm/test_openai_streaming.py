"""
Tests for OpenAI-style *streaming* tool-call chunks.

These tests exercise the pattern shown in the OpenAI docs: the model sends
incremental deltas which must be merged into a final tool-call object.
"""
from __future__ import annotations

import json

import pytest

pytest_plugins = ["pytest_asyncio"]


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_streaming_chunks():
    """Stream for a single `get_weather` call accumulated over many chunks."""
    return [
        # initial declaration
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_12345xyz",
                    "function": {"arguments": "", "name": "get_weather"},
                    "type": "function",
                }
            ]
        },
        # argument pieces
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "{\"", "name": None}, "type": None}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "location", "name": None}, "type": None}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\":\"", "name": None}, "type": None}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "Paris", "name": None}, "type": None}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": ",", "name": None}, "type": None}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": " France", "name": None}, "type": None}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\"}", "name": None}, "type": None}]},
        # end-of-stream
        {"tool_calls": None},
    ]


# --------------------------------------------------------------------------- #
# Single-call accumulator                                                     #
# --------------------------------------------------------------------------- #

def test_accumulate_streaming_tool_calls(mock_streaming_chunks):
    final: dict[int, dict] = {}

    for chunk in mock_streaming_chunks:
        for tc in chunk.get("tool_calls") or []:
            idx = tc["index"]
            if idx not in final:
                final[idx] = {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
            else:
                # append argument delta
                final[idx]["function"]["arguments"] += tc["function"]["arguments"]

    assert 0 in final
    c0 = final[0]
    assert c0["id"] == "call_12345xyz"
    assert c0["function"]["name"] == "get_weather"
    assert json.loads(c0["function"]["arguments"]) == {"location": "Paris, France"}


# --------------------------------------------------------------------------- #
# Multiple concurrent calls                                                   #
# --------------------------------------------------------------------------- #

def test_multiple_streaming_tool_calls():
    chunks = [
        # first call
        {"tool_calls": [{"index": 0, "id": "call_weather", "function": {"arguments": "", "name": "get_weather"}, "type": "function"}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "{\"location\":\"Paris\"}", "name": None}, "type": None}]},

        # second call
        {"tool_calls": [{"index": 1, "id": "call_email", "function": {"arguments": "", "name": "send_email"}, "type": "function"}]},
        {"tool_calls": [{"index": 1, "id": None, "function": {"arguments": "{\"to\":\"bob@example.com\"}", "name": None}, "type": None}]},
        {"tool_calls": None},
    ]

    final: dict[int, dict] = {}

    for chunk in chunks:
        for tc in chunk.get("tool_calls") or []:
            idx = tc["index"]
            if idx not in final:
                final[idx] = {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
            else:
                if tc["function"] and "arguments" in tc["function"]:
                    final[idx]["function"]["arguments"] += tc["function"]["arguments"]

    # weather call
    assert json.loads(final[0]["function"]["arguments"]) == {"location": "Paris"}
    # email call
    assert json.loads(final[1]["function"]["arguments"]) == {"to": "bob@example.com"}


# --------------------------------------------------------------------------- #
# Async helper for user apps                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_streaming_helper():
    async def process_stream(chunks):
        acc: dict[int, dict] = {}
        for chunk in chunks:
            for tc in chunk.get("tool_calls") or []:
                idx = tc["index"]
                if idx not in acc:
                    acc[idx] = {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                else:
                    # merge deltas
                    if tc["id"] is not None:
                        acc[idx]["id"] = tc["id"]
                    if tc["type"] is not None:
                        acc[idx]["type"] = tc["type"]
                    if tc["function"]["name"] is not None:
                        acc[idx]["function"]["name"] = tc["function"]["name"]
                    if "arguments" in tc["function"]:
                        acc[idx]["function"]["arguments"] += tc["function"]["arguments"]

        # parse JSON for convenience
        for data in acc.values():
            try:
                data["function"]["parsed_arguments"] = json.loads(data["function"]["arguments"])
            except json.JSONDecodeError:
                data["function"]["parsed_arguments"] = None
        return acc

    chunks = [
        {"tool_calls": [{"index": 0, "id": "call_test", "function": {"arguments": "{\"key", "name": "test_tool"}, "type": "function"}]},
        {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\":\"value\"}", "name": None}, "type": None}]},
        {"tool_calls": None},
    ]

    result = await process_stream(chunks)

    assert json.loads(result[0]["function"]["arguments"]) == {"key": "value"}
    assert result[0]["function"]["parsed_arguments"] == {"key": "value"}
