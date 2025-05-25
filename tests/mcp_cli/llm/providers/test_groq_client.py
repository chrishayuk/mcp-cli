# tests/test_groq_client.py
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from mcp_cli.llm.providers.groq_client import GroqAILLMClient


# ---------------------------------------------------------------------------
# helpers / dummies
# ---------------------------------------------------------------------------


class _DummyResp:
    """Mimic return type of groq-sdk for the non-streaming path."""
    def __init__(self, message: Dict[str, Any]):
        self.choices = [SimpleNamespace(message=message)]


class _DummyDelta:
    def __init__(self, content: str = "", tool_calls: List[Dict[str, Any]] | None = None):
        self.content = content
        # Groq only sets tool_calls on the final chunk
        if tool_calls is not None:
            self.tool_calls = tool_calls


class _DummyChunk:
    def __init__(self, delta: _DummyDelta):
        self.choices = [SimpleNamespace(delta=delta)]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch) -> GroqAILLMClient:
    """Return a GroqAILLMClient with all blocking helpers stubbed out."""
    cl = GroqAILLMClient(model="dummy")

    # --- stub the sanitiser to identity ---
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda t: t)

    return cl


@pytest.fixture(autouse=True)
def groq_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")


# ---------------------------------------------------------------------------
# non-streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_non_streaming(monkeypatch, client):
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "a.b", "parameters": {}}}]

    # capture args passed to _call_blocking
    called: dict[str, Any] = {}

    async def _fake_call_blocking(fn, *args, **kwargs):
        # record for later assertions
        called["args"] = args
        called["kwargs"] = kwargs
        # pretend the SDK callback returned _DummyResp
        return _DummyResp({"foo": "BAR"})

    # stub _normalise_message to a simple passthrough so we can predict output
    monkeypatch.setattr(client, "_call_blocking", _fake_call_blocking)
    monkeypatch.setattr(client, "_normalise_message", lambda m: {"norm": m})

    result = await client.create_completion(messages, tools, stream=False)

    # result comes from _normalise_message
    assert result == {"norm": {"foo": "BAR"}}

    # verify Groq API call was forwarded correctly
    assert called["kwargs"]["model"] == "dummy"
    assert called["kwargs"]["messages"] == messages
    assert called["kwargs"]["tools"] == tools


# ---------------------------------------------------------------------------
# streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_streaming(monkeypatch, client):
    messages = [{"role": "user", "content": "stream please"}]

    # Build three fake chunks: two with content, one final with tool_calls
    chunks = [
        _DummyChunk(_DummyDelta("Hello")),
        _DummyChunk(_DummyDelta(" world")),
        _DummyChunk(_DummyDelta("", tool_calls=[{"name": "foo"}])),
    ]

    class _DummyCompletions:
        def create(self, *, model, messages, tools, stream):
            assert stream is True
            # yield the chunks synchronously (Groq SDK behaviour)
            for c in chunks:
                yield c

    # patch client.client.chat.completions.create
    dummy_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_DummyCompletions())
    )
    monkeypatch.setattr(client, "client", dummy_client)

    # call streaming path
    stream_iter = await client.create_completion(messages, tools=None, stream=True)

    assert hasattr(stream_iter, "__aiter__"), "Should return an async iterator"

    collected = [c async for c in stream_iter]

    # we expect exactly three deltas mirroring our chunks
    assert collected == [
        {"response": "Hello", "tool_calls": []},
        {"response": " world", "tool_calls": []},
        {"response": "", "tool_calls": [{"name": "foo"}]},
    ]

    # sanity-check that the iterator has closed
    assert stream_iter.__aiter__() is stream_iter
    # (no further chunks should arrive)
    with pytest.raises(StopAsyncIteration):
        await stream_iter.__anext__()
