# tests/test_ollama_client.py
import sys
import types
import pytest
import asyncio

# ---------------------------------------------------------------------------
# Stub the ``ollama`` package *before* importing the adapter so that the real
# binary‑binding library isn't required and no network calls are attempted.
# ---------------------------------------------------------------------------

ollama_stub = types.ModuleType("ollama")

# Provide a dummy ``chat`` callable so the constructor's hasattr check passes.

def _dummy_chat(*args, **kwargs):  # noqa: D401 – simple function
    # This will never be called because `_create_sync` is monkey‑patched in the
    # tests, but keep it here for safety if other code paths hit it.
    class _Msg:
        content = "stub"
        tool_calls = []
    class _Resp:
        message = _Msg()
    return _Resp()

ollama_stub.chat = _dummy_chat

# Some versions of the python SDK expose ``set_host``; include it so the API
# base logic doesn't warn.
ollama_stub.set_host = lambda host: None  # noqa: E731

sys.modules["ollama"] = ollama_stub

# ---------------------------------------------------------------------------
# Now import the Ollama client (will see the stub).
# ---------------------------------------------------------------------------

from mcp_cli.llm.providers.ollama_client import OllamaLLMClient  # noqa: E402  pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    # Provide api_base so the set_host branch is exercised
    return OllamaLLMClient(model="ollama-test", api_base="http://fake.local")


# ---------------------------------------------------------------------------
# Tests – synchronous usage (i.e. user does *not* await the result)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_sync(monkeypatch, client):
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"name": "t"}]

    # Track how often _create_sync is invoked
    calls = {"n": 0}

    def fake_create_sync(self, msgs, tls):  # noqa: D401 – simple func
        calls["n"] += 1
        # Ensure arguments propagate unchanged
        assert msgs == messages
        assert tls == tools
        return {"response": "hello", "tool_calls": []}

    monkeypatch.setattr(OllamaLLMClient, "_create_sync", fake_create_sync, raising=True)

    result = client.create_completion(messages, tools=tools)

    # Immediately we should have exactly one call (the initial synchronous one)
    assert calls["n"] == 1

    # Result should behave like a dict and include helper methods (__await__)
    assert isinstance(result, dict)
    assert result["response"] == "hello"
    assert hasattr(result, "__await__")


# ---------------------------------------------------------------------------
# Tests – asynchronous usage (awaiting the returned AwaitableDict)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_async(monkeypatch, client):
    messages = [{"role": "user", "content": "async plz"}]

    # Counter to ensure _create_sync called twice (sync + async)!
    calls = {"n": 0}

    def fake_create_sync(self, msgs, tls):  # noqa: D401
        calls["n"] += 1
        return {"response": f"call{calls['n']}", "tool_calls": []}

    monkeypatch.setattr(OllamaLLMClient, "_create_sync", fake_create_sync, raising=True)

    # Patch asyncio.to_thread so it executes inline (no thread pool needed)
    async def fake_to_thread(fn):  # noqa: D401
        return fn()
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    result = client.create_completion(messages, tools=None)
    # first synchronous invocation done
    assert calls["n"] == 1 and result["response"] == "call1"

    # Await the same object – should trigger second _create_sync call
    awaited = await result
    assert calls["n"] == 2
    assert awaited["response"] == "call2"

    # The original result dict should remain unchanged
    assert result["response"] == "call1"
