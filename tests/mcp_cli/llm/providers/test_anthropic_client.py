# tests/test_anthropic_client.py
import sys
import types
import json
import pytest

# ---------------------------------------------------------------------------
# Stub the `anthropic` SDK before importing the adapter.
# ---------------------------------------------------------------------------

anthropic_mod = types.ModuleType("anthropic")
sys.modules["anthropic"] = anthropic_mod

# Create submodule anthropic.types so that "from anthropic.types import X" works
anthropic_types_mod = types.ModuleType("anthropic.types")
sys.modules["anthropic.types"] = anthropic_types_mod
anthropic_mod.types = anthropic_types_mod

# Minimal ToolUseBlock type stub
class ToolUseBlock(dict):
    pass

# Expose ToolUseBlock under both anthropic and anthropic.types namespaces
anthropic_mod.ToolUseBlock = ToolUseBlock
anthropic_types_mod.ToolUseBlock = ToolUseBlock

# Fake Messages client
class _DummyMessages:
    def create(self, *args, **kwargs):
        return None  # will be monkey-patched per-test

# Fake Anthropic client
class DummyAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _DummyMessages()

anthropic_mod.Anthropic = DummyAnthropic

# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from mcp_cli.llm.providers.anthropic_client import AnthropicLLMClient  # noqa: E402  pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return AnthropicLLMClient(model="claude-test", api_key="fake-key")

# Convenience helper to capture kwargs
class Capture:
    kwargs = None

# ---------------------------------------------------------------------------
# Non‑streaming test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_stream(monkeypatch, client):
    # Simple chat sequence
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi Claude"},
    ]
    tools = [
        {"type": "function", "function": {"name": "foo", "parameters": {}}}
    ]

    # Sanitise no‑op so we can assert
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    # Patch _call_blocking to validate payload and return dummy response
    async def fake_call_blocking(func, **payload):  # noqa: D401
        Capture.kwargs = payload
        # Simulate Claude text response
        resp = types.SimpleNamespace(
            content=[types.SimpleNamespace(text="Hello there!")]
        )
        return resp

    monkeypatch.setattr(client, "_call_blocking", fake_call_blocking)

    result = await client.create_completion(messages, tools=tools, stream=False)
    assert result == {"response": "Hello there!", "tool_calls": []}

    # Validate key bits of the payload sent to Claude
    assert Capture.kwargs["model"] == "claude-test"
    assert Capture.kwargs["system"] == "You are helpful."
    assert Capture.kwargs["stream"] is False
    # tools converted gets placed into payload["tools"] – check basic structure
    conv_tools = Capture.kwargs["tools"]
    assert isinstance(conv_tools, list) and conv_tools[0]["name"] == "foo"

# ---------------------------------------------------------------------------
# Streaming test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Stream please"}]

    # Patch stream helper to return async generator
    async def _gen():
        yield {"delta": 1}
        yield {"delta": 2}

    def fake_stream_from_blocking(func, **payload):  # noqa: D401
        # Ensure model & stream flag are correct
        assert payload["model"] == "claude-test"
        assert payload["stream"] is True
        return _gen()

    monkeypatch.setattr(client, "_stream_from_blocking", fake_stream_from_blocking)
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    iterator = await client.create_completion(messages, tools=None, stream=True)
    assert hasattr(iterator, "__aiter__")
    received = [d async for d in iterator]
    assert received == [{"delta": 1}, {"delta": 2}]
