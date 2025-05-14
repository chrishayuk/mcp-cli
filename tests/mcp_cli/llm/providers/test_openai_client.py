# tests/test_openai_client.py
import sys
import types
import pytest

# -----------------------------------------------------------------------------
# Provide a stub "openai" module *before* importing the client implementation so
# the real SDK is never required (and no network calls are made).
# -----------------------------------------------------------------------------

dummy_openai = types.ModuleType("openai")


class _DummyCompletions:
    # Placeholder attribute – tests will monkey‑patch the actual callable
    create = lambda *a, **k: None  # noqa: E731


class _DummyChat:
    completions = _DummyCompletions()


class DummyOpenAI:
    """Mimics ``openai.OpenAI`` enough for the client to instantiate."""

    def __init__(self, *args, **kwargs):  # accept arbitrary kwargs
        self.chat = _DummyChat()


dummy_openai.OpenAI = DummyOpenAI
sys.modules["openai"] = dummy_openai

# Now the import is safe – it will pick up our stub instead of the real SDK.
from mcp_cli.llm.providers.openai_client import OpenAILLMClient  # noqa: E402  pylint: disable=wrong-import-position


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def client():
    """Return a *fresh* client instance for each test."""
    return OpenAILLMClient(model="test-model", api_key="sk-test")


# -----------------------------------------------------------------------------
# Tests – non‑streaming completion
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_one_shot(monkeypatch, client):
    messages = [{"role": "user", "content": "Hi!"}]
    tools_in = [{"name": "demo"}]

    # Make tool sanitisation a no‑op so we can assert exactly what is forwarded.
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    # Stub ``_normalise_message`` so we control the final output easily.
    monkeypatch.setattr(client, "_normalise_message", lambda msg: {"normalised": msg})

    # Patch ``_call_blocking`` so no real threads / network are used and so we
    # can assert the parameters with which it was invoked.
    async def fake_call_blocking(func, *args, **kwargs):
        # The passed callable must be the SDK's ``create``.
        assert callable(func)
        # Key arguments propagated from the client
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == messages
        assert kwargs["tools"] == tools_in

        # Fabricate a minimal SDK‑like response object.
        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Resp:
            def __init__(self):
                self.choices = [_Choice({"role": "assistant", "content": "Hello"})]

        return _Resp()

    monkeypatch.setattr(client, "_call_blocking", fake_call_blocking)

    # Ensure the placeholder SDK callable exists to satisfy attribute access.
    client.client.chat.completions.create = lambda *a, **k: None  # noqa: E731

    result = await client.create_completion(messages, tools=tools_in, stream=False)
    assert result == {"normalised": {"role": "assistant", "content": "Hello"}}


# -----------------------------------------------------------------------------
# Tests – streaming completion
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Hello again"}]

    # Sanitise tools is a no‑op for simplicity.
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    # Build an async generator to return from the fake streaming helper.
    async def _gen():
        yield {"delta": 1}
        yield {"delta": 2}

    # ``_stream_from_blocking`` is synchronous in the real impl – it *returns*
    # an async iterator. We therefore stub it with a sync function that does
    # exactly that.
    def fake_stream_from_blocking(func, *args, **kwargs):  # noqa: D401 – simple function
        # Confirm forwarding of critical args
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == messages
        # No tools provided ⇒ the client should pass an empty list
        assert kwargs["tools"] == []
        return _gen()

    monkeypatch.setattr(client, "_stream_from_blocking", fake_stream_from_blocking)

    # Dummy SDK callable so attribute lookup succeeds
    client.client.chat.completions.create = lambda *a, **k: None  # noqa: E731

    async_iter = await client.create_completion(messages, tools=None, stream=True)
    assert hasattr(async_iter, "__aiter__")  # basic sanity – is async iterator

    collected = [item async for item in async_iter]
    assert collected == [{"delta": 1}, {"delta": 2}]
