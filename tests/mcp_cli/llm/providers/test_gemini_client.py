# tests/test_gemini_client.py
import sys
import types
import asyncio
import pytest

# ---------------------------------------------------------------------------
# Build a stub for the ``google.genai`` SDK *before* importing the client, so
# that the real heavy package is never needed and no network calls are made.
# ---------------------------------------------------------------------------

google_mod = sys.modules.get("google") or types.ModuleType("google")
if "google" not in sys.modules:
    sys.modules["google"] = google_mod

# --- sub-module ``google.genai`` -------------------------------------------

genai_mod = types.ModuleType("google.genai")
sys.modules["google.genai"] = genai_mod
setattr(google_mod, "genai", genai_mod)

# --- sub-module ``google.genai.types`` -------------------------------------

types_mod = types.ModuleType("google.genai.types")
sys.modules["google.genai.types"] = types_mod
setattr(genai_mod, "types", types_mod)

# Provide *minimal* class stubs used by the adapter's helper code. We keep
# them extremely simple – they only need to accept the constructor args.

class _Simple:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"<_Simple {self.__dict__}>"


class Part(_Simple):
    @staticmethod
    def from_text(text: str):
        return Part(text=text)

    @staticmethod
    def from_function_response(name: str, response):
        return Part(func_resp={"name": name, "response": response})

    @staticmethod
    def from_function_call(name: str, args):
        return Part(func_call={"name": name, "args": args})


types_mod.Part = Part

types_mod.Content = _Simple

types_mod.Schema = _Simple

types_mod.FunctionDeclaration = _Simple

types_mod.Tool = _Simple


class _FCCMode:
    AUTO = "AUTO"

types_mod.FunctionCallingConfigMode = _FCCMode

types_mod.FunctionCallingConfig = _Simple

types_mod.ToolConfig = _Simple

types_mod.GenerateContentConfig = _Simple


# ---------------------------------------------------------------------------
# Fake client that the adapter will instantiate
# ---------------------------------------------------------------------------

class _DummyModels:
    def generate_content(self, *a, **k):
        # Never used in our patched tests – but must exist
        return None

    def generate_content_stream(self, *a, **k):
        # Never used because we patch _stream – but must exist
        return []


class DummyGenAIClient:
    def __init__(self, *args, **kwargs):
        self.models = _DummyModels()


genai_mod.Client = DummyGenAIClient

# ---------------------------------------------------------------------------
# Now import the adapter under test (it will pick up the stubs).
# ---------------------------------------------------------------------------

from mcp_cli.llm.providers.gemini_client import GeminiLLMClient  # noqa: E402  pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Fixture producing a fresh client for each test.
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return GeminiLLMClient(model="gemini-test", api_key="fake-key")


# ---------------------------------------------------------------------------
# Non-streaming path – create_completion(stream=False)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_non_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Hello Gemini"}]
    tools = [{"type": "function", "function": {"name": "demo.fn", "parameters": {}}}]

    # Patch asyncio.to_thread so it simply calls the fn inline (no threads).
    async def fake_to_thread(func, *args, **kwargs):  # noqa: D401
        return func(*args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    # Patch the _create_sync method to validate the forwarded args and return a
    # predictable result.
    def fake_create_sync(self, msgs, tls):  # noqa: D401 – simple func
        assert msgs == messages
        assert tls == tools
        return {"response": "Hi!", "tool_calls": []}

    monkeypatch.setattr(GeminiLLMClient, "_create_sync", fake_create_sync, raising=True)

    out = await client.create_completion(messages, tools=tools, stream=False)
    assert out == {"response": "Hi!", "tool_calls": []}


# ---------------------------------------------------------------------------
# Streaming path – create_completion(stream=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Stream it"}]

    # Fake _stream returns an async-generator so we can iterate
    async def _gen():
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": [{"name": "fn"}]}

    async def fake_stream(self, msgs, tls):  # noqa: D401
        assert msgs == messages
        # No tools provided → adapter should forward None
        assert tls is None
        async for chunk in _gen():
            yield chunk

    monkeypatch.setattr(GeminiLLMClient, "_stream", fake_stream, raising=True)

    # Call streaming variant
    ai = await client.create_completion(messages, tools=None, stream=True)
    assert hasattr(ai, "__aiter__")

    pieces = [c async for c in ai]
    assert pieces == [
        {"response": "chunk1", "tool_calls": []},
        {"response": "chunk2", "tool_calls": [{"name": "fn"}]},
    ]
