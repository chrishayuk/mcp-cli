# tests/test_base.py
import pytest
import inspect
from typing import Dict, Any, List, Optional

from mcp_cli.llm.providers.base import BaseLLMClient


def test_cannot_instantiate_abstract():
    """BaseLLMClient is abstract – direct instantiation must fail."""
    with pytest.raises(TypeError):
        BaseLLMClient()  # type: ignore[abstract-instantiated]


# ---------------------------------------------------------------------------
# Minimal concrete subclass for behavioural checks
# ---------------------------------------------------------------------------

class DummyLLM(BaseLLMClient):
    """A minimal concrete implementation used only for unit tests."""

    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Echo back last user content for predictable testing
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        return {
            "response": last_user.get("content", ""),
            "tool_calls": tools or [],
        }


@pytest.mark.asyncio
async def test_dummy_llm_behaviour():
    client = DummyLLM()
    inp_msgs = [
        {"role": "user", "content": "Ping"},
        {"role": "assistant", "content": "Pong"},
    ]
    tools = [{"name": "noop"}]

    result = await client.create_completion(inp_msgs, tools=tools)
    assert result == {"response": "Ping", "tool_calls": tools}


def test_dummy_llm_is_subclass():
    """Ensure DummyLLM is recognised as a concrete subclass (no abstract methods)."""
    assert not inspect.isabstract(DummyLLM)
