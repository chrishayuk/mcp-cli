# tests/mcp_cli/chat/test_chat_context.py
"""Unit-tests for the re-worked *ChatContext* class.

We avoid the heavyweight real ToolManager by supplying a tiny stub that
implements just enough of the async API surface the context expects.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.tools.models import ToolInfo, ServerInfo

# ---------------------------------------------------------------------------
# Dummy async ToolManager stub
# ---------------------------------------------------------------------------
class DummyToolManager:  # noqa: WPS110 - test helper
    """Minimal stand-in that satisfies the methods ChatContext uses."""

    def __init__(self) -> None:
        self._tools = [
            ToolInfo(
                name="tool1",
                namespace="srv1",
                description="demo-1",
                parameters={},
                is_async=False,
            ),
            ToolInfo(
                name="tool2",
                namespace="srv2",
                description="demo-2",
                parameters={},
                is_async=False,
            ),
        ]

        self._servers = [
            ServerInfo(
                id=0,
                name="srv1",
                status="ok",
                tool_count=1,
                namespace="srv1",
            ),
            ServerInfo(
                id=1,
                name="srv2",
                status="ok",
                tool_count=1,
                namespace="srv2",
            ),
        ]

        self._openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"{t.namespace}_{t.name}",
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools
        ]

    # ----- discovery --------------------------------------------------
    async def get_unique_tools(self):  # noqa: D401 - match signature
        return self._tools

    async def get_server_info(self):  # noqa: D401 - match signature
        return self._servers

    async def get_adapted_tools_for_llm(self, provider: str = "openai"):
        mapping = {
            f"{t.namespace}_{t.name}": f"{t.namespace}.{t.name}"
            for t in self._tools
        }
        return self._openai_tools, mapping

    async def get_tools_for_llm(self):
        return self._openai_tools

    async def get_server_for_tool(self, tool_name: str):
        if "." in tool_name:
            return tool_name.split(".", 1)[0]
        if "_" in tool_name:
            return tool_name.split("_", 1)[0]
        return "Unknown"

    # ----- execution stubs -------------------------------------------
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        return {"success": True, "result": {"echo": arguments}}

    async def stream_execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        yield {"success": True, "result": {"echo": arguments}}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def dummy_tool_manager():
    return DummyToolManager()


@pytest.fixture()
def chat_context(dummy_tool_manager, monkeypatch):
    # Use deterministic system prompt
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
    )
    ctx = ChatContext.create(tool_manager=dummy_tool_manager)
    return ctx

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_initialize_chat_context(chat_context):
    ok = await chat_context.initialize()
    assert ok is True

    # tools discovered
    assert chat_context.get_tool_count() == 2

    # system prompt injected as first conversation turn
    assert chat_context.conversation_history[0] == {
        "role": "system",
        "content": "SYS_PROMPT",
    }

    # OpenAI tools adapted
    assert len(chat_context.openai_tools) == 2
    assert chat_context.tool_name_mapping  # non-empty


@pytest.mark.asyncio
async def test_get_server_for_tool(chat_context):
    await chat_context.initialize()

    assert await chat_context.get_server_for_tool("srv1.tool1") == "srv1"
    assert await chat_context.get_server_for_tool("srv2_tool2") == "srv2"
    assert await chat_context.get_server_for_tool("unknown") == "Unknown"


@pytest.mark.asyncio
async def test_to_dict_and_update_roundtrip(chat_context):
    await chat_context.initialize()
    original_len = chat_context.get_conversation_length()

    exported = chat_context.to_dict()

    # mutate exported copy
    exported["exit_requested"] = True
    exported["conversation_history"].append({"role": "user", "content": "Hi"})

    chat_context.update_from_dict(exported)

    assert chat_context.exit_requested is True
    assert chat_context.get_conversation_length() == original_len + 1
    assert chat_context.conversation_history[-1]["content"] == "Hi"
