"""
Tests for mcp_cli.context.context_manager
==========================================

Covers ApplicationContext, ContextManager singleton, and the convenience
functions get_context() / initialize_context().

Target: >90 % line coverage on src/mcp_cli/context/context_manager.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_cli.context.context_manager import (
    ApplicationContext,
    ContextManager,
    get_context,
    initialize_context,
)
from mcp_cli.tools.models import (
    ConversationMessage,
    ServerInfo,
    ToolInfo,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #


def _make_server(name: str = "test-server", **overrides) -> ServerInfo:
    """Create a minimal ServerInfo for testing."""
    defaults = dict(
        id=0,
        name=name,
        status="connected",
        tool_count=0,
        namespace="stdio",
    )
    defaults.update(overrides)
    return ServerInfo(**defaults)


def _make_tool(name: str = "tool1", namespace: str = "ns", **overrides) -> ToolInfo:
    """Create a minimal ToolInfo for testing."""
    defaults = dict(
        name=name,
        namespace=namespace,
        description="A test tool",
    )
    defaults.update(overrides)
    return ToolInfo(**defaults)


def _make_mock_tool_manager(servers=None, tools=None) -> MagicMock:
    """Return a mock ToolManager with async helpers wired up."""
    tm = MagicMock()
    tm.get_server_info = AsyncMock(return_value=servers or [])
    tm.get_all_tools = AsyncMock(return_value=tools or [])
    return tm


# --------------------------------------------------------------------------- #
# ApplicationContext - construction
# --------------------------------------------------------------------------- #


class TestApplicationContextConstruction:
    """Tests for basic creation and model_post_init."""

    def test_default_construction(self):
        ctx = ApplicationContext()
        assert ctx.provider == "openai"
        assert ctx.model == "gpt-4o-mini"
        assert ctx.model_manager is not None  # created automatically
        assert ctx.conversation_history == []
        assert ctx.servers == []
        assert ctx.tools == []

    def test_create_factory(self):
        ctx = ApplicationContext.create(provider="anthropic", model="claude-3")
        assert ctx.provider == "anthropic"
        assert ctx.model == "claude-3"
        assert ctx.config_path == Path("server_config.json")

    def test_create_with_tool_manager(self):
        tm = _make_mock_tool_manager()
        ctx = ApplicationContext.create(tool_manager=tm)
        assert ctx.tool_manager is tm

    def test_create_with_config_path(self):
        ctx = ApplicationContext.create(config_path=Path("/tmp/custom.json"))
        assert ctx.config_path == Path("/tmp/custom.json")


# --------------------------------------------------------------------------- #
# ApplicationContext.initialize  (async)
# --------------------------------------------------------------------------- #


class TestApplicationContextInitialize:
    @pytest.mark.asyncio
    async def test_initialize_loads_servers_and_tools(self):
        server = _make_server()
        tool = _make_tool()
        tm = _make_mock_tool_manager(servers=[server], tools=[tool])
        ctx = ApplicationContext.create(tool_manager=tm)

        await ctx.initialize()

        assert ctx.servers == [server]
        assert ctx.tools == [tool]
        # Single server -> set as current_server automatically (line 113)
        assert ctx.current_server is server

    @pytest.mark.asyncio
    async def test_initialize_no_auto_current_when_multiple_servers(self):
        s1 = _make_server("s1")
        s2 = _make_server("s2", id=1)
        tm = _make_mock_tool_manager(servers=[s1, s2])
        ctx = ApplicationContext.create(tool_manager=tm)

        await ctx.initialize()

        assert ctx.current_server is None  # not auto-set

    @pytest.mark.asyncio
    async def test_initialize_no_tool_manager(self):
        ctx = ApplicationContext.create()
        await ctx.initialize()
        assert ctx.servers == []
        assert ctx.tools == []


# --------------------------------------------------------------------------- #
# get_current_server / set_current_server (lines 117, 121)
# --------------------------------------------------------------------------- #


class TestCurrentServer:
    def test_get_current_server_none_by_default(self):
        ctx = ApplicationContext.create()
        assert ctx.get_current_server() is None

    def test_set_and_get_current_server(self):
        ctx = ApplicationContext.create()
        server = _make_server("my-server")
        ctx.set_current_server(server)
        assert ctx.get_current_server() is server
        assert ctx.get_current_server().name == "my-server"


# --------------------------------------------------------------------------- #
# find_server / find_tool (lines 125-128, 132-135)
# --------------------------------------------------------------------------- #


class TestFindServerAndTool:
    def test_find_server_by_name(self):
        s1 = _make_server("Alpha")
        s2 = _make_server("Beta", id=1)
        ctx = ApplicationContext.create()
        ctx.servers = [s1, s2]

        assert ctx.find_server("alpha") is s1  # case-insensitive
        assert ctx.find_server("BETA") is s2
        assert ctx.find_server("gamma") is None

    def test_find_tool_by_name(self):
        t1 = _make_tool("read_file", "fs")
        t2 = _make_tool("write_file", "fs")
        ctx = ApplicationContext.create()
        ctx.tools = [t1, t2]

        assert ctx.find_tool("read_file") is t1
        assert ctx.find_tool("write_file") is t2
        assert ctx.find_tool("delete_file") is None

    def test_find_tool_by_fully_qualified_name(self):
        t1 = _make_tool("read_file", "fs")
        ctx = ApplicationContext.create()
        ctx.tools = [t1]

        # fully_qualified_name is "fs.read_file"
        assert ctx.find_tool("fs.read_file") is t1

    def test_find_server_empty_list(self):
        ctx = ApplicationContext.create()
        assert ctx.find_server("anything") is None

    def test_find_tool_empty_list(self):
        ctx = ApplicationContext.create()
        assert ctx.find_tool("anything") is None


# --------------------------------------------------------------------------- #
# get / set  (lines 144-148, 154-157)
# --------------------------------------------------------------------------- #


class TestGetSet:
    def test_get_known_attribute(self):
        ctx = ApplicationContext.create(provider="anthropic")
        assert ctx.get("provider") == "anthropic"

    def test_get_unknown_key_returns_default(self):
        ctx = ApplicationContext.create()
        assert ctx.get("nonexistent") is None
        assert ctx.get("nonexistent", 42) == 42

    def test_set_known_attribute(self):
        ctx = ApplicationContext.create()
        ctx.set("provider", "anthropic")
        assert ctx.provider == "anthropic"

    def test_set_unknown_key_stored_in_extra(self):
        ctx = ApplicationContext.create()
        ctx.set("custom_key", "custom_value")
        assert ctx.get("custom_key") == "custom_value"

    def test_get_from_extra(self):
        ctx = ApplicationContext.create()
        ctx.set("my_extra", 123)
        assert ctx.get("my_extra") == 123


# --------------------------------------------------------------------------- #
# to_dict  (line 165)
# --------------------------------------------------------------------------- #


class TestToDict:
    def test_to_dict_basic(self):
        ctx = ApplicationContext.create(provider="openai", model="gpt-4o-mini")
        d = ctx.to_dict()

        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o-mini"
        assert d["config_path"] == "server_config.json"
        assert d["servers"] == []
        assert d["tools"] == []
        assert d["conversation_history"] == []
        assert d["is_interactive"] is False

    def test_to_dict_includes_extra(self):
        ctx = ApplicationContext.create()
        ctx.set("bonus", "data")
        d = ctx.to_dict()
        assert d["bonus"] == "data"


# --------------------------------------------------------------------------- #
# update_from_dict  (lines 193-197)
# --------------------------------------------------------------------------- #


class TestUpdateFromDict:
    def test_update_from_dict_known_keys(self):
        ctx = ApplicationContext.create()
        ctx.update_from_dict({"provider": "anthropic", "model": "claude-3"})
        assert ctx.provider == "anthropic"
        assert ctx.model == "claude-3"

    def test_update_from_dict_unknown_keys(self):
        ctx = ApplicationContext.create()
        ctx.update_from_dict({"custom_field": 99})
        assert ctx.get("custom_field") == 99

    def test_update_from_dict_mixed(self):
        ctx = ApplicationContext.create()
        ctx.update_from_dict(
            {
                "provider": "groq",
                "some_extra": "value",
            }
        )
        assert ctx.provider == "groq"
        assert ctx.get("some_extra") == "value"


# --------------------------------------------------------------------------- #
# update(**kwargs)  (lines 205-209)
# --------------------------------------------------------------------------- #


class TestUpdate:
    def test_update_known_attributes(self):
        ctx = ApplicationContext.create()
        ctx.update(provider="deepseek", verbose_mode=False)
        assert ctx.provider == "deepseek"
        assert ctx.verbose_mode is False

    def test_update_unknown_attributes(self):
        ctx = ApplicationContext.create()
        ctx.update(new_key="new_val")
        assert ctx.get("new_key") == "new_val"

    def test_update_mixed(self):
        ctx = ApplicationContext.create()
        ctx.update(model="big-model", custom_flag=True)
        assert ctx.model == "big-model"
        assert ctx.get("custom_flag") is True


# --------------------------------------------------------------------------- #
# Conversation message helpers (lines 214-245)
# --------------------------------------------------------------------------- #


class TestConversationMessages:
    def test_add_message_dict(self):
        ctx = ApplicationContext.create()
        ctx.add_message({"role": "user", "content": "hello"})
        assert len(ctx.conversation_history) == 1
        assert ctx.conversation_history[0]["role"] == "user"

    def test_add_message_conversation_message(self):
        ctx = ApplicationContext.create()
        msg = ConversationMessage.user_message("hi")
        ctx.add_message(msg)
        assert len(ctx.conversation_history) == 1
        assert ctx.conversation_history[0]["role"] == "user"
        assert ctx.conversation_history[0]["content"] == "hi"

    def test_add_user_message(self):
        ctx = ApplicationContext.create()
        ctx.add_user_message("What is 2+2?")
        assert len(ctx.conversation_history) == 1
        assert ctx.conversation_history[0]["role"] == "user"
        assert ctx.conversation_history[0]["content"] == "What is 2+2?"

    def test_add_assistant_message_text_only(self):
        ctx = ApplicationContext.create()
        ctx.add_assistant_message(content="It is 4.")
        assert ctx.conversation_history[0]["role"] == "assistant"
        assert ctx.conversation_history[0]["content"] == "It is 4."

    def test_add_assistant_message_with_tool_calls(self):
        ctx = ApplicationContext.create()
        tool_calls = [
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        ctx.add_assistant_message(content=None, tool_calls=tool_calls)
        msg = ctx.conversation_history[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg

    def test_add_system_message(self):
        ctx = ApplicationContext.create()
        ctx.add_system_message("You are helpful.")
        assert ctx.conversation_history[0]["role"] == "system"
        assert ctx.conversation_history[0]["content"] == "You are helpful."

    def test_add_tool_message(self):
        ctx = ApplicationContext.create()
        ctx.add_tool_message(content="result", tool_call_id="tc1", name="my_tool")
        msg = ctx.conversation_history[0]
        assert msg["role"] == "tool"
        assert msg["content"] == "result"
        assert msg["tool_call_id"] == "tc1"
        assert msg["name"] == "my_tool"

    def test_add_tool_message_without_name(self):
        ctx = ApplicationContext.create()
        ctx.add_tool_message(content="result", tool_call_id="tc2")
        msg = ctx.conversation_history[0]
        assert msg["role"] == "tool"
        assert "name" not in msg  # excluded because None

    def test_get_messages_returns_typed(self):
        ctx = ApplicationContext.create()
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi")
        msgs = ctx.get_messages()
        assert len(msgs) == 2
        assert all(isinstance(m, ConversationMessage) for m in msgs)
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_clear_conversation(self):
        ctx = ApplicationContext.create()
        ctx.add_user_message("a")
        ctx.add_user_message("b")
        assert len(ctx.conversation_history) == 2
        ctx.clear_conversation()
        assert ctx.conversation_history == []

    def test_multiple_messages_in_sequence(self):
        ctx = ApplicationContext.create()
        ctx.add_system_message("sys")
        ctx.add_user_message("usr")
        ctx.add_assistant_message("asst")
        ctx.add_tool_message("res", "tc1")
        assert len(ctx.conversation_history) == 4
        roles = [m["role"] for m in ctx.conversation_history]
        assert roles == ["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# ContextManager singleton
# --------------------------------------------------------------------------- #


class TestContextManager:
    def test_singleton(self):
        cm1 = ContextManager()
        cm2 = ContextManager()
        assert cm1 is cm2

    def test_get_context_before_initialize_raises(self):
        with pytest.raises(RuntimeError, match="Context not initialized"):
            ContextManager().get_context()

    def test_initialize_and_get_context(self):
        cm = ContextManager()
        ctx = cm.initialize(provider="openai", model="gpt-4o-mini")
        assert isinstance(ctx, ApplicationContext)
        assert cm.get_context() is ctx

    def test_initialize_idempotent(self):
        cm = ContextManager()
        ctx1 = cm.initialize(provider="openai")
        ctx2 = cm.initialize(provider="anthropic")
        assert ctx1 is ctx2  # second call returns same context

    def test_reset_clears_context(self):
        cm = ContextManager()
        cm.initialize()
        cm.reset()
        with pytest.raises(RuntimeError):
            cm.get_context()

    def test_initialize_with_tool_manager(self):
        tm = _make_mock_tool_manager()
        cm = ContextManager()
        ctx = cm.initialize(tool_manager=tm)
        assert ctx.tool_manager is tm


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #


class TestConvenienceFunctions:
    def test_get_context_raises_when_uninitialized(self):
        with pytest.raises(RuntimeError):
            get_context()

    def test_initialize_context_and_get_context(self):
        ctx = initialize_context(provider="openai")
        assert isinstance(ctx, ApplicationContext)
        retrieved = get_context()
        assert retrieved is ctx

    def test_initialize_context_with_kwargs(self):
        ctx = initialize_context(
            provider="anthropic",
            model="claude-3",
            verbose_mode=False,
        )
        assert ctx.provider == "anthropic"
        assert ctx.model == "claude-3"
        assert ctx.verbose_mode is False
