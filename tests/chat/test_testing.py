# tests/chat/test_testing.py
"""Tests for mcp_cli.chat.testing.TestChatContext."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name="my_tool", description="desc", parameters=None):
    """Create a fake ToolInfo-like object."""
    return SimpleNamespace(
        name=name, description=description, parameters=parameters or {}
    )


def _make_stream_manager(
    tools=None, server_info=None, has_call_tool=True, has_internal_tools=True
):
    """Build a mock stream_manager with configurable behaviour."""
    mgr = MagicMock()
    tools = tools if tools is not None else [_make_tool()]
    server_info = server_info if server_info is not None else [{"name": "server1"}]

    if has_internal_tools:
        mgr.get_internal_tools.return_value = tools
    else:
        del mgr.get_internal_tools  # remove the attr so hasattr returns False
        mgr.get_all_tools.return_value = tools

    mgr.get_server_info.return_value = server_info
    mgr.get_server_for_tool.return_value = "test-server"

    if has_call_tool:
        mgr.call_tool = AsyncMock(return_value={"result": "ok"})
    else:
        del mgr.call_tool  # remove so hasattr returns False

    return mgr


def _make_model_manager():
    """Build a mock ModelManager."""
    mm = MagicMock()
    mm.get_active_provider.return_value = "openai"
    mm.get_active_model.return_value = "gpt-4"
    return mm


@pytest.fixture(autouse=True)
def _allow_conversation_history_assignment():
    """
    ChatContext.conversation_history is a read-only @property.
    TestChatContext.__init__ tries to assign self.conversation_history = [].
    We temporarily replace the property with a simple read/write descriptor
    so the assignment in __init__ succeeds.
    """
    from mcp_cli.chat.chat_context import ChatContext

    original = ChatContext.__dict__.get("conversation_history")
    # Remove the property so instances can have their own attribute
    if isinstance(original, property):
        delattr(ChatContext, "conversation_history")
    yield
    # Restore
    if original is not None:
        ChatContext.conversation_history = original


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestTestChatContextInit:
    """Test the __init__ path."""

    def test_basic_init(self):
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager()
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        assert ctx.stream_manager is sm
        assert ctx.model_manager is mm
        assert ctx.tool_manager is None
        assert ctx.exit_requested is False
        assert ctx.tools == []
        assert ctx.internal_tools == []
        assert ctx.server_info == []
        assert ctx.tool_to_server_map == {}
        assert ctx.openai_tools == []
        assert ctx.tool_name_mapping == {}
        assert ctx.tool_processor is None
        assert isinstance(ctx.conversation_history, list)
        assert isinstance(ctx.tool_history, list)

    def test_init_uses_model_manager_properties(self):
        """The debug log line accesses self.provider / self.model."""
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager()
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)
        # provider / model come from ChatContext @property -> model_manager
        assert ctx.provider == "openai"
        assert ctx.model == "gpt-4"


# ---------------------------------------------------------------------------
# create_for_testing tests
# ---------------------------------------------------------------------------


class TestCreateForTesting:
    """Test the classmethod factory."""

    @patch("mcp_cli.chat.testing.ModelManager")
    def test_no_provider_no_model(self, MockMM):
        from mcp_cli.chat.testing import TestChatContext

        mm_inst = _make_model_manager()
        MockMM.return_value = mm_inst
        sm = _make_stream_manager()

        ctx = TestChatContext.create_for_testing(sm)
        assert ctx.stream_manager is sm
        mm_inst.switch_model.assert_not_called()
        mm_inst.switch_provider.assert_not_called()

    @patch("mcp_cli.chat.testing.ModelManager")
    def test_provider_and_model(self, MockMM):
        from mcp_cli.chat.testing import TestChatContext

        mm_inst = _make_model_manager()
        MockMM.return_value = mm_inst
        sm = _make_stream_manager()

        TestChatContext.create_for_testing(sm, provider="anthropic", model="claude-3")
        mm_inst.switch_model.assert_called_once_with("anthropic", "claude-3")

    @patch("mcp_cli.chat.testing.ModelManager")
    def test_provider_only(self, MockMM):
        from mcp_cli.chat.testing import TestChatContext

        mm_inst = _make_model_manager()
        MockMM.return_value = mm_inst
        sm = _make_stream_manager()

        TestChatContext.create_for_testing(sm, provider="groq")
        mm_inst.switch_provider.assert_called_once_with("groq")
        mm_inst.switch_model.assert_not_called()

    @patch("mcp_cli.chat.testing.ModelManager")
    def test_model_only(self, MockMM):
        from mcp_cli.chat.testing import TestChatContext

        mm_inst = _make_model_manager()
        mm_inst.get_active_provider.return_value = "openai"
        MockMM.return_value = mm_inst
        sm = _make_stream_manager()

        TestChatContext.create_for_testing(sm, model="gpt-3.5")
        mm_inst.switch_model.assert_called_once_with("openai", "gpt-3.5")


# ---------------------------------------------------------------------------
# _initialize_tools tests
# ---------------------------------------------------------------------------


class TestInitializeTools:
    """Test the async _initialize_tools method."""

    @pytest.mark.asyncio
    async def test_with_internal_tools(self):
        from mcp_cli.chat.testing import TestChatContext

        tools = [_make_tool("t1", "Tool 1"), _make_tool("t2", "Tool 2")]
        sm = _make_stream_manager(tools=tools, has_internal_tools=True)
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        await ctx._initialize_tools()

        assert len(ctx.tools) == 2
        assert len(ctx.openai_tools) == 2
        assert ctx.openai_tools[0]["type"] == "function"
        assert ctx.openai_tools[0]["function"]["name"] == "t1"
        assert ctx.tool_to_server_map == {"t1": "test-server", "t2": "test-server"}
        assert ctx.internal_tools == list(ctx.tools)
        assert ctx.tool_name_mapping == {}
        sm.get_server_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_get_all_tools(self):
        from mcp_cli.chat.testing import TestChatContext

        tools = [_make_tool("fallback_tool")]
        sm = _make_stream_manager(tools=tools, has_internal_tools=False)
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        await ctx._initialize_tools()

        assert len(ctx.tools) == 1
        assert ctx.tools[0].name == "fallback_tool"
        sm.get_all_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_tools(self):
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager(tools=[], server_info=[])
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        await ctx._initialize_tools()

        assert ctx.tools == []
        assert ctx.openai_tools == []
        assert ctx.tool_to_server_map == {}

    @pytest.mark.asyncio
    async def test_tool_parameters_in_openai_format(self):
        from mcp_cli.chat.testing import TestChatContext

        params = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tools = [_make_tool("calc", "Calculator", params)]
        sm = _make_stream_manager(tools=tools)
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        await ctx._initialize_tools()

        func_def = ctx.openai_tools[0]["function"]
        assert func_def["parameters"] == params
        assert func_def["description"] == "Calculator"


# ---------------------------------------------------------------------------
# execute_tool tests
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Test the async execute_tool method."""

    @pytest.mark.asyncio
    async def test_calls_stream_manager_call_tool(self):
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager()
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        result = await ctx.execute_tool("my_tool", {"arg": "val"})
        assert result == {"result": "ok"}
        sm.call_tool.assert_awaited_once_with("my_tool", {"arg": "val"})

    @pytest.mark.asyncio
    async def test_raises_when_no_call_tool(self):
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager(has_call_tool=False)
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        with pytest.raises(ValueError, match="doesn't support tool execution"):
            await ctx.execute_tool("any_tool", {})


# ---------------------------------------------------------------------------
# get_server_for_tool tests
# ---------------------------------------------------------------------------


class TestGetServerForTool:
    """Test the async get_server_for_tool method."""

    @pytest.mark.asyncio
    async def test_returns_server_name(self):
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager()
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        server = await ctx.get_server_for_tool("my_tool")
        assert server == "test-server"
        sm.get_server_for_tool.assert_called_once_with("my_tool")

    @pytest.mark.asyncio
    async def test_returns_unknown_when_none(self):
        from mcp_cli.chat.testing import TestChatContext

        sm = _make_stream_manager()
        sm.get_server_for_tool.return_value = None
        mm = _make_model_manager()
        ctx = TestChatContext(sm, mm)

        server = await ctx.get_server_for_tool("missing_tool")
        assert server == "Unknown"
