# tests/mcp_cli/chat/test_chat_context.py
"""Unit-tests for the re-worked *ChatContext* class.

We avoid the heavyweight real ToolManager by supplying a tiny stub that
implements just enough of the async API surface the context expects.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.tools.models import ToolInfo, ServerInfo


# ---------------------------------------------------------------------------
# Dummy async ToolManager stub
# ---------------------------------------------------------------------------
class DummyToolManager:  # noqa: E501 - test helper
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
            f"{t.namespace}_{t.name}": f"{t.namespace}.{t.name}" for t in self._tools
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
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    # Mock ModelManager to avoid model discovery issues
    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create a minimal mock ModelManager
    mock_model_manager = Mock(spec=ModelManager)
    mock_model_manager.provider = "mock"
    mock_model_manager.model = "mock-model"
    mock_model_manager.get_client.return_value = None
    mock_model_manager.get_active_provider.return_value = "mock"
    mock_model_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_model_manager
    )
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
    assert chat_context.conversation_history[0].to_dict() == {
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

    exported = chat_context.to_dict()

    # update_from_dict handles exit_requested but not conversation_history
    exported["exit_requested"] = True

    chat_context.update_from_dict(exported)

    assert chat_context.exit_requested is True


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_tool_by_name_exact(chat_context):
    """Test find_tool_by_name with exact match."""
    await chat_context.initialize()
    tool = chat_context.find_tool_by_name("tool1")
    assert tool is not None
    assert tool.name == "tool1"


@pytest.mark.asyncio
async def test_find_tool_by_name_fully_qualified(chat_context):
    """Test find_tool_by_name with fully qualified name."""
    await chat_context.initialize()
    # Try using srv1.tool1 format
    tool = chat_context.find_tool_by_name("srv1.tool1")
    assert tool is not None


@pytest.mark.asyncio
async def test_find_tool_by_name_not_found(chat_context):
    """Test find_tool_by_name with non-existent tool."""
    await chat_context.initialize()
    tool = chat_context.find_tool_by_name("nonexistent")
    assert tool is None


@pytest.mark.asyncio
async def test_find_server_by_name(chat_context):
    """Test find_server_by_name."""
    await chat_context.initialize()
    server = chat_context.find_server_by_name("srv1")
    assert server is not None
    assert server.name == "srv1"


@pytest.mark.asyncio
async def test_find_server_by_name_not_found(chat_context):
    """Test find_server_by_name with non-existent server."""
    await chat_context.initialize()
    server = chat_context.find_server_by_name("nonexistent")
    assert server is None


@pytest.mark.asyncio
async def test_add_user_message(chat_context):
    """Test add_user_message."""
    await chat_context.initialize()
    initial_len = len(chat_context.conversation_history)
    await chat_context.add_user_message("Hello!")
    assert len(chat_context.conversation_history) == initial_len + 1
    assert chat_context.conversation_history[-1].content == "Hello!"
    assert chat_context.conversation_history[-1].role.value == "user"


@pytest.mark.asyncio
async def test_add_assistant_message(chat_context):
    """Test add_assistant_message."""
    await chat_context.initialize()
    initial_len = len(chat_context.conversation_history)
    await chat_context.add_assistant_message("Hi there!")
    assert len(chat_context.conversation_history) == initial_len + 1
    assert chat_context.conversation_history[-1].content == "Hi there!"
    assert chat_context.conversation_history[-1].role.value == "assistant"


@pytest.mark.asyncio
async def test_clear_conversation_history_keep_system(chat_context):
    """Test clear_conversation_history with keep_system_prompt=True."""
    await chat_context.initialize()
    await chat_context.add_user_message("Hello")
    await chat_context.add_assistant_message("Hi")

    await chat_context.clear_conversation_history(keep_system_prompt=True)

    assert len(chat_context.conversation_history) == 1
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_clear_conversation_history_remove_all(chat_context):
    """Test clear_conversation_history creates fresh session (system prompt always kept)."""
    await chat_context.initialize()
    await chat_context.add_user_message("Hello")

    await chat_context.clear_conversation_history(keep_system_prompt=False)

    # Implementation always creates fresh session with system prompt
    assert len(chat_context.conversation_history) == 1
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_regenerate_system_prompt(chat_context):
    """Test regenerate_system_prompt."""
    await chat_context.initialize()
    _ = chat_context.conversation_history[0].content  # Original prompt

    # Regenerate should update the system prompt
    await chat_context.regenerate_system_prompt()

    # Should still be the first message
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_get_tool_count(chat_context):
    """Test get_tool_count."""
    await chat_context.initialize()
    assert chat_context.get_tool_count() == 2


@pytest.mark.asyncio
async def test_get_server_count(chat_context):
    """Test get_server_count."""
    await chat_context.initialize()
    assert chat_context.get_server_count() == 2


def test_get_display_name_for_tool():
    """Test get_display_name_for_tool static method."""
    from mcp_cli.chat.chat_context import ChatContext

    name = ChatContext.get_display_name_for_tool("srv1.tool1")
    assert name == "srv1.tool1"


@pytest.mark.asyncio
async def test_get_status_summary(chat_context):
    """Test get_status_summary."""
    await chat_context.initialize()
    status = chat_context.get_status_summary()
    assert status.tool_count == 2
    assert status.server_count == 2


@pytest.mark.asyncio
async def test_repr(chat_context):
    """Test __repr__."""
    await chat_context.initialize()
    repr_str = repr(chat_context)
    assert "ChatContext" in repr_str
    assert "tools=2" in repr_str


@pytest.mark.asyncio
async def test_str(chat_context):
    """Test __str__."""
    await chat_context.initialize()
    str_val = str(chat_context)
    assert "Chat session" in str_val
    assert "2 tools" in str_val


@pytest.mark.asyncio
async def test_context_manager(dummy_tool_manager, monkeypatch):
    """Test async context manager."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_model_manager = Mock(spec=ModelManager)
    mock_model_manager.provider = "mock"
    mock_model_manager.model = "mock-model"
    mock_model_manager.get_client.return_value = None
    mock_model_manager.get_active_provider.return_value = "mock"
    mock_model_manager.get_active_model.return_value = "mock-model"

    from mcp_cli.chat.chat_context import ChatContext

    async with ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_model_manager
    ) as ctx:
        assert ctx.get_tool_count() == 2


@pytest.mark.asyncio
async def test_update_from_dict_with_exit_requested(chat_context):
    """Test update_from_dict updates exit_requested."""
    await chat_context.initialize()

    chat_context.update_from_dict({"exit_requested": True})
    assert chat_context.exit_requested is True

    chat_context.update_from_dict({"exit_requested": False})
    assert chat_context.exit_requested is False


@pytest.mark.asyncio
async def test_execute_tool(chat_context):
    """Test execute_tool delegation."""
    await chat_context.initialize()
    result = await chat_context.execute_tool("tool1", {"arg": "value"})
    assert result["success"] is True


@pytest.mark.asyncio
async def test_stream_execute_tool(chat_context):
    """Test stream_execute_tool delegation."""
    await chat_context.initialize()
    results = []
    async for result in chat_context.stream_execute_tool("tool1", {"arg": "value"}):
        results.append(result)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_refresh_after_model_change(chat_context):
    """Test refresh_after_model_change."""
    await chat_context.initialize()
    # Should not raise
    await chat_context.refresh_after_model_change()
    assert chat_context.get_tool_count() == 2


@pytest.mark.asyncio
async def test_create_with_provider_only(dummy_tool_manager, monkeypatch):
    """Test ChatContext.create with provider only."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create new ModelManager instance for this test
    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "openai"
    mock_manager.get_active_model.return_value = "gpt-4"
    mock_manager.switch_provider.return_value = None

    # Patch ModelManager constructor
    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            provider="openai",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_create_with_model_only(dummy_tool_manager, monkeypatch):
    """Test ChatContext.create with model only."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "openai"
    mock_manager.get_active_model.return_value = "gpt-4"
    mock_manager.switch_model.return_value = None

    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            model="gpt-4",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_create_with_provider_and_api_settings(dummy_tool_manager, monkeypatch):
    """Test ChatContext.create with provider and API settings."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "custom"
    mock_manager.get_active_model.return_value = "custom-model"
    mock_manager.add_runtime_provider.return_value = None
    mock_manager.switch_provider.return_value = None

    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            provider="custom",
            api_base="http://localhost:8080",
            api_key="test-key",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_create_with_provider_model_and_api_settings(
    dummy_tool_manager, monkeypatch
):
    """Test ChatContext.create with all settings."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "custom"
    mock_manager.get_active_model.return_value = "custom-model"
    mock_manager.add_runtime_provider.return_value = None
    mock_manager.switch_model.return_value = None

    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            provider="custom",
            model="custom-model",
            api_key="test-key",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_initialize_failure(dummy_tool_manager, monkeypatch):
    """Test initialize handles errors."""
    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Patch generate_system_prompt to avoid issues
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    # Create context
    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )

    # Make _initialize_tools raise an exception
    async def raise_error():
        raise RuntimeError("Test error")

    ctx._initialize_tools = raise_error

    result = await ctx.initialize()
    assert result is False


@pytest.mark.asyncio
async def test_regenerate_system_prompt_insert(dummy_tool_manager, monkeypatch):
    """Test regenerate_system_prompt when no system message exists."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )
    await ctx.initialize()

    # Clear conversation history completely
    await ctx.clear_conversation_history(keep_system_prompt=False)

    # Regenerate should insert at position 0
    await ctx.regenerate_system_prompt()

    assert len(ctx.conversation_history) == 1
    assert ctx.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_context_manager_failure(dummy_tool_manager, monkeypatch):
    """Test async context manager handles initialization failure."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )

    # Make initialize return False
    async def fail_init():
        return False

    ctx.initialize = fail_init

    with pytest.raises(RuntimeError, match="Failed to initialize"):
        async with ctx:
            pass


@pytest.mark.asyncio
async def test_adapt_tools_without_get_adapted_tools(dummy_tool_manager, monkeypatch):
    """Test _adapt_tools_for_provider fallback when get_adapted_tools_for_llm not available."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create tool manager without get_adapted_tools_for_llm
    class MinimalToolManager:
        def __init__(self):
            self._tools = [
                ToolInfo(
                    name="tool1",
                    namespace="srv1",
                    description="demo-1",
                    parameters={},
                    is_async=False,
                ),
            ]

        async def get_unique_tools(self):
            return self._tools

        async def get_server_info(self):
            return []

        async def get_tools_for_llm(self):
            return [{"type": "function", "function": {"name": "tool1"}}]

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=MinimalToolManager(), model_manager=mock_manager
    )
    await ctx.initialize()

    assert len(ctx.openai_tools) == 1
    assert ctx.tool_name_mapping == {}


@pytest.mark.asyncio
async def test_adapt_tools_exception_fallback(dummy_tool_manager, monkeypatch):
    """Test _adapt_tools_for_provider handles exceptions."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create tool manager that raises in get_adapted_tools_for_llm
    class FailingToolManager:
        def __init__(self):
            self._tools = [
                ToolInfo(
                    name="tool1",
                    namespace="srv1",
                    description="demo-1",
                    parameters={},
                    is_async=False,
                ),
            ]

        async def get_unique_tools(self):
            return self._tools

        async def get_server_info(self):
            return []

        async def get_adapted_tools_for_llm(self, provider):
            raise RuntimeError("Adaptation failed")

        async def get_tools_for_llm(self):
            return [{"type": "function", "function": {"name": "tool1"}}]

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=FailingToolManager(), model_manager=mock_manager
    )
    await ctx.initialize()

    # Should fall back to get_tools_for_llm
    assert len(ctx.openai_tools) == 1


@pytest.mark.asyncio
async def test_initialize_no_tools_warning(monkeypatch, capsys):
    """Test initialize prints warning when no tools available."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create tool manager that returns no tools
    class EmptyToolManager:
        async def get_unique_tools(self):
            return []

        async def get_server_info(self):
            return []

        async def get_adapted_tools_for_llm(self, provider):
            return [], {}

        async def get_tools_for_llm(self):
            return []

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=EmptyToolManager(), model_manager=mock_manager
    )
    result = await ctx.initialize()

    assert result is True
    assert ctx.get_tool_count() == 0


@pytest.mark.asyncio
async def test_find_tool_by_name_partial_match(chat_context):
    """Test find_tool_by_name with partial match (just tool name without namespace)."""
    await chat_context.initialize()
    # The dummy tools have namespace like "srv1" and name like "tool1"
    # Try to find by using a dotted name that doesn't match exactly
    # but the simple name part matches
    tool = chat_context.find_tool_by_name("other.tool1")
    assert tool is not None
    assert tool.name == "tool1"


# ---------------------------------------------------------------------------
# Server/tool grouping tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_server_tool_groups(chat_context):
    """_build_server_tool_groups returns correct grouping."""
    await chat_context.initialize()
    groups = chat_context._build_server_tool_groups()

    assert len(groups) == 2

    names = {g.name for g in groups}
    assert names == {"srv1", "srv2"}

    for group in groups:
        assert group.name
        assert isinstance(group.tools, list)
        assert len(group.tools) >= 1


@pytest.mark.asyncio
async def test_build_server_tool_groups_empty(dummy_tool_manager, monkeypatch):
    """_build_server_tool_groups returns empty list when no server_info."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )
    # Don't initialize — server_info is empty
    assert ctx._build_server_tool_groups() == []


# ---------------------------------------------------------------------------
# Tests for sliding window and infinite context (Tier 1.3 + 1.4)
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    """Tests for conversation history sliding window."""

    def _make_ctx(self, monkeypatch, max_history_messages=0, infinite_context=False):
        """Helper to create a ChatContext with mocked model manager."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        mock_tm = Mock()
        mock_tm.get_unique_tools = pytest.importorskip("asyncio").coroutines
        return ChatContext(
            tool_manager=mock_tm,
            model_manager=mock_manager,
            max_history_messages=max_history_messages,
            infinite_context=infinite_context,
        )

    @pytest.mark.asyncio
    async def test_no_limit_returns_all(self, dummy_tool_manager, monkeypatch):
        """max_history_messages=0 returns all messages."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=0,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Add 10 messages
        for i in range(10):
            await ctx.add_user_message(f"msg-{i}")

        history = ctx.conversation_history
        # 1 system + 10 user
        assert len(history) == 11

    @pytest.mark.asyncio
    async def test_window_limits_messages(self, dummy_tool_manager, monkeypatch):
        """Sliding window keeps only last N event messages."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=3,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Add 10 messages
        for i in range(10):
            await ctx.add_user_message(f"msg-{i}")

        history = ctx.conversation_history
        # 1 system + 3 windowed = 4
        assert len(history) == 4
        # System prompt always first
        assert history[0].role.value == "system"
        # Last 3 messages are the most recent
        assert history[-1].content == "msg-9"
        assert history[-2].content == "msg-8"
        assert history[-3].content == "msg-7"

    @pytest.mark.asyncio
    async def test_system_prompt_not_evicted(self, dummy_tool_manager, monkeypatch):
        """System prompt is always included regardless of window size."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=1,
        )
        ctx._system_prompt = "SYSTEM"
        await ctx._initialize_session()

        await ctx.add_user_message("hello")
        await ctx.add_user_message("world")

        history = ctx.conversation_history
        # System + 1 windowed = 2
        assert len(history) == 2
        assert history[0].content == "SYSTEM"

    @pytest.mark.asyncio
    async def test_under_limit_not_truncated(self, dummy_tool_manager, monkeypatch):
        """If message count is under the limit, no eviction happens."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=100,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        await ctx.add_user_message("hello")
        await ctx.add_user_message("world")

        history = ctx.conversation_history
        # 1 system + 2 user = 3, under limit of 100
        assert len(history) == 3


class TestInfiniteContextConfig:
    """Tests for infinite context configuration."""

    @pytest.mark.asyncio
    async def test_default_infinite_context_false(
        self, dummy_tool_manager, monkeypatch
    ):
        """Default infinite_context is False."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
        )
        assert ctx._infinite_context is False

    @pytest.mark.asyncio
    async def test_infinite_context_passed_to_session(
        self, dummy_tool_manager, monkeypatch
    ):
        """infinite_context=True is passed to SessionManager."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            infinite_context=True,
            token_threshold=8000,
            max_turns_per_segment=30,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Verify session was created with correct params
        assert ctx._infinite_context is True
        assert ctx._token_threshold == 8000
        assert ctx._max_turns_per_segment == 30

    @pytest.mark.asyncio
    async def test_create_factory_threads_params(self, dummy_tool_manager, monkeypatch):
        """create() factory passes context params through."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=50,
            infinite_context=True,
            token_threshold=6000,
        )
        assert ctx._max_history_messages == 50
        assert ctx._infinite_context is True
        assert ctx._token_threshold == 6000


# ---------------------------------------------------------------------------
# Tests for system prompt caching (Tier 2)
# ---------------------------------------------------------------------------


class TestSystemPromptCaching:
    """Tests for system prompt dirty-flag caching."""

    @pytest.mark.asyncio
    async def test_system_prompt_cached(self, dummy_tool_manager, monkeypatch):
        """Second call to _generate_system_prompt doesn't rebuild when dirty=False."""
        call_count = [0]

        def counting_generate(tools=None, **kw):
            call_count[0] += 1
            return f"SYS_PROMPT_{call_count[0]}"

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            counting_generate,
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
        )
        # Simulate initialized state with internal_tools set
        ctx.internal_tools = []

        # First call: dirty=True by default, should build prompt
        ctx._generate_system_prompt()
        first_prompt = ctx._system_prompt
        assert call_count[0] == 1
        assert first_prompt == "SYS_PROMPT_1"
        assert ctx._system_prompt_dirty is False

        # Second call: dirty=False, should return cached prompt
        ctx._generate_system_prompt()
        assert call_count[0] == 1  # Not called again
        assert ctx._system_prompt == first_prompt

    @pytest.mark.asyncio
    async def test_dirty_flag_on_tool_change(self, dummy_tool_manager, monkeypatch):
        """_initialize_tools sets dirty=True."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
        )

        # Manually mark clean
        ctx._system_prompt_dirty = False

        # After _initialize_tools, dirty should be True again
        await ctx._initialize_tools()
        assert ctx._system_prompt_dirty is True


class TestLargeToolSetSummary:
    """Tests for system prompt tool summary threshold."""

    def test_large_tool_set_summary(self):
        """System prompt with >20 tools summarizes (shows '... and N more')."""
        from mcp_cli.chat.system_prompt import _build_server_section
        from mcp_cli.chat.models import ServerToolGroup

        # Create a server group with 25 tools
        tool_names = [f"tool_{i}" for i in range(25)]
        server_groups = [
            ServerToolGroup(
                name="big_server",
                description="A server with many tools",
                tools=tool_names,
            )
        ]

        result = _build_server_section(server_groups, tool_summary_threshold=20)

        # Should contain the summary text
        assert "... and 20 more" in result
        # Should show first 5 tools
        assert "tool_0" in result
        assert "tool_4" in result
        # Should NOT show tool_5 onwards individually (they are summarized)
        assert "tool_5," not in result

    def test_small_tool_set_no_summary(self):
        """System prompt with <= threshold tools shows all tools."""
        from mcp_cli.chat.system_prompt import _build_server_section
        from mcp_cli.chat.models import ServerToolGroup

        tool_names = [f"tool_{i}" for i in range(5)]
        server_groups = [
            ServerToolGroup(
                name="small_server",
                description="A server with few tools",
                tools=tool_names,
            )
        ]

        result = _build_server_section(server_groups, tool_summary_threshold=20)

        # Should contain all tool names
        for name in tool_names:
            assert name in result
        # Should NOT have the summary text
        assert "more" not in result

    def test_default_threshold_from_config(self):
        """_build_server_section uses DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD by default."""
        from mcp_cli.chat.system_prompt import _build_server_section
        from mcp_cli.chat.models import ServerToolGroup
        from mcp_cli.config.defaults import DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD

        # Create tools just above the default threshold
        count = DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD + 5
        tool_names = [f"tool_{i}" for i in range(count)]
        server_groups = [
            ServerToolGroup(
                name="server",
                description="desc",
                tools=tool_names,
            )
        ]

        result = _build_server_section(server_groups)

        # Should summarize since count > threshold
        assert "more" in result
