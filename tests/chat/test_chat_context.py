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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
    original_len = chat_context.get_conversation_length()

    exported = chat_context.to_dict()

    # mutate exported copy
    exported["exit_requested"] = True
    exported["conversation_history"].append({"role": "user", "content": "Hi"})

    chat_context.update_from_dict(exported)

    assert chat_context.exit_requested is True
    assert chat_context.get_conversation_length() == original_len + 1
    assert chat_context.conversation_history[-1].content == "Hi"


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
    chat_context.add_user_message("Hello!")
    assert len(chat_context.conversation_history) == initial_len + 1
    assert chat_context.conversation_history[-1].content == "Hello!"
    assert chat_context.conversation_history[-1].role.value == "user"


@pytest.mark.asyncio
async def test_add_assistant_message(chat_context):
    """Test add_assistant_message."""
    await chat_context.initialize()
    initial_len = len(chat_context.conversation_history)
    chat_context.add_assistant_message("Hi there!")
    assert len(chat_context.conversation_history) == initial_len + 1
    assert chat_context.conversation_history[-1].content == "Hi there!"
    assert chat_context.conversation_history[-1].role.value == "assistant"


@pytest.mark.asyncio
async def test_clear_conversation_history_keep_system(chat_context):
    """Test clear_conversation_history with keep_system_prompt=True."""
    await chat_context.initialize()
    chat_context.add_user_message("Hello")
    chat_context.add_assistant_message("Hi")

    chat_context.clear_conversation_history(keep_system_prompt=True)

    assert len(chat_context.conversation_history) == 1
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_clear_conversation_history_remove_all(chat_context):
    """Test clear_conversation_history with keep_system_prompt=False."""
    await chat_context.initialize()
    chat_context.add_user_message("Hello")

    chat_context.clear_conversation_history(keep_system_prompt=False)

    assert len(chat_context.conversation_history) == 0


@pytest.mark.asyncio
async def test_regenerate_system_prompt(chat_context):
    """Test regenerate_system_prompt."""
    await chat_context.initialize()
    _ = chat_context.conversation_history[0].content  # Original prompt

    # Regenerate should update the system prompt
    chat_context.regenerate_system_prompt()

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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
async def test_update_from_dict_with_message_objects(chat_context):
    """Test update_from_dict with Message objects instead of dicts."""
    await chat_context.initialize()

    from mcp_cli.chat.models import Message, MessageRole

    new_messages = [
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi"),
    ]

    chat_context.update_from_dict({"conversation_history": new_messages})

    assert len(chat_context.conversation_history) == 2
    assert chat_context.conversation_history[0].content == "Hello"


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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
    ctx.conversation_history = []

    # Regenerate should insert at position 0
    ctx.regenerate_system_prompt()

    assert len(ctx.conversation_history) == 1
    assert ctx.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_context_manager_failure(dummy_tool_manager, monkeypatch):
    """Test async context manager handles initialization failure."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
        "mcp_cli.chat.chat_context.generate_system_prompt", lambda tools: "SYS_PROMPT"
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
