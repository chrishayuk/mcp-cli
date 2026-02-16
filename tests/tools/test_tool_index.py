# tests/tools/test_tool_index.py
"""Tests for O(1) tool lookup index in ToolManager and ChatContext."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_cli.tools.models import ToolInfo
from mcp_cli.tools.manager import ToolManager


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_tools() -> list[ToolInfo]:
    """Create a set of test tools."""
    return [
        ToolInfo(name="search", namespace="server1", description="Search tool"),
        ToolInfo(name="read", namespace="server2", description="Read tool"),
        ToolInfo(name="search", namespace="default", description="Default search"),
        ToolInfo(name="write", namespace="server1", description="Write tool"),
    ]


@pytest.fixture
def manager():
    """ToolManager with mocked get_all_tools."""
    tm = ToolManager(config_file="dummy.json", servers=[])
    tm.get_all_tools = AsyncMock(return_value=_make_tools())
    return tm


# ──────────────────────────────────────────────────────────────────────────────
# ToolManager index tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_built_lazily(manager):
    """Index is None until first lookup, then built."""
    assert manager._tool_index is None
    tool = await manager.get_tool_by_name("search")
    assert manager._tool_index is not None
    assert tool is not None
    assert tool.name == "search"


@pytest.mark.asyncio
async def test_lookup_by_name(manager):
    """Simple name lookup returns correct tool."""
    tool = await manager.get_tool_by_name("read")
    assert tool is not None
    assert tool.name == "read"
    assert tool.namespace == "server2"


@pytest.mark.asyncio
async def test_lookup_prefers_non_default_namespace(manager):
    """When duplicate names exist, non-default namespace wins."""
    tool = await manager.get_tool_by_name("search")
    assert tool is not None
    assert tool.namespace == "server1"  # Not "default"


@pytest.mark.asyncio
async def test_lookup_by_namespace(manager):
    """Namespace-qualified lookup returns correct tool."""
    tool = await manager.get_tool_by_name("search", namespace="default")
    assert tool is not None
    assert tool.namespace == "default"


@pytest.mark.asyncio
async def test_lookup_missing_tool(manager):
    """Missing tool returns None."""
    tool = await manager.get_tool_by_name("nonexistent")
    assert tool is None


@pytest.mark.asyncio
async def test_lookup_missing_namespace(manager):
    """Tool exists but not in requested namespace → None."""
    tool = await manager.get_tool_by_name("read", namespace="server1")
    assert tool is None


@pytest.mark.asyncio
async def test_index_invalidated_on_disable(manager):
    """disable_tool() invalidates the index."""
    await manager.get_tool_by_name("search")  # Build index
    assert manager._tool_index is not None

    manager.disable_tool("search")
    assert manager._tool_index is None


@pytest.mark.asyncio
async def test_index_invalidated_on_enable(manager):
    """enable_tool() invalidates the index."""
    await manager.get_tool_by_name("search")  # Build index
    assert manager._tool_index is not None

    manager.enable_tool("search")
    assert manager._tool_index is None


@pytest.mark.asyncio
async def test_index_rebuilt_after_invalidation(manager):
    """After invalidation, next lookup rebuilds the index."""
    await manager.get_tool_by_name("search")  # Build
    manager._invalidate_caches()  # Invalidate
    assert manager._tool_index is None

    tool = await manager.get_tool_by_name("search")  # Rebuild
    assert manager._tool_index is not None
    assert tool is not None


@pytest.mark.asyncio
async def test_index_uses_single_call(manager):
    """Multiple lookups use the cached index (only one get_all_tools call)."""
    await manager.get_tool_by_name("search")
    await manager.get_tool_by_name("read")
    await manager.get_tool_by_name("write")

    # get_all_tools should only be called once (for the initial build)
    assert manager.get_all_tools.call_count == 1


@pytest.mark.asyncio
async def test_fully_qualified_lookup(manager):
    """Lookup by fully qualified name (namespace.name) works."""
    tool = await manager.get_tool_by_name("search", namespace="server1")
    assert tool is not None
    assert tool.namespace == "server1"
    assert tool.name == "search"


# ──────────────────────────────────────────────────────────────────────────────
# ChatContext index tests
# ──────────────────────────────────────────────────────────────────────────────


class TestChatContextToolIndex:
    """Test ChatContext.find_tool_by_name() with O(1) index."""

    def _make_context_with_tools(self):
        """Create a minimal ChatContext-like object with _tool_index populated."""
        from mcp_cli.chat.chat_context import ChatContext

        # Build a ChatContext with mocked dependencies
        tool_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.provider = "openai"
        model_manager.model = "gpt-4"
        model_manager.api_base = None
        model_manager.api_key = None

        ctx = ChatContext(tool_manager=tool_manager, model_manager=model_manager)

        # Manually populate tools and index (simulating _initialize_tools)
        ctx.tools = [
            ToolInfo(name="search", namespace="server1", description="Search"),
            ToolInfo(name="read", namespace="server2", description="Read"),
        ]
        ctx._tool_index = {}
        for tool in ctx.tools:
            ctx._tool_index[tool.name] = tool
            if tool.namespace:
                ctx._tool_index[tool.fully_qualified_name] = tool

        return ctx

    def test_find_by_simple_name(self):
        ctx = self._make_context_with_tools()
        tool = ctx.find_tool_by_name("search")
        assert tool is not None
        assert tool.name == "search"

    def test_find_by_fully_qualified_name(self):
        ctx = self._make_context_with_tools()
        tool = ctx.find_tool_by_name("server2.read")
        assert tool is not None
        assert tool.name == "read"

    def test_find_dotted_name_fallback(self):
        """Dotted name falls back to simple name extraction."""
        ctx = self._make_context_with_tools()
        tool = ctx.find_tool_by_name("unknown_ns.search")
        assert tool is not None
        assert tool.name == "search"

    def test_find_missing_returns_none(self):
        ctx = self._make_context_with_tools()
        tool = ctx.find_tool_by_name("nonexistent")
        assert tool is None

    def test_empty_index_returns_none(self):
        ctx = self._make_context_with_tools()
        ctx._tool_index = {}
        tool = ctx.find_tool_by_name("search")
        assert tool is None


# ──────────────────────────────────────────────────────────────────────────────
# Startup progress tests (Step 3.3)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_progress_callback_called(manager):
    """on_progress callback is called during initialize."""
    progress_msgs: list[str] = []
    manager._on_progress = lambda msg: progress_msgs.append(msg)
    manager._report_progress("test message")
    assert "test message" in progress_msgs


@pytest.mark.asyncio
async def test_progress_callback_none_is_safe(manager):
    """No crash when on_progress is None."""
    manager._on_progress = None
    manager._report_progress("test message")  # Should not raise


# ──────────────────────────────────────────────────────────────────────────────
# LLM tools cache tests (Step 3.2)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cache_manager():
    """ToolManager with mocked get_all_tools for cache testing."""
    tm = ToolManager(config_file="dummy.json", servers=[])
    tm.get_all_tools = AsyncMock(return_value=_make_tools())
    return tm


@pytest.mark.asyncio
async def test_llm_cache_hit(cache_manager):
    """Second call returns cached result without calling get_all_tools again."""
    result1 = await cache_manager.get_tools_for_llm("openai")
    result2 = await cache_manager.get_tools_for_llm("openai")

    assert result1 == result2
    # get_all_tools called only once (for the first build)
    assert cache_manager.get_all_tools.call_count == 1


@pytest.mark.asyncio
async def test_llm_cache_per_provider(cache_manager):
    """Different providers get separate cache entries."""
    result_openai = await cache_manager.get_tools_for_llm("openai")
    result_anthropic = await cache_manager.get_tools_for_llm("anthropic")

    # Both should work (may have same content but separate cache entries)
    assert isinstance(result_openai, list)
    assert isinstance(result_anthropic, list)
    # Two calls to get_all_tools (one per provider)
    assert cache_manager.get_all_tools.call_count == 2


@pytest.mark.asyncio
async def test_llm_cache_invalidated_on_disable(cache_manager):
    """disable_tool() clears the LLM tools cache."""
    await cache_manager.get_tools_for_llm("openai")
    assert "openai" in cache_manager._llm_tools_cache

    cache_manager.disable_tool("search")
    assert cache_manager._llm_tools_cache == {}


@pytest.mark.asyncio
async def test_llm_cache_invalidated_on_enable(cache_manager):
    """enable_tool() clears the LLM tools cache."""
    await cache_manager.get_tools_for_llm("openai")
    assert "openai" in cache_manager._llm_tools_cache

    cache_manager.enable_tool("search")
    assert cache_manager._llm_tools_cache == {}


@pytest.mark.asyncio
async def test_llm_cache_rebuild_after_invalidation(cache_manager):
    """After cache invalidation, next call rebuilds."""
    await cache_manager.get_tools_for_llm("openai")
    cache_manager._invalidate_caches()
    await cache_manager.get_tools_for_llm("openai")

    # Two builds: initial + after invalidation
    assert cache_manager.get_all_tools.call_count == 2
