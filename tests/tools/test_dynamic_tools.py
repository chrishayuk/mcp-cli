# tests/tools/test_dynamic_tools.py
"""Tests for dynamic tool discovery functionality."""

import pytest
from unittest.mock import AsyncMock

from chuk_tool_processor.discovery import DynamicToolName
from mcp_cli.tools.dynamic_tools import DynamicToolProvider
from mcp_cli.tools.models import ToolInfo, ToolCallResult


class DummyToolManager:
    """Mock tool manager for testing DynamicToolProvider."""

    def __init__(self, tools: list[ToolInfo] | None = None):
        self.tools = tools or []
        self.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool",
                success=True,
                result={"data": "test_result"},
            )
        )

    async def get_all_tools(self) -> list[ToolInfo]:
        return self.tools

    def format_tool_response(self, response):
        if isinstance(response, dict):
            import json

            return json.dumps(response)
        return str(response)


# ----------------------------------------------------------------------------
# DynamicToolName enum tests
# ----------------------------------------------------------------------------


def test_dynamic_tool_name_values():
    """Verify enum values match expected tool names."""
    assert DynamicToolName.LIST_TOOLS.value == "list_tools"
    assert DynamicToolName.SEARCH_TOOLS.value == "search_tools"
    assert DynamicToolName.GET_TOOL_SCHEMA.value == "get_tool_schema"
    assert DynamicToolName.CALL_TOOL.value == "call_tool"


def test_dynamic_tool_name_is_string_enum():
    """Ensure enum values can be used as strings."""
    assert isinstance(DynamicToolName.LIST_TOOLS.value, str)
    assert DynamicToolName.LIST_TOOLS == "list_tools"


# ----------------------------------------------------------------------------
# DynamicToolProvider tests
# ----------------------------------------------------------------------------


@pytest.fixture
def sample_tools() -> list[ToolInfo]:
    """Create sample tools for testing."""
    return [
        ToolInfo(
            name="calculator",
            namespace="math",
            description="Performs mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        ),
        ToolInfo(
            name="weather",
            namespace="api",
            description="Gets current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        ),
        ToolInfo(
            name="search",
            namespace="web",
            description="Searches the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        ),
    ]


@pytest.fixture
def provider(sample_tools) -> DynamicToolProvider:
    """Create a DynamicToolProvider with mock tool manager."""
    tool_manager = DummyToolManager(sample_tools)
    return DynamicToolProvider(tool_manager)


def test_get_dynamic_tools_returns_all_tools(provider):
    """Verify get_dynamic_tools returns all 5 dynamic tools."""
    tools = provider.get_dynamic_tools()

    assert len(tools) == 5

    tool_names = {t["function"]["name"] for t in tools}
    expected = {
        DynamicToolName.LIST_TOOLS.value,
        DynamicToolName.SEARCH_TOOLS.value,
        DynamicToolName.GET_TOOL_SCHEMA.value,
        DynamicToolName.GET_TOOL_SCHEMAS.value,
        DynamicToolName.CALL_TOOL.value,
    }
    assert tool_names == expected


def test_get_dynamic_tools_format(provider):
    """Verify dynamic tools follow OpenAI function format."""
    tools = provider.get_dynamic_tools()

    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_is_dynamic_tool_returns_true_for_dynamic_tools(provider):
    """Verify is_dynamic_tool correctly identifies dynamic tools."""
    assert provider.is_dynamic_tool("list_tools") is True
    assert provider.is_dynamic_tool("search_tools") is True
    assert provider.is_dynamic_tool("get_tool_schema") is True
    assert provider.is_dynamic_tool("call_tool") is True


def test_is_dynamic_tool_returns_false_for_regular_tools(provider):
    """Verify is_dynamic_tool returns False for non-dynamic tools."""
    assert provider.is_dynamic_tool("calculator") is False
    assert provider.is_dynamic_tool("weather") is False
    assert provider.is_dynamic_tool("some_random_tool") is False


@pytest.mark.asyncio
async def test_list_tools(provider):
    """Test list_tools returns tool summaries."""
    results = await provider.list_tools(limit=50)

    assert len(results) == 3
    names = {r["name"] for r in results}
    assert names == {"calculator", "weather", "search"}

    # Check structure
    for result in results:
        assert "name" in result
        assert "description" in result
        assert "namespace" in result


@pytest.mark.asyncio
async def test_list_tools_respects_limit(provider):
    """Test list_tools respects the limit parameter."""
    results = await provider.list_tools(limit=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_tools_finds_matches(provider):
    """Test search_tools finds tools matching query."""
    results = await provider.search_tools("calc", limit=10)

    assert len(results) >= 1
    assert any(r["name"] == "calculator" for r in results)


@pytest.mark.asyncio
async def test_search_tools_searches_descriptions(provider):
    """Test search_tools searches in descriptions."""
    results = await provider.search_tools("weather", limit=10)

    assert len(results) >= 1
    assert any(r["name"] == "weather" for r in results)


@pytest.mark.asyncio
async def test_search_tools_respects_limit(provider):
    """Test search_tools respects limit."""
    results = await provider.search_tools("a", limit=1)
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_get_tool_schema_returns_schema(provider):
    """Test get_tool_schema returns full schema for known tool."""
    schema = await provider.get_tool_schema("calculator")

    assert "function" in schema
    assert schema["function"]["name"] == "calculator"
    assert "parameters" in schema["function"]


@pytest.mark.asyncio
async def test_get_tool_schema_caches_results(provider):
    """Test get_tool_schema caches results."""
    # First call
    await provider.get_tool_schema("calculator")
    assert "calculator" in provider._tool_cache

    # Second call should use cache
    schema = await provider.get_tool_schema("calculator")
    assert schema["function"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_get_tool_schema_unknown_tool(provider):
    """Test get_tool_schema returns error for unknown tool."""
    schema = await provider.get_tool_schema("nonexistent_tool")
    assert "error" in schema


@pytest.mark.asyncio
async def test_call_tool_executes_tool(provider):
    """Test call_tool delegates to tool_manager."""
    # Must fetch schema first (enforced workflow)
    await provider.get_tool_schema("calculator")
    result = await provider.call_tool("calculator", {"expression": "2+2"})

    assert result["success"] is True
    provider.tool_manager.execute_tool.assert_called_once()


@pytest.mark.asyncio
async def test_execute_dynamic_tool_list_tools(provider):
    """Test execute_dynamic_tool handles list_tools."""
    result = await provider.execute_dynamic_tool("list_tools", {"limit": 10})

    assert "result" in result
    assert "count" in result
    assert "total_available" in result


@pytest.mark.asyncio
async def test_execute_dynamic_tool_search_tools(provider):
    """Test execute_dynamic_tool handles search_tools."""
    result = await provider.execute_dynamic_tool(
        "search_tools", {"query": "calc", "limit": 5}
    )

    assert "result" in result
    assert "count" in result


@pytest.mark.asyncio
async def test_execute_dynamic_tool_get_tool_schema(provider):
    """Test execute_dynamic_tool handles get_tool_schema."""
    result = await provider.execute_dynamic_tool(
        "get_tool_schema", {"tool_name": "calculator"}
    )

    assert "function" in result


@pytest.mark.asyncio
async def test_execute_dynamic_tool_call_tool(provider):
    """Test execute_dynamic_tool handles call_tool."""
    result = await provider.execute_dynamic_tool(
        "call_tool", {"tool_name": "calculator", "expression": "2+2"}
    )

    assert "success" in result


@pytest.mark.asyncio
async def test_execute_dynamic_tool_unknown(provider):
    """Test execute_dynamic_tool returns error for unknown tool."""
    result = await provider.execute_dynamic_tool("unknown_tool", {})

    assert "error" in result
    assert "Unknown dynamic tool" in result["error"]


# ----------------------------------------------------------------------------
# Additional coverage tests
# ----------------------------------------------------------------------------


@pytest.fixture
def tools_with_long_descriptions() -> list[ToolInfo]:
    """Create tools with long descriptions for truncation tests."""
    return [
        ToolInfo(
            name="verbose_tool",
            namespace="test",
            description="A" * 250,  # More than 200 chars
            parameters={"type": "object", "properties": {}},
        ),
    ]


@pytest.fixture
def provider_with_long_desc(tools_with_long_descriptions) -> DynamicToolProvider:
    """Create provider with long description tools."""
    tool_manager = DummyToolManager(tools_with_long_descriptions)
    return DynamicToolProvider(tool_manager)


@pytest.mark.asyncio
async def test_list_tools_truncates_long_descriptions(provider_with_long_desc):
    """Test list_tools truncates descriptions over 200 chars."""
    results = await provider_with_long_desc.list_tools(limit=50)

    assert len(results) == 1
    desc = results[0]["description"]
    assert len(desc) == 200  # 197 + "..."
    assert desc.endswith("...")


@pytest.mark.asyncio
async def test_list_tools_exception_handling():
    """Test list_tools handles exceptions gracefully."""
    tool_manager = DummyToolManager([])
    # Make get_all_tools raise an exception
    tool_manager.get_all_tools = AsyncMock(side_effect=RuntimeError("db error"))
    provider = DynamicToolProvider(tool_manager)

    results = await provider.list_tools()
    assert results == []


@pytest.mark.asyncio
async def test_search_tools_truncates_long_descriptions(provider_with_long_desc):
    """Test search_tools truncates descriptions over 200 chars."""
    results = await provider_with_long_desc.search_tools("verbose", limit=10)

    assert len(results) == 1
    desc = results[0]["description"]
    assert len(desc) == 200
    assert desc.endswith("...")


@pytest.mark.asyncio
async def test_search_tools_exception_handling():
    """Test search_tools handles exceptions gracefully."""
    tool_manager = DummyToolManager([])
    tool_manager.get_all_tools = AsyncMock(side_effect=RuntimeError("db error"))
    provider = DynamicToolProvider(tool_manager)

    results = await provider.search_tools("test")
    assert results == []


@pytest.mark.asyncio
async def test_get_tool_schema_exception_handling():
    """Test get_tool_schema handles exceptions gracefully."""
    tool_manager = DummyToolManager([])
    tool_manager.get_all_tools = AsyncMock(side_effect=RuntimeError("db error"))
    provider = DynamicToolProvider(tool_manager)

    result = await provider.get_tool_schema("any_tool")
    assert "error" in result
    assert "db error" in result["error"]


@pytest.mark.asyncio
async def test_call_tool_unwraps_tool_result():
    """Test call_tool unwraps nested ToolResult objects."""

    # Create a mock result with a nested .result attribute
    class NestedResult:
        def __init__(self):
            self.result = {"actual": "data"}

    tools = [
        ToolInfo(name="test", namespace="test", description="Test tool", parameters={}),
    ]
    tool_manager = DummyToolManager(tools)
    tool_manager.execute_tool = AsyncMock(
        return_value=ToolCallResult(
            tool_name="test",
            success=True,
            result=NestedResult(),
        )
    )

    provider = DynamicToolProvider(tool_manager)
    # Must fetch schema first (enforced workflow)
    await provider.get_tool_schema("test")
    result = await provider.call_tool("test", {})

    assert result["success"] is True


@pytest.mark.asyncio
async def test_call_tool_unwraps_content_dict():
    """Test call_tool extracts 'content' from result dicts."""
    tools = [
        ToolInfo(name="test", namespace="test", description="Test tool", parameters={}),
    ]
    tool_manager = DummyToolManager(tools)
    tool_manager.execute_tool = AsyncMock(
        return_value=ToolCallResult(
            tool_name="test",
            success=True,
            result={"content": "extracted_value"},
        )
    )

    provider = DynamicToolProvider(tool_manager)
    # Must fetch schema first (enforced workflow)
    await provider.get_tool_schema("test")
    result = await provider.call_tool("test", {})

    assert result["success"] is True


@pytest.mark.asyncio
async def test_call_tool_format_exception():
    """Test call_tool handles format_tool_response exception."""
    tools = [
        ToolInfo(name="test", namespace="test", description="Test tool", parameters={}),
    ]
    tool_manager = DummyToolManager(tools)
    tool_manager.execute_tool = AsyncMock(
        return_value=ToolCallResult(
            tool_name="test",
            success=True,
            result={"data": "test"},
        )
    )
    # Make format_tool_response raise an exception
    tool_manager.format_tool_response = lambda x: (_ for _ in ()).throw(
        ValueError("format error")
    )

    provider = DynamicToolProvider(tool_manager)
    # Must fetch schema first (enforced workflow)
    await provider.get_tool_schema("test")
    result = await provider.call_tool("test", {})

    # Should fallback to str() representation
    assert result["success"] is True
    assert "result" in result


@pytest.mark.asyncio
async def test_call_tool_failure():
    """Test call_tool handles tool execution failure."""
    tools = [
        ToolInfo(name="test", namespace="test", description="Test tool", parameters={}),
    ]
    tool_manager = DummyToolManager(tools)
    tool_manager.execute_tool = AsyncMock(
        return_value=ToolCallResult(
            tool_name="test",
            success=False,
            error="execution failed",
        )
    )

    provider = DynamicToolProvider(tool_manager)
    # Must fetch schema first (enforced workflow)
    await provider.get_tool_schema("test")
    result = await provider.call_tool("test", {})

    assert result["success"] is False
    assert result["error"] == "execution failed"


@pytest.mark.asyncio
async def test_call_tool_exception():
    """Test call_tool handles exceptions."""
    tools = [
        ToolInfo(name="test", namespace="test", description="Test tool", parameters={}),
    ]
    tool_manager = DummyToolManager(tools)
    tool_manager.execute_tool = AsyncMock(side_effect=RuntimeError("network error"))

    provider = DynamicToolProvider(tool_manager)
    # Must fetch schema first (enforced workflow)
    await provider.get_tool_schema("test")
    result = await provider.call_tool("test", {})

    assert result["success"] is False
    assert "network error" in result["error"]


@pytest.mark.asyncio
async def test_call_tool_auto_fetches_schema():
    """Test call_tool auto-fetches schema when not already fetched."""
    tools = [
        ToolInfo(name="test", namespace="test", description="Test tool", parameters={}),
    ]
    tool_manager = DummyToolManager(tools)

    provider = DynamicToolProvider(tool_manager)
    # Don't fetch schema explicitly - call directly
    # The provider should auto-fetch schema before execution
    result = await provider.call_tool("test", {})

    # Should succeed because schema is auto-fetched
    assert result["success"] is True
    # Schema should now be marked as fetched
    assert "test" in provider._schema_fetched


@pytest.mark.asyncio
async def test_get_tool_schema_with_no_description():
    """Test get_tool_schema handles tools with no description."""
    tools = [
        ToolInfo(
            name="no_desc_tool",
            namespace="test",
            description=None,
            parameters={"type": "object", "properties": {}},
        ),
    ]
    tool_manager = DummyToolManager(tools)
    provider = DynamicToolProvider(tool_manager)

    schema = await provider.get_tool_schema("no_desc_tool")

    assert schema["function"]["description"] == "No description provided"


@pytest.mark.asyncio
async def test_get_tool_schema_with_no_parameters():
    """Test get_tool_schema handles tools with no parameters."""
    tools = [
        ToolInfo(
            name="no_params_tool",
            namespace="test",
            description="A tool without params",
            parameters=None,
        ),
    ]
    tool_manager = DummyToolManager(tools)
    provider = DynamicToolProvider(tool_manager)

    schema = await provider.get_tool_schema("no_params_tool")

    assert schema["function"]["parameters"] == {"type": "object", "properties": {}}
