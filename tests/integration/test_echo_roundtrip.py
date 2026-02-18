# tests/integration/test_echo_roundtrip.py
"""Integration test: init → list tools → call tool → verify result → close.

Run with:
    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_manager_lifecycle(tool_manager_sqlite):
    """Verify the full ToolManager lifecycle with a real server."""
    tm = tool_manager_sqlite

    # 1. Should have tools available
    tools = await tm.get_unique_tools()
    assert len(tools) > 0, "No tools discovered from sqlite server"

    # 2. Should have server info
    servers = await tm.get_server_info()
    assert len(servers) > 0, "No server info available"

    # 3. Verify tool names are non-empty strings
    for tool in tools:
        assert tool.name, f"Tool has empty name: {tool}"
        assert isinstance(tool.name, str)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_execution(tool_manager_sqlite):
    """Verify a tool can be called and returns a result."""
    tm = tool_manager_sqlite

    # Find a tool to call (sqlite usually has list_tables or read_query)
    tools = await tm.get_unique_tools()
    tool_names = [t.name for t in tools]

    # Try common sqlite tool names
    target = None
    for candidate in ("list_tables", "list-tables", "read_query", "read-query"):
        if candidate in tool_names:
            target = candidate
            break

    if target is None:
        pytest.skip(f"No known sqlite tool found in: {tool_names[:10]}")

    # Execute the tool
    result = await tm.execute_tool(target, {})
    assert result.success, f"Tool {target} failed: {result.error}"
    assert result.result is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_tool_adaptation(tool_manager_sqlite):
    """Verify tools can be adapted for LLM consumption."""
    tm = tool_manager_sqlite

    adapted_tools, name_mapping = await tm.get_adapted_tools_for_llm("openai")
    assert len(adapted_tools) > 0, "No adapted tools produced"

    # Each adapted tool should have the OpenAI function format
    for tool in adapted_tools:
        assert "type" in tool
        assert "function" in tool
        assert "name" in tool["function"]
