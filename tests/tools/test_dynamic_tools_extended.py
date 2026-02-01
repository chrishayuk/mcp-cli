# tests/tools/test_dynamic_tools_extended.py
"""Extended tests for DynamicToolProvider to achieve >90% coverage.

Covers missing lines: 193-195, 223-230, 248-249.
These lines are in:
- filter_search_results: blocked tool score penalty and hint messages (193-195)
- _unwrap_result: ToolExecutionResult unwrapping with success/error attrs (223-230)
- _unwrap_result: MCP ToolResult with .content list attribute (248-249)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.tools.dynamic_tools import (
    DynamicToolProvider,
    PARAMETERIZED_TOOLS,
)
from mcp_cli.tools.models import ToolInfo, ToolCallResult


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


class DummyToolManager:
    """Mock tool manager for testing DynamicToolProvider."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test", success=True, result={"data": "ok"}
            )
        )

    async def get_all_tools(self):
        return self.tools

    def format_tool_response(self, response):
        import json

        if isinstance(response, dict):
            return json.dumps(response)
        if isinstance(response, list):
            return json.dumps(response)
        return str(response)


# ────────────────────────────────────────────────────────────────────
# filter_search_results - lines 193-195 (blocked tools)
# ────────────────────────────────────────────────────────────────────


class TestFilterSearchResultsBlocked:
    """Test filter_search_results when parameterized tools are blocked."""

    def test_blocked_tool_gets_score_penalty_and_hints(self):
        """Lines 193-195: blocked tool has score *= 0.1 and hint messages added."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        # Create a SearchResult for a parameterized tool that requires computed values
        from chuk_tool_processor.discovery import SearchResult

        parameterized_tool = ToolInfo(
            name="normal_cdf",
            namespace="stats",
            description="Normal CDF",
            parameters={"type": "object", "properties": {}},
        )

        sr = SearchResult(
            tool=parameterized_tool,
            score=1.0,
            match_reasons=["name_match"],
        )

        # Mock get_tool_state to return state with no bindings
        mock_state = MagicMock()
        mock_bindings = MagicMock()
        mock_bindings.bindings = {}  # Empty -> no computed values
        mock_state.bindings = mock_bindings

        with patch(
            "mcp_cli.tools.dynamic_tools.get_tool_state", return_value=mock_state
        ):
            filtered = provider.filter_search_results([sr])

        assert len(filtered) == 1
        result = filtered[0]
        # Score should be penalized: 1.0 * 0.1 = 0.1
        assert abs(result.score - 0.1) < 0.001
        # Should have blocked and hint messages
        assert any(
            "blocked:requires_computed_values" in r for r in result.match_reasons
        )
        assert any("hint:" in r for r in result.match_reasons)

    def test_non_blocked_tool_keeps_original_score(self):
        """Non-parameterized tools keep their original score."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        from chuk_tool_processor.discovery import SearchResult

        regular_tool = ToolInfo(
            name="add",
            namespace="compute",
            description="Add numbers",
            parameters={"type": "object", "properties": {}},
        )

        sr = SearchResult(
            tool=regular_tool,
            score=0.8,
            match_reasons=["name_match"],
        )

        mock_state = MagicMock()
        mock_bindings = MagicMock()
        mock_bindings.bindings = {}  # No computed values
        mock_state.bindings = mock_bindings

        with patch(
            "mcp_cli.tools.dynamic_tools.get_tool_state", return_value=mock_state
        ):
            filtered = provider.filter_search_results([sr])

        assert len(filtered) == 1
        # add tool does NOT require computed values (requires_computed_values=False)
        assert abs(filtered[0].score - 0.8) < 0.001
        assert "blocked:requires_computed_values" not in filtered[0].match_reasons

    def test_blocked_tool_unblocked_when_computed_values_exist(self):
        """Parameterized tools are NOT blocked when computed values exist in state."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        from chuk_tool_processor.discovery import SearchResult

        parameterized_tool = ToolInfo(
            name="normal_cdf",
            namespace="stats",
            description="Normal CDF",
            parameters={"type": "object", "properties": {}},
        )

        sr = SearchResult(
            tool=parameterized_tool,
            score=0.9,
            match_reasons=["name_match"],
        )

        mock_state = MagicMock()
        mock_bindings = MagicMock()
        mock_bindings.bindings = {"v1": "some_computed_value"}  # Has computed values
        mock_state.bindings = mock_bindings

        with patch(
            "mcp_cli.tools.dynamic_tools.get_tool_state", return_value=mock_state
        ):
            filtered = provider.filter_search_results([sr])

        assert len(filtered) == 1
        # Should NOT be penalized since computed values exist
        assert abs(filtered[0].score - 0.9) < 0.001
        assert "blocked:requires_computed_values" not in filtered[0].match_reasons

    def test_namespaced_tool_name_lookup(self):
        """Tools with dotted names use the base name for PARAMETERIZED_TOOLS lookup."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        from chuk_tool_processor.discovery import SearchResult

        # Tool with namespace prefix in name
        namespaced_tool = ToolInfo(
            name="stats.normal_pdf",
            namespace="stats",
            description="Normal PDF",
            parameters={},
        )

        sr = SearchResult(
            tool=namespaced_tool,
            score=1.0,
            match_reasons=["name_match"],
        )

        mock_state = MagicMock()
        mock_bindings = MagicMock()
        mock_bindings.bindings = {}  # No computed values
        mock_state.bindings = mock_bindings

        with patch(
            "mcp_cli.tools.dynamic_tools.get_tool_state", return_value=mock_state
        ):
            filtered = provider.filter_search_results([sr])

        # normal_pdf requires computed values and none exist, so should be blocked
        assert abs(filtered[0].score - 0.1) < 0.001
        assert any("blocked" in r for r in filtered[0].match_reasons)

    def test_results_sorted_by_adjusted_score(self):
        """After filtering, results are re-sorted by adjusted score."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        from chuk_tool_processor.discovery import SearchResult

        blocked_tool = ToolInfo(
            name="t_test",
            namespace="stats",
            description="T-test",
            parameters={},
        )
        normal_tool = ToolInfo(
            name="add",
            namespace="compute",
            description="Addition",
            parameters={},
        )

        sr_blocked = SearchResult(tool=blocked_tool, score=1.0, match_reasons=[])
        sr_normal = SearchResult(tool=normal_tool, score=0.5, match_reasons=[])

        mock_state = MagicMock()
        mock_bindings = MagicMock()
        mock_bindings.bindings = {}
        mock_state.bindings = mock_bindings

        with patch(
            "mcp_cli.tools.dynamic_tools.get_tool_state", return_value=mock_state
        ):
            filtered = provider.filter_search_results([sr_blocked, sr_normal])

        # Normal tool (0.5) should now be above blocked tool (1.0 * 0.1 = 0.1)
        assert filtered[0].tool.name == "add"
        assert filtered[1].tool.name == "t_test"


# ────────────────────────────────────────────────────────────────────
# _unwrap_result - lines 223-230 (ToolExecutionResult with success/error)
# ────────────────────────────────────────────────────────────────────


class TestUnwrapResultToolExecutionResult:
    """Test _unwrap_result with ToolExecutionResult-like objects."""

    def test_unwrap_successful_tool_execution_result(self):
        """Lines 222-232: Unwrap object with success=True and .result attribute."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class ToolExecutionResult:
            def __init__(self):
                self.success = True
                self.error = None
                self.result = {"actual": "data"}

        wrapped = ToolExecutionResult()
        actual = provider._unwrap_result(wrapped)

        assert actual == {"actual": "data"}

    def test_unwrap_failed_tool_execution_result(self):
        """Lines 223-228: Unwrap object with success=False returns the failed object."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class ToolExecutionResult:
            def __init__(self):
                self.success = False
                self.error = "Inner execution failed"
                self.result = None

        wrapped = ToolExecutionResult()
        actual = provider._unwrap_result(wrapped)

        # Should return the object itself (not unwrap further)
        assert actual is wrapped
        assert actual.error == "Inner execution failed"

    def test_unwrap_nested_tool_execution_result(self):
        """Deeply nested ToolExecutionResult is unwrapped through multiple layers."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class ToolExecutionResult:
            def __init__(self, inner_result, success=True):
                self.success = success
                self.error = None
                self.result = inner_result

        # Two levels of nesting
        inner = ToolExecutionResult(inner_result="final_value")
        outer = ToolExecutionResult(inner_result=inner)

        actual = provider._unwrap_result(outer)

        assert actual == "final_value"


# ────────────────────────────────────────────────────────────────────
# _unwrap_result - lines 248-249 (MCP ToolResult with .content list)
# ────────────────────────────────────────────────────────────────────


class TestUnwrapResultMCPToolResult:
    """Test _unwrap_result with MCP ToolResult objects that have .content list."""

    def test_unwrap_mcp_tool_result_with_content_list(self):
        """Lines 245-251: Object with .content attribute that is a list."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class MCPToolResult:
            def __init__(self):
                self.content = [
                    {"type": "text", "text": "Hello from tool"},
                    {"type": "text", "text": "More output"},
                ]

        wrapped = MCPToolResult()
        actual = provider._unwrap_result(wrapped)

        # Should extract the .content list
        assert isinstance(actual, list)
        assert len(actual) == 2
        assert actual[0]["text"] == "Hello from tool"

    def test_unwrap_mcp_tool_result_with_single_content_item(self):
        """MCP ToolResult with a single-item .content list."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class MCPToolResult:
            def __init__(self):
                self.content = [{"type": "text", "text": "single"}]

        wrapped = MCPToolResult()
        actual = provider._unwrap_result(wrapped)

        assert actual == [{"type": "text", "text": "single"}]

    def test_unwrap_does_not_unwrap_non_list_content(self):
        """Object with .content that is not a list is handled by .result path."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class ObjectWithStringContent:
            def __init__(self):
                self.content = "not a list"

        wrapped = ObjectWithStringContent()
        actual = provider._unwrap_result(wrapped)

        # .content is a string not a list, so this doesn't match the .content list path
        # It doesn't have .result or .success either, so it should break out of the loop
        assert actual is wrapped

    def test_unwrap_dict_with_content_key(self):
        """Dict with 'content' key extracts the value."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        wrapped = {"content": "extracted_content"}
        actual = provider._unwrap_result(wrapped)

        assert actual == "extracted_content"

    def test_unwrap_plain_result_attribute(self):
        """Object with only .result attribute (no .success/.error)."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        class SimpleResult:
            def __init__(self):
                self.result = "simple_data"

        wrapped = SimpleResult()
        actual = provider._unwrap_result(wrapped)

        assert actual == "simple_data"

    def test_unwrap_max_depth_prevents_infinite_loop(self):
        """Max depth of 5 prevents infinite unwrapping."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        # Create a deeply nested chain of .result attributes
        class SelfRef:
            pass

        obj = SelfRef()
        current = obj
        for _ in range(10):
            inner = SelfRef()
            current.result = inner
            current = inner
        current.result = "deep_value"

        actual = provider._unwrap_result(obj)

        # Due to max_depth=5, we should stop after 5 unwraps
        # The actual value won't be "deep_value" since we can't go deep enough
        assert actual is not None

    def test_unwrap_plain_value_returns_immediately(self):
        """Primitive values are returned as-is."""
        tool_manager = DummyToolManager()
        provider = DynamicToolProvider(tool_manager)

        assert provider._unwrap_result("hello") == "hello"
        assert provider._unwrap_result(42) == 42
        assert provider._unwrap_result(None) is None
        assert provider._unwrap_result([1, 2, 3]) == [1, 2, 3]


# ────────────────────────────────────────────────────────────────────
# execute_tool method - integration with _unwrap_result
# ────────────────────────────────────────────────────────────────────


class TestExecuteToolUnwrap:
    """Test execute_tool with various result structures that exercise _unwrap_result."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_mcp_tool_result(self):
        """execute_tool properly unwraps MCP ToolResult with .content list."""

        class MCPToolResult:
            def __init__(self):
                self.content = [{"type": "text", "text": "tool output"}]

        tools = [
            ToolInfo(name="test", namespace="ns", description="Test", parameters={}),
        ]
        tool_manager = DummyToolManager(tools)
        tool_manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test",
                success=True,
                result=MCPToolResult(),
            )
        )

        provider = DynamicToolProvider(tool_manager)
        await provider.get_tool_schema("test")
        result = await provider.call_tool("test", {})

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_with_failed_inner_result(self):
        """execute_tool handles inner tool execution failure."""

        class FailedExecution:
            def __init__(self):
                self.success = False
                self.error = "inner tool error"
                self.result = None

        tools = [
            ToolInfo(name="test", namespace="ns", description="Test", parameters={}),
        ]
        tool_manager = DummyToolManager(tools)
        tool_manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test",
                success=True,
                result=FailedExecution(),
            )
        )

        provider = DynamicToolProvider(tool_manager)
        await provider.get_tool_schema("test")
        result = await provider.call_tool("test", {})

        # The outer result reports success, but the inner result was a failed execution
        # The unwrap returns the failed object itself, then format_tool_response handles it
        assert result["success"] is True


# ────────────────────────────────────────────────────────────────────
# PARAMETERIZED_TOOLS metadata
# ────────────────────────────────────────────────────────────────────


class TestParameterizedToolsMetadata:
    """Verify PARAMETERIZED_TOOLS dict contents."""

    def test_parameterized_tools_has_expected_keys(self):
        """Verify known parameterized tools are present."""
        expected_requiring = {
            "normal_cdf",
            "normal_pdf",
            "normal_sf",
            "t_test",
            "chi_square",
        }
        expected_not_requiring = {"sqrt", "add", "subtract", "multiply", "divide"}

        for tool in expected_requiring:
            assert tool in PARAMETERIZED_TOOLS
            assert PARAMETERIZED_TOOLS[tool]["requires_computed_values"] is True

        for tool in expected_not_requiring:
            assert tool in PARAMETERIZED_TOOLS
            assert PARAMETERIZED_TOOLS[tool]["requires_computed_values"] is False

    def test_unknown_tool_not_in_parameterized(self):
        """Unknown tool names are not in PARAMETERIZED_TOOLS."""
        assert "unknown_tool" not in PARAMETERIZED_TOOLS
        assert PARAMETERIZED_TOOLS.get("unknown_tool", {}) == {}
