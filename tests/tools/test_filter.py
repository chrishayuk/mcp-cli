# tests/tools/test_filter.py
"""
Comprehensive tests for tools/filter.py module.
Target: 90%+ coverage
"""

from mcp_cli.tools.filter import ToolFilter


class TestToolFilter:
    """Test ToolFilter class."""

    def test_tool_filter_initialization(self):
        """Test ToolFilter initialization."""
        tf = ToolFilter()
        assert tf.disabled_tools == set()
        assert tf.disabled_by_validation == set()
        assert tf.disabled_by_user == set()
        assert tf.auto_fix_enabled is True
        assert tf._validation_cache == {}
        assert tf._fix_stats == {"attempted": 0, "successful": 0, "failed": 0}

    def test_is_tool_enabled(self):
        """Test is_tool_enabled method."""
        tf = ToolFilter()
        assert tf.is_tool_enabled("test_tool") is True

        tf.disabled_tools.add("test_tool")
        assert tf.is_tool_enabled("test_tool") is False

    def test_disable_tool_user_reason(self):
        """Test disable_tool with user reason."""
        tf = ToolFilter()
        tf.disable_tool("tool1", reason="user")

        assert "tool1" in tf.disabled_tools
        assert "tool1" in tf.disabled_by_user
        assert "tool1" not in tf.disabled_by_validation

    def test_disable_tool_validation_reason(self):
        """Test disable_tool with validation reason."""
        tf = ToolFilter()
        tf.disable_tool("tool2", reason="validation")

        assert "tool2" in tf.disabled_tools
        assert "tool2" in tf.disabled_by_validation
        assert "tool2" not in tf.disabled_by_user

    def test_enable_tool(self):
        """Test enable_tool method."""
        tf = ToolFilter()
        tf.disable_tool("tool1", reason="user")
        tf.disable_tool("tool2", reason="validation")

        tf.enable_tool("tool1")
        assert "tool1" not in tf.disabled_tools
        assert "tool1" not in tf.disabled_by_user

        tf.enable_tool("tool2")
        assert "tool2" not in tf.disabled_tools
        assert "tool2" not in tf.disabled_by_validation

    def test_get_disabled_tools(self):
        """Test get_disabled_tools method."""
        tf = ToolFilter()
        tf.disable_tool("tool1", reason="user")
        tf.disable_tool("tool2", reason="validation")

        disabled = tf.get_disabled_tools()
        assert disabled["tool1"] == "user"
        assert disabled["tool2"] == "validation"

    def test_get_disabled_tools_by_reason(self):
        """Test get_disabled_tools_by_reason method."""
        tf = ToolFilter()
        tf.disable_tool("tool1", reason="user")
        tf.disable_tool("tool2", reason="user")
        tf.disable_tool("tool3", reason="validation")

        user_disabled = tf.get_disabled_tools_by_reason("user")
        assert "tool1" in user_disabled
        assert "tool2" in user_disabled
        assert "tool3" not in user_disabled

        validation_disabled = tf.get_disabled_tools_by_reason("validation")
        assert "tool3" in validation_disabled
        assert "tool1" not in validation_disabled

        unknown_disabled = tf.get_disabled_tools_by_reason("unknown")
        assert unknown_disabled == set()

    def test_clear_validation_disabled(self):
        """Test clear_validation_disabled method."""
        tf = ToolFilter()
        tf.disable_tool("tool1", reason="validation")
        tf.disable_tool("tool2", reason="user")

        tf.clear_validation_disabled()

        assert "tool1" not in tf.disabled_tools
        assert "tool1" not in tf.disabled_by_validation
        assert "tool2" in tf.disabled_tools  # User disabled should remain
        assert tf._validation_cache == {}
        assert tf._fix_stats == {"attempted": 0, "successful": 0, "failed": 0}

    def test_filter_tools_with_valid_openai_tools(self):
        """Test filter_tools with valid OpenAI tools."""
        tf = ToolFilter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "valid_tool",
                    "description": "A valid tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"arg1": {"type": "string"}},
                    },
                },
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        assert len(valid) == 1
        assert len(invalid) == 0
        assert valid[0]["function"]["name"] == "valid_tool"

    def test_filter_tools_with_invalid_openai_tools(self):
        """Test filter_tools with invalid OpenAI tools."""
        tf = ToolFilter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "invalid@tool",  # Invalid character
                    "description": "Invalid tool",
                },
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        assert len(valid) == 0
        assert len(invalid) == 1
        assert "invalid@tool" in tf.disabled_by_validation

    def test_filter_tools_auto_fix_enabled(self):
        """Test filter_tools with auto-fix enabled."""
        tf = ToolFilter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_with_extras",
                    "description": "Tool with unsupported props",
                    "title": "Should be removed",  # Unsupported
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        assert len(valid) == 1
        assert "title" not in valid[0]["function"]
        assert tf._fix_stats["successful"] > 0

    def test_filter_tools_auto_fix_disabled(self):
        """Test filter_tools with auto-fix disabled."""
        tf = ToolFilter()
        tf.set_auto_fix_enabled(False)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "Tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        assert len(valid) == 1
        assert tf._fix_stats["attempted"] == 0

    def test_filter_tools_with_manually_disabled(self):
        """Test filter_tools with manually disabled tools."""
        tf = ToolFilter()
        tf.disable_tool("disabled_tool", reason="user")

        tools = [
            {
                "type": "function",
                "function": {"name": "disabled_tool", "description": "Will be skipped"},
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        assert len(valid) == 0
        assert len(invalid) == 1
        assert invalid[0]["_disabled_reason"] == "user"

    def test_filter_tools_non_openai_provider(self):
        """Test filter_tools with non-OpenAI provider."""
        tf = ToolFilter()
        tools = [
            {
                "type": "function",
                "function": {"name": "any_tool", "description": "Any tool"},
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="anthropic")

        # Non-OpenAI providers pass through
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_extract_tool_name_with_function(self):
        """Test _extract_tool_name with function structure."""
        tf = ToolFilter()
        tool = {"function": {"name": "test_tool"}}
        assert tf._extract_tool_name(tool) == "test_tool"

    def test_extract_tool_name_without_function(self):
        """Test _extract_tool_name without function structure."""
        tf = ToolFilter()
        tool = {"name": "direct_tool"}
        assert tf._extract_tool_name(tool) == "direct_tool"

    def test_extract_tool_name_unknown(self):
        """Test _extract_tool_name with unknown structure."""
        tf = ToolFilter()
        tool = {}
        assert tf._extract_tool_name(tool) == "unknown"

    def test_try_fix_tool_openai(self):
        """Test _try_fix_tool for OpenAI provider."""
        tf = ToolFilter()
        tool = {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "Test",
                "title": "Remove me",
                "parameters": {"type": "object"},
            },
        }

        fixed = tf._try_fix_tool(tool, "openai")
        assert fixed is not None
        assert "title" not in fixed["function"]

    def test_try_fix_tool_non_openai(self):
        """Test _try_fix_tool for non-OpenAI provider."""
        tf = ToolFilter()
        tool = {"name": "tool"}
        fixed = tf._try_fix_tool(tool, "anthropic")
        assert fixed is None

    def test_get_validation_summary(self):
        """Test get_validation_summary method."""
        tf = ToolFilter()
        tf.disable_tool("tool1", reason="user")
        tf.disable_tool("tool2", reason="validation")

        summary = tf.get_validation_summary()

        assert summary["total_disabled"] == 2
        assert summary["disabled_by_validation"] == 1
        assert summary["disabled_by_user"] == 1
        assert summary["auto_fix_enabled"] is True
        assert "fix_stats" in summary

    def test_get_fix_statistics(self):
        """Test get_fix_statistics method."""
        tf = ToolFilter()
        stats = tf.get_fix_statistics()

        assert "attempted" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert stats["attempted"] == 0

    def test_reset_statistics(self):
        """Test reset_statistics method."""
        tf = ToolFilter()
        tf._fix_stats = {"attempted": 10, "successful": 8, "failed": 2}

        tf.reset_statistics()

        assert tf._fix_stats == {"attempted": 0, "successful": 0, "failed": 0}

    def test_set_auto_fix_enabled(self):
        """Test set_auto_fix_enabled method."""
        tf = ToolFilter()
        assert tf.auto_fix_enabled is True

        tf.set_auto_fix_enabled(False)
        assert tf.auto_fix_enabled is False

        tf.set_auto_fix_enabled(True)
        assert tf.auto_fix_enabled is True

    def test_is_auto_fix_enabled(self):
        """Test is_auto_fix_enabled method."""
        tf = ToolFilter()
        assert tf.is_auto_fix_enabled() is True

        tf.auto_fix_enabled = False
        assert tf.is_auto_fix_enabled() is False

    def test_filter_tools_with_array_schema_issues(self):
        """Test filter_tools with array schema that needs fixing."""
        tf = ToolFilter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "Tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array"}  # Missing items property
                        },
                    },
                },
            }
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        # Should be fixed automatically
        assert len(valid) == 1

    def test_filter_tools_mixed_valid_invalid(self):
        """Test filter_tools with mix of valid and invalid tools."""
        tf = ToolFilter()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "valid_tool",
                    "description": "Valid",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "invalid@tool",
                    "description": "Invalid",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "another_valid",
                    "description": "Valid",
                    "parameters": {"type": "object"},
                },
            },
        ]

        valid, invalid = tf.filter_tools(tools, provider="openai")

        assert len(valid) == 2
        assert len(invalid) == 1
        assert any(t["function"]["name"] == "valid_tool" for t in valid)
        assert any(t["function"]["name"] == "another_valid" for t in valid)
