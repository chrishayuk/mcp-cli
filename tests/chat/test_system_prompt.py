# tests/chat/test_system_prompt.py
"""Tests for mcp_cli.chat.system_prompt."""


class TestGenerateSystemPrompt:
    """Tests for generate_system_prompt (normal / non-dynamic mode)."""

    def test_no_tools_none(self):
        """Calling with tools=None should mention 0 tools."""
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=None)
        assert "0 tools" in result
        assert isinstance(result, str)
        assert len(result) > 100  # sanity: prompt is non-trivial

    def test_no_tools_empty_list(self):
        """Calling with an empty list should mention 0 tools."""
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=[])
        assert "0 tools" in result

    def test_with_tools(self):
        """Tool count should be reflected in the prompt."""
        from mcp_cli.chat.system_prompt import generate_system_prompt

        fake_tools = [{"name": f"tool_{i}"} for i in range(5)]
        result = generate_system_prompt(tools=fake_tools)
        assert "5 tools" in result

    def test_prompt_contains_guidelines(self):
        """The normal prompt should contain key sections."""
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=[{"name": "t"}])
        assert "GENERAL GUIDELINES" in result
        assert "Step-by-step reasoning" in result
        assert "Effective tool usage" in result
        assert "REMEMBER" in result

    def test_not_dynamic_when_env_unset(self, monkeypatch):
        """Without MCP_CLI_DYNAMIC_TOOLS, should return the normal prompt."""
        monkeypatch.delenv("MCP_CLI_DYNAMIC_TOOLS", raising=False)
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=[])
        # Normal prompt does NOT contain "TOOL DISCOVERY SYSTEM"
        assert "TOOL DISCOVERY SYSTEM" not in result

    def test_not_dynamic_when_env_zero(self, monkeypatch):
        """MCP_CLI_DYNAMIC_TOOLS=0 should return normal prompt."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "0")
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=[])
        assert "TOOL DISCOVERY SYSTEM" not in result


class TestGenerateDynamicToolsPrompt:
    """Tests for the dynamic tools path (_generate_dynamic_tools_prompt)."""

    def test_dynamic_mode_via_env(self, monkeypatch):
        """Setting MCP_CLI_DYNAMIC_TOOLS=1 should trigger dynamic prompt."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "1")
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=None)
        assert "TOOL DISCOVERY SYSTEM" in result
        assert "0 tools" in result

    def test_dynamic_mode_with_tools(self, monkeypatch):
        """Dynamic prompt should include tool count."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "1")
        from mcp_cli.chat.system_prompt import generate_system_prompt

        fake_tools = ["a", "b", "c"]
        result = generate_system_prompt(tools=fake_tools)
        assert "3 tools" in result

    def test_dynamic_prompt_contains_workflow(self, monkeypatch):
        """Dynamic prompt should describe the discovery workflow."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "1")
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=[])
        assert "search_tools" in result
        assert "list_tools" in result
        assert "get_tool_schema" in result
        assert "call_tool" in result
        assert "WORKFLOW EXAMPLE" in result
        assert "CRITICAL RULES" in result

    def test_dynamic_prompt_with_none_tools(self, monkeypatch):
        """Dynamic prompt with tools=None should still work."""
        monkeypatch.setenv("MCP_CLI_DYNAMIC_TOOLS", "1")
        from mcp_cli.chat.system_prompt import generate_system_prompt

        result = generate_system_prompt(tools=None)
        assert "0 tools" in result


class TestPrivateFunctionDirectly:
    """Direct calls to _generate_dynamic_tools_prompt for completeness."""

    def test_direct_call_no_tools(self):
        from mcp_cli.chat.system_prompt import _generate_dynamic_tools_prompt

        result = _generate_dynamic_tools_prompt(tools=None)
        assert "0 tools" in result
        assert "TOOL DISCOVERY SYSTEM" in result

    def test_direct_call_with_tools(self):
        from mcp_cli.chat.system_prompt import _generate_dynamic_tools_prompt

        result = _generate_dynamic_tools_prompt(tools=[1, 2])
        assert "2 tools" in result
