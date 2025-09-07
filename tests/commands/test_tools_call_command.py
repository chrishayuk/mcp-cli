# commands/test_tools_call_command.py
import pytest
from unittest.mock import patch, MagicMock
import json

from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.models import ToolInfo, ToolCallResult
from tests.conftest import setup_test_context


class DummyTMWithTools:
    def __init__(self, tools):
        self._tools = tools

    async def get_unique_tools(self):
        return self._tools

    async def execute_tool(self, tool_name, arguments):
        # Mock successful execution
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result={"status": "success", "data": arguments},
            error=None,
        )


class DummyTMNoTools:
    async def get_unique_tools(self):
        return []


@pytest.mark.asyncio
async def test_tools_call_no_tools():
    """Test tools_call_action when no tools are available."""
    tm = DummyTMNoTools()
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.tools_call.output") as mock_output:
        mock_output.print = MagicMock()

        await tools_call_action()

        # Should print message about no tools
        mock_output.print.assert_called()
        calls = mock_output.print.call_args_list
        assert any("No tools available" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_tools_call_with_tools_user_cancels():
    """Test when user cancels tool selection."""
    tools = [
        ToolInfo(
            name="test_tool",
            namespace="test",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            is_async=True,
            tags=[],
        )
    ]
    tm = DummyTMWithTools(tools)
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.tools_call.output") as mock_output:
        mock_output.print = MagicMock()

        # Mock user input to cancel
        with patch("builtins.input", side_effect=["q"]):
            await tools_call_action()

        # Should print the tools list
        mock_output.print.assert_called()
        calls = str(mock_output.print.call_args_list)
        assert "test_tool" in calls


@pytest.mark.asyncio
async def test_tools_call_successful_execution():
    """Test successful tool execution."""
    tools = [
        ToolInfo(
            name="echo",
            namespace="test",
            description="Echo tool",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            is_async=True,
            tags=[],
        )
    ]
    tm = DummyTMWithTools(tools)
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    with patch("mcp_cli.commands.tools_call.output") as mock_output:
        mock_output.print = MagicMock()

        # Mock user input to select tool and provide arguments
        test_args = json.dumps({"message": "hello"})
        with patch("builtins.input", side_effect=["1", test_args]):
            with patch(
                "mcp_cli.commands.tools_call.display_tool_call_result"
            ) as mock_display:
                await tools_call_action()

                # Should display the result
                mock_display.assert_called_once()

                # Check that the tool was called with correct arguments
                result = mock_display.call_args[0][0]
                assert result.tool_name == "echo"
                assert not result.is_error
