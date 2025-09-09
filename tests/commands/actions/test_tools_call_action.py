"""Tests for the tools_call action command."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.actions.tools_call import tools_call_action
from mcp_cli.tools.models import ToolInfo, ToolCallResult


@pytest.fixture
def mock_context():
    """Create a mock application context."""
    context = MagicMock()
    context.tool_manager = MagicMock()
    return context


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager."""
    tm = MagicMock()
    tm.get_unique_tools = AsyncMock()
    tm.execute_tool = AsyncMock()
    return tm


@pytest.fixture
def sample_tools():
    """Create sample tool data."""
    return [
        ToolInfo(
            name="test_tool",
            namespace="test_server",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg1": {"type": "string"}}},
        ),
        ToolInfo(
            name="no_args_tool",
            namespace="test_server",
            description="Tool with no arguments",
            parameters={"type": "object"},
        ),
        ToolInfo(
            name="no_desc_tool",
            namespace="other_server",
            description=None,
            parameters={"type": "object", "properties": {}},
        ),
    ]


@pytest.mark.asyncio
async def test_tools_call_action_no_tool_manager(mock_context):
    """Test tools call when tool manager is not available."""
    mock_context.tool_manager = None

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            await tools_call_action()

            mock_output.print.assert_any_call(
                "[red]Error:[/red] No tool manager available"
            )


@pytest.mark.asyncio
async def test_tools_call_action_no_tools(mock_context, mock_tool_manager):
    """Test tools call when no tools are available."""
    mock_tool_manager.get_unique_tools.return_value = []

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            await tools_call_action()

            mock_output.print.assert_any_call(
                "[yellow]No tools available from any server.[/yellow]"
            )


@pytest.mark.asyncio
async def test_tools_call_action_successful_call_with_args(
    mock_context, mock_tool_manager, sample_tools
):
    """Test successful tool call with arguments."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools
    mock_result = ToolCallResult(
        tool_name="test_tool",
        success=True,
        result={"data": "test_result"},
    )
    mock_result.duration_ms = 100
    mock_tool_manager.execute_tool.return_value = mock_result

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Simulate user selecting tool 1 and providing JSON arguments
                mock_to_thread.side_effect = ["1", '{"arg1": "value1"}']

                with patch(
                    "mcp_cli.commands.actions.tools_call.display_tool_call_result"
                ) as mock_display:
                    await tools_call_action()

                    # Verify tool list was displayed
                    mock_output.print.assert_any_call("[green]Available tools:[/green]")

                    # Verify tool was executed with correct arguments
                    mock_tool_manager.execute_tool.assert_called_once_with(
                        "test_server.test_tool", {"arg1": "value1"}
                    )

                    # Verify result was displayed
                    mock_display.assert_called_once_with(mock_result)


@pytest.mark.asyncio
async def test_tools_call_action_tool_no_args(
    mock_context, mock_tool_manager, sample_tools
):
    """Test tool call for tool with no arguments."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools
    mock_result = ToolCallResult(
        tool_name="no_args_tool",
        success=True,
        result={"data": "success"},
    )
    mock_result.duration_ms = 100
    mock_tool_manager.execute_tool.return_value = mock_result

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Select tool 2 (no_args_tool)
                mock_to_thread.return_value = "2"

                with patch(
                    "mcp_cli.commands.actions.tools_call.display_tool_call_result"
                ):
                    await tools_call_action()

                    # Should show "Tool takes no arguments"
                    mock_output.print.assert_any_call(
                        "[dim]Tool takes no arguments.[/dim]"
                    )

                    # Should execute with empty args
                    mock_tool_manager.execute_tool.assert_called_once_with(
                        "test_server.no_args_tool", {}
                    )


@pytest.mark.asyncio
async def test_tools_call_action_invalid_selection(
    mock_context, mock_tool_manager, sample_tools
):
    """Test invalid tool selection."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Invalid selection
                mock_to_thread.return_value = "99"

                await tools_call_action()

                mock_output.print.assert_any_call("[red]Invalid selection.[/red]")
                mock_tool_manager.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_tools_call_action_non_numeric_selection(
    mock_context, mock_tool_manager, sample_tools
):
    """Test non-numeric tool selection."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Non-numeric selection
                mock_to_thread.return_value = "abc"

                await tools_call_action()

                mock_output.print.assert_any_call("[red]Invalid selection.[/red]")
                mock_tool_manager.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_tools_call_action_invalid_json_args(
    mock_context, mock_tool_manager, sample_tools
):
    """Test invalid JSON arguments."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Select tool 1, provide invalid JSON
                mock_to_thread.side_effect = ["1", "not valid json"]

                await tools_call_action()

                mock_output.print.assert_any_call("[red]Invalid JSON - aborting.[/red]")
                mock_tool_manager.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_tools_call_action_empty_args_input(
    mock_context, mock_tool_manager, sample_tools
):
    """Test empty arguments input (should use empty dict)."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools
    mock_result = ToolCallResult(
        tool_name="test_tool",
        success=True,
        result={"data": "success"},
    )
    mock_result.duration_ms = 100
    mock_tool_manager.execute_tool.return_value = mock_result

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output"):
            with patch("asyncio.to_thread") as mock_to_thread:
                # Select tool 1, provide empty string for args
                mock_to_thread.side_effect = ["1", "  "]

                with patch(
                    "mcp_cli.commands.actions.tools_call.display_tool_call_result"
                ):
                    await tools_call_action()

                    # Should execute with empty args
                    mock_tool_manager.execute_tool.assert_called_once_with(
                        "test_server.test_tool", {}
                    )


@pytest.mark.asyncio
async def test_tools_call_action_execution_error(
    mock_context, mock_tool_manager, sample_tools
):
    """Test error during tool execution."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools
    mock_tool_manager.execute_tool.side_effect = Exception("Execution failed")

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Select tool 1
                mock_to_thread.side_effect = ["1", ""]

                await tools_call_action()

                mock_output.print.assert_any_call("[red]Error: Execution failed[/red]")


@pytest.mark.asyncio
async def test_tools_call_action_tool_no_description(
    mock_context, mock_tool_manager, sample_tools
):
    """Test displaying tool with no description."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Select tool 3 (no_desc_tool) then cancel
                mock_to_thread.return_value = "99"  # Invalid to exit

                await tools_call_action()

                # Should show "No description" for tool 3
                calls = [str(call) for call in mock_output.print.call_args_list]
                assert any(
                    "no_desc_tool" in str(call) and "No description" in str(call)
                    for call in calls
                )


@pytest.mark.asyncio
async def test_tools_call_action_displays_selected_tool_info(
    mock_context, mock_tool_manager, sample_tools
):
    """Test that selected tool information is displayed."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools
    mock_result = ToolCallResult(
        tool_name="test_tool",
        success=True,
        result={"data": "success"},
    )
    mock_result.duration_ms = 100
    mock_tool_manager.execute_tool.return_value = mock_result

    with patch(
        "mcp_cli.commands.actions.tools_call.get_context", return_value=mock_context
    ):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools_call.output") as mock_output:
            with patch("asyncio.to_thread") as mock_to_thread:
                # Select tool 1
                mock_to_thread.side_effect = ["1", ""]

                with patch(
                    "mcp_cli.commands.actions.tools_call.display_tool_call_result"
                ):
                    await tools_call_action()

                    # Should display selected tool info
                    # Check that the tool info was displayed (exact format may vary)
                    assert any(
                        "test_tool" in str(call)
                        for call in mock_output.print.call_args_list
                    )
                    assert any(
                        "test_server" in str(call) or "A test tool" in str(call)
                        for call in mock_output.print.call_args_list
                    )
