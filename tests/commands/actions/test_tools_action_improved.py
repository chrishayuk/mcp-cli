"""Improved tests for tools action with higher coverage."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mcp_cli.commands.actions.tools import (
    tools_action_async,
    _show_validation_info,
)
from mcp_cli.tools.models import ToolInfo


@pytest.fixture
def mock_context():
    """Create a mock application context."""
    context = MagicMock()
    context.tool_manager = MagicMock()
    return context


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager."""
    tm = MagicMock()
    tm.get_unique_tools = AsyncMock()
    tm.get_adapted_tools_for_llm = AsyncMock()
    tm.get_validation_summary = MagicMock(return_value={})
    tm.is_auto_fix_enabled = MagicMock(return_value=True)
    return tm


@pytest.fixture
def sample_tools():
    """Create sample tool data."""
    return [
        ToolInfo(
            name="test_tool",
            namespace="test_server",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        ),
        ToolInfo(
            name="another_tool",
            namespace="another_server",
            description="Another tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        ),
    ]


@pytest.fixture
def adapted_tools():
    """Create adapted tool definitions for LLM."""
    return [
        {
            "function": {
                "name": "test_server_test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        },
        {
            "function": {
                "name": "another_tool",
                "description": "Another tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                },
            }
        },
    ]


@pytest.mark.asyncio
async def test_tools_action_async_no_tool_manager(mock_context):
    """Test tools action when tool manager is not available."""
    mock_context.tool_manager = None

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        with patch("mcp_cli.commands.actions.tools.output") as mock_output:
            result = await tools_action_async()

            mock_output.error.assert_called_with("No tool manager available")
            assert result == []


@pytest.mark.asyncio
async def test_tools_action_async_basic(mock_context, mock_tool_manager, sample_tools):
    """Test basic tools action async."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output") as mock_output:
            with patch(
                "mcp_cli.commands.actions.tools.create_tools_table"
            ) as mock_create_table:
                mock_table = MagicMock()
                mock_create_table.return_value = mock_table

                result = await tools_action_async()

                # Verify output calls
                mock_output.info.assert_called_with(
                    "\nFetching tool catalogue from all servers…"
                )
                mock_output.print_table.assert_called_once_with(mock_table)
                mock_output.success.assert_called_with("Total tools available: 2")

                # Verify result structure
                assert len(result) == 2
                assert result[0]["name"] == "test_tool"
                assert result[0]["namespace"] == "test_server"
                assert result[1]["name"] == "another_tool"


@pytest.mark.asyncio
async def test_tools_action_async_with_validation(mock_context, mock_tool_manager):
    """Test tools action with validation flag."""
    mock_tool_manager.get_validation_summary.return_value = {
        "total_tools": 10,
        "valid_tools": 8,
        "invalid_tools": 2,
        "disabled_by_user": 1,
        "disabled_by_validation": 1,
        "validation_errors": [
            {"tool": "bad_tool", "error": "Invalid schema", "reason": "schema_error"}
        ],
        "disabled_tools": {"disabled_tool": "User disabled"},
    }

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output") as mock_output:
            with patch(
                "mcp_cli.commands.actions.tools.format_table"
            ) as mock_format_table:
                mock_format_table.return_value = "formatted_table"

                result = await tools_action_async(show_validation=True)

                mock_output.info.assert_any_call("Tool Validation Report for openai")
                mock_output.print_table.assert_called()
                mock_output.error.assert_called()  # For validation errors
                mock_output.warning.assert_called()  # For disabled tools

                assert result == [
                    {
                        "validation_summary": mock_tool_manager.get_validation_summary.return_value
                    }
                ]


@pytest.mark.asyncio
async def test_tools_action_async_adapted_tools(
    mock_context, mock_tool_manager, adapted_tools
):
    """Test tools action with adapted tools for LLM."""
    mock_tool_manager.get_adapted_tools_for_llm.return_value = (adapted_tools, None)
    mock_tool_manager.get_validation_summary.return_value = {
        "invalid_tools": 2,
        "total_tools": 4,
    }

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output") as mock_output:
            with patch(
                "mcp_cli.commands.actions.tools.create_tools_table"
            ) as mock_create_table:
                mock_table = MagicMock()
                mock_create_table.return_value = mock_table

                result = await tools_action_async()

                # Should call get_adapted_tools_for_llm
                mock_tool_manager.get_adapted_tools_for_llm.assert_called_with("openai")

                # Should show validation note
                assert any(
                    "2 tools filtered out" in str(call)
                    for call in mock_output.print.call_args_list
                )
                mock_output.hint.assert_called_with(
                    "Use --validation flag to see details"
                )

                assert len(result) == 2


@pytest.mark.asyncio
async def test_tools_action_async_adapted_tools_error_fallback(
    mock_context, mock_tool_manager, sample_tools
):
    """Test tools action falls back when adapted tools fail."""
    mock_tool_manager.get_adapted_tools_for_llm.side_effect = Exception("Adapter error")
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output"):
            with patch(
                "mcp_cli.commands.actions.tools.create_tools_table"
            ) as mock_create_table:
                mock_create_table.return_value = MagicMock()

                result = await tools_action_async()

                # Should fall back to get_unique_tools
                mock_tool_manager.get_unique_tools.assert_called_once()
                assert len(result) == 2


@pytest.mark.asyncio
async def test_tools_action_async_no_tools(mock_context, mock_tool_manager):
    """Test tools action when no tools available."""
    mock_tool_manager.get_unique_tools.return_value = []

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output") as mock_output:
            result = await tools_action_async()

            mock_output.info.assert_called_once()
            mock_output.warning.assert_called_with(
                "No tools available from any server."
            )
            assert result == []


@pytest.mark.asyncio
async def test_tools_action_async_raw_mode(
    mock_context, mock_tool_manager, sample_tools
):
    """Test tools action with raw JSON output."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output") as mock_output:
            await tools_action_async(show_raw=True)

            # Should call json output instead of table
            mock_output.json.assert_called_once()

            # Verify JSON structure was output
            call_args = mock_output.json.call_args[0][0]
            json_data = json.loads(call_args)
            assert len(json_data) == 2
            assert json_data[0]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_tools_action_async_show_details(
    mock_context, mock_tool_manager, sample_tools
):
    """Test tools action with show_details=True."""
    mock_tool_manager.get_unique_tools.return_value = sample_tools

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output"):
            with patch(
                "mcp_cli.commands.actions.tools.create_tools_table"
            ) as mock_create_table:
                mock_table = MagicMock()
                mock_create_table.return_value = mock_table

                await tools_action_async(show_details=True)

                # Verify create_tools_table was called with show_details=True
                mock_create_table.assert_called_once_with(
                    sample_tools, show_details=True
                )


@pytest.mark.asyncio
async def test_tools_action_async_with_provider(
    mock_context, mock_tool_manager, adapted_tools
):
    """Test tools action with specific provider."""
    mock_tool_manager.get_adapted_tools_for_llm.return_value = (adapted_tools, None)

    with patch("mcp_cli.commands.actions.tools.get_context", return_value=mock_context):
        mock_context.tool_manager = mock_tool_manager

        with patch("mcp_cli.commands.actions.tools.output"):
            with patch("mcp_cli.commands.actions.tools.create_tools_table"):
                await tools_action_async(provider="anthropic")

                # Should use specified provider
                mock_tool_manager.get_adapted_tools_for_llm.assert_called_with(
                    "anthropic"
                )


@pytest.mark.asyncio
async def test_show_validation_info_no_validation(mock_tool_manager):
    """Test validation info when validation not available."""
    # Remove validation methods
    delattr(mock_tool_manager, "get_validation_summary")

    with patch("mcp_cli.commands.actions.tools.output") as mock_output:
        result = await _show_validation_info(mock_tool_manager, "openai")

        mock_output.print.assert_called_with(
            "Validation not available - using basic ToolManager"
        )
        assert result == []


@pytest.mark.asyncio
async def test_show_validation_info_with_errors(mock_tool_manager):
    """Test validation info display with errors."""
    mock_tool_manager.get_validation_summary.return_value = {
        "total_tools": 15,
        "valid_tools": 10,
        "invalid_tools": 5,
        "disabled_by_user": 2,
        "disabled_by_validation": 3,
        "validation_errors": [
            {"tool": f"bad_tool_{i}", "error": f"Error {i}", "reason": f"reason_{i}"}
            for i in range(15)
        ],
        "disabled_tools": {"tool1": "User disabled", "tool2": "Validation failed"},
    }

    with patch("mcp_cli.commands.actions.tools.output") as mock_output:
        with patch("mcp_cli.commands.actions.tools.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            result = await _show_validation_info(mock_tool_manager, "openai")

            # Should show validation summary
            mock_output.info.assert_any_call("Tool Validation Report for openai")

            # Should show errors (first 10)
            mock_output.error.assert_called()

            # Should show "more errors" message
            assert any(
                "and 5 more errors" in str(call)
                for call in mock_output.info.call_args_list
            )

            # Should show disabled tools
            mock_output.warning.assert_called()

            # Should show auto-fix status
            assert any(
                "Auto-fix: Enabled" in str(call)
                for call in mock_output.info.call_args_list
            )

            # Should show commands
            assert mock_output.print.call_count >= 5  # Multiple command prints

            assert result == [
                {
                    "validation_summary": mock_tool_manager.get_validation_summary.return_value
                }
            ]


@pytest.mark.asyncio
async def test_show_validation_info_long_error_messages(mock_tool_manager):
    """Test validation info truncates long error messages."""
    long_error = "A" * 100
    mock_tool_manager.get_validation_summary.return_value = {
        "total_tools": 1,
        "valid_tools": 0,
        "invalid_tools": 1,
        "disabled_by_user": 0,
        "disabled_by_validation": 0,
        "validation_errors": [
            {"tool": "bad_tool", "error": long_error, "reason": "too_long"}
        ],
        "disabled_tools": {},
    }

    with patch("mcp_cli.commands.actions.tools.output"):
        with patch("mcp_cli.commands.actions.tools.format_table"):
            await _show_validation_info(mock_tool_manager, "openai")

            # Test passes
