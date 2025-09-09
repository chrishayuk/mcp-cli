"""Tests for tools manage action."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mcp_cli.commands.actions.tools_manage import (
    tools_manage_action_async,
    tools_manage_action,
)


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager."""
    tm = MagicMock()
    tm.enable_tool = MagicMock()
    tm.disable_tool = MagicMock()
    tm.validate_single_tool = AsyncMock()
    tm.revalidate_tools = AsyncMock()
    tm.get_validation_summary = MagicMock()
    tm.get_disabled_tools = MagicMock()
    tm.get_tool_validation_details = MagicMock()
    tm.set_auto_fix_enabled = MagicMock()
    tm.clear_validation_disabled_tools = MagicMock()
    return tm


@pytest.mark.asyncio
async def test_tools_manage_enable_action(mock_tool_manager):
    """Test enable action."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "enable", "test_tool"
        )

        mock_tool_manager.enable_tool.assert_called_once_with("test_tool")
        mock_output.success.assert_called_once_with("✓ Enabled tool: test_tool")

        assert result == {"success": True, "action": "enable", "tool": "test_tool"}


@pytest.mark.asyncio
async def test_tools_manage_enable_no_tool_name(mock_tool_manager):
    """Test enable action without tool name."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "enable")

        mock_output.error.assert_called_once_with(
            "Tool name required for enable action"
        )
        mock_tool_manager.enable_tool.assert_not_called()

        assert result == {"success": False, "error": "Tool name required"}


@pytest.mark.asyncio
async def test_tools_manage_disable_action(mock_tool_manager):
    """Test disable action."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "disable", "test_tool"
        )

        mock_tool_manager.disable_tool.assert_called_once_with(
            "test_tool", reason="user"
        )
        mock_output.warning.assert_called_once_with("✗ Disabled tool: test_tool")

        assert result == {"success": True, "action": "disable", "tool": "test_tool"}


@pytest.mark.asyncio
async def test_tools_manage_disable_no_tool_name(mock_tool_manager):
    """Test disable action without tool name."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "disable")

        mock_output.error.assert_called_once_with(
            "Tool name required for disable action"
        )
        mock_tool_manager.disable_tool.assert_not_called()

        assert result == {"success": False, "error": "Tool name required"}


@pytest.mark.asyncio
async def test_tools_manage_validate_single_tool_valid(mock_tool_manager):
    """Test validate action for single valid tool."""
    mock_tool_manager.validate_single_tool.return_value = (True, None)

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "validate", "test_tool"
        )

        mock_tool_manager.validate_single_tool.assert_called_once_with("test_tool")
        mock_output.success.assert_called_once_with("✓ Tool 'test_tool' is valid")

        assert result == {
            "success": True,
            "action": "validate",
            "tool": "test_tool",
            "is_valid": True,
            "error": None,
        }


@pytest.mark.asyncio
async def test_tools_manage_validate_single_tool_invalid(mock_tool_manager):
    """Test validate action for single invalid tool."""
    mock_tool_manager.validate_single_tool.return_value = (False, "Invalid schema")

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "validate", "test_tool"
        )

        mock_tool_manager.validate_single_tool.assert_called_once_with("test_tool")
        mock_output.error.assert_called_once_with(
            "✗ Tool 'test_tool' is invalid: Invalid schema"
        )

        assert result == {
            "success": True,
            "action": "validate",
            "tool": "test_tool",
            "is_valid": False,
            "error": "Invalid schema",
        }


@pytest.mark.asyncio
async def test_tools_manage_validate_all_tools(mock_tool_manager):
    """Test validate action for all tools."""
    summary = {"total_tools": 5, "valid_tools": 4, "invalid_tools": 1}
    mock_tool_manager.revalidate_tools.return_value = summary

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "validate", provider="anthropic"
        )

        mock_tool_manager.revalidate_tools.assert_called_once_with("anthropic")
        mock_output.info.assert_called_once_with(
            "Validating all tools for anthropic..."
        )
        mock_output.success.assert_called_once_with("Validation complete:")

        assert result == {"success": True, "action": "validate_all", "summary": summary}


@pytest.mark.asyncio
async def test_tools_manage_status_action(mock_tool_manager):
    """Test status action."""
    summary = {
        "total_tools": 10,
        "valid_tools": 8,
        "invalid_tools": 2,
        "disabled_by_user": 1,
        "disabled_by_validation": 1,
        "auto_fix_enabled": True,
        "provider": "openai",
    }
    mock_tool_manager.get_validation_summary.return_value = summary

    with (
        patch("mcp_cli.commands.actions.tools_manage.output") as mock_output,
        patch(
            "mcp_cli.commands.actions.tools_manage.format_table"
        ) as mock_format_table,
    ):
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        result = await tools_manage_action_async(mock_tool_manager, "status")

        mock_output.print_table.assert_called_once_with(mock_table)

        assert result == {"success": True, "action": "status", "summary": summary}


@pytest.mark.asyncio
async def test_tools_manage_list_disabled_with_tools(mock_tool_manager):
    """Test list-disabled action with disabled tools."""
    disabled_tools = {"tool1": "user", "tool2": "validation"}
    mock_tool_manager.get_disabled_tools.return_value = disabled_tools

    with (
        patch("mcp_cli.commands.actions.tools_manage.output") as mock_output,
        patch(
            "mcp_cli.commands.actions.tools_manage.format_table"
        ) as mock_format_table,
    ):
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        result = await tools_manage_action_async(mock_tool_manager, "list-disabled")

        mock_output.print_table.assert_called_once_with(mock_table)

        # Verify table data
        table_data = mock_format_table.call_args[0][0]
        assert len(table_data) == 2
        assert {"Tool Name": "tool1", "Reason": "user"} in table_data
        assert {"Tool Name": "tool2", "Reason": "validation"} in table_data

        assert result == {
            "success": True,
            "action": "list_disabled",
            "disabled_tools": disabled_tools,
        }


@pytest.mark.asyncio
async def test_tools_manage_list_disabled_no_tools(mock_tool_manager):
    """Test list-disabled action with no disabled tools."""
    mock_tool_manager.get_disabled_tools.return_value = {}

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "list-disabled")

        mock_output.success.assert_called_once_with("No disabled tools")

        assert result == {
            "success": True,
            "action": "list_disabled",
            "disabled_tools": {},
        }


@pytest.mark.asyncio
async def test_tools_manage_details_action(mock_tool_manager):
    """Test details action."""
    details = {
        "is_enabled": True,
        "disabled_reason": None,
        "validation_error": None,
        "can_auto_fix": False,
    }
    mock_tool_manager.get_tool_validation_details.return_value = details

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "details", "test_tool"
        )

        mock_tool_manager.get_tool_validation_details.assert_called_once_with(
            "test_tool"
        )
        mock_output.panel.assert_called_once()

        assert result == {
            "success": True,
            "action": "details",
            "tool": "test_tool",
            "details": details,
        }


@pytest.mark.asyncio
async def test_tools_manage_details_no_tool_name(mock_tool_manager):
    """Test details action without tool name."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "details")

        mock_output.error.assert_called_once_with(
            "Tool name required for details action"
        )

        assert result == {"success": False, "error": "Tool name required"}


@pytest.mark.asyncio
async def test_tools_manage_details_tool_not_found(mock_tool_manager):
    """Test details action with non-existent tool."""
    mock_tool_manager.get_tool_validation_details.return_value = None

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "details", "nonexistent"
        )

        mock_output.error.assert_called_once_with("Tool 'nonexistent' not found")

        assert result == {"success": False, "error": "Tool not found"}


@pytest.mark.asyncio
async def test_tools_manage_auto_fix_enable(mock_tool_manager):
    """Test auto-fix enable action."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "auto-fix", enabled=True
        )

        mock_tool_manager.set_auto_fix_enabled.assert_called_once_with(True)
        mock_output.info.assert_called_once_with("Auto-fix enabled")

        assert result == {"success": True, "action": "auto_fix", "enabled": True}


@pytest.mark.asyncio
async def test_tools_manage_auto_fix_disable(mock_tool_manager):
    """Test auto-fix disable action."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(
            mock_tool_manager, "auto-fix", enabled=False
        )

        mock_tool_manager.set_auto_fix_enabled.assert_called_once_with(False)
        mock_output.info.assert_called_once_with("Auto-fix disabled")

        assert result == {"success": True, "action": "auto_fix", "enabled": False}


@pytest.mark.asyncio
async def test_tools_manage_clear_validation(mock_tool_manager):
    """Test clear-validation action."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "clear-validation")

        mock_tool_manager.clear_validation_disabled_tools.assert_called_once()
        mock_output.success.assert_called_once_with(
            "Cleared all validation-disabled tools"
        )

        assert result == {"success": True, "action": "clear_validation"}


@pytest.mark.asyncio
async def test_tools_manage_validation_errors_with_errors(mock_tool_manager):
    """Test validation-errors action with errors."""
    errors = [
        {"tool": "tool1", "error": "Invalid schema"},
        {"tool": "tool2", "error": "Missing parameter"},
    ]
    summary = {"validation_errors": errors}
    mock_tool_manager.get_validation_summary.return_value = summary

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "validation-errors")

        mock_output.error.assert_called_once_with("Found 2 validation errors:")

        assert result == {
            "success": True,
            "action": "validation_errors",
            "errors": errors,
        }


@pytest.mark.asyncio
async def test_tools_manage_validation_errors_no_errors(mock_tool_manager):
    """Test validation-errors action with no errors."""
    summary = {"validation_errors": []}
    mock_tool_manager.get_validation_summary.return_value = summary

    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "validation-errors")

        mock_output.success.assert_called_once_with("No validation errors")

        assert result == {"success": True, "action": "validation_errors", "errors": []}


@pytest.mark.asyncio
async def test_tools_manage_unknown_action(mock_tool_manager):
    """Test unknown action."""
    with patch("mcp_cli.commands.actions.tools_manage.output") as mock_output:
        result = await tools_manage_action_async(mock_tool_manager, "unknown")

        mock_output.error.assert_called_once_with("Unknown action: unknown")

        assert result == {"success": False, "error": "Unknown action: unknown"}


def test_tools_manage_action_sync_wrapper(mock_tool_manager):
    """Test the sync wrapper."""
    with patch("mcp_cli.commands.actions.tools_manage.asyncio.run") as mock_run:
        mock_run.return_value = {"success": True}

        result = tools_manage_action(mock_tool_manager, "status", "tool")

        mock_run.assert_called_once()
        assert result == {"success": True}
