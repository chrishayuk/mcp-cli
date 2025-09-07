# mcp_cli/chat/commands/tools_manage.py
"""
Chat commands for tool management.
"""

from __future__ import annotations

from typing import List

from chuk_term.ui import output
from mcp_cli.chat.commands import register_command
from mcp_cli.context import get_context


async def tools_enable_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Enable a disabled tool."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "enable_tool"):
        output.error("Enhanced ToolManager not available.")
        return True

    if len(parts) < 2:
        output.error("Tool name required")
        return True

    tool_name = parts[1]

    try:
        tm.enable_tool(tool_name)
        output.success(f"✓ Enabled tool: {tool_name}")

        # Refresh chat context if needed
        # Note: chat_context refresh might be handled elsewhere

    except Exception as e:
        output.error(f"Error enabling tool: {e}")

    return True


async def tools_disable_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Disable a tool."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "disable_tool"):
        output.error("Enhanced ToolManager not available.")
        return True

    if len(parts) < 2:
        output.error("Tool name required")
        return True

    tool_name = parts[1]

    try:
        tm.disable_tool(tool_name, reason="user")
        output.warning(f"✗ Disabled tool: {tool_name}")

        # Refresh chat context if needed
        # Note: chat_context refresh might be handled elsewhere

    except Exception as e:
        output.error(f"Error disabling tool: {e}")

    return True


async def tools_validate_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Validate tool schemas."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "validate_single_tool"):
        output.error("Enhanced ToolManager not available.")
        return True

    tool_name = parts[1] if len(parts) > 1 else None
    provider = context.provider

    try:
        if tool_name:
            is_valid, error_msg = await tm.validate_single_tool(tool_name, provider)
            if is_valid:
                output.success(f"✓ Tool '{tool_name}' is valid")
            else:
                output.error(f"✗ Tool '{tool_name}' is invalid: {error_msg}")
        else:
            output.info(f"Validating all tools for {provider}...")

            summary = await tm.revalidate_tools(provider)
            output.success("Validation complete:")
            output.print(f"  • Total tools: {summary.get('total_tools', 0)}")
            output.print(f"  • Valid: {summary.get('valid_tools', 0)}")
            output.print(f"  • Invalid: {summary.get('invalid_tools', 0)}")

            # Note: In the future, we might need to refresh tool context
            # For now, validation is complete

    except Exception as e:
        output.error(f"Error during validation: {e}")

    return True


async def tools_status_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Show tool management status."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "get_validation_summary"):
        output.error("Enhanced ToolManager not available.")
        return True

    try:
        summary = tm.get_validation_summary()

        from rich.table import Table

        table = Table(title="Tool Management Status")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Total Tools", str(summary.get("total_tools", "Unknown")))
        table.add_row("Valid Tools", str(summary.get("valid_tools", "Unknown")))
        table.add_row("Invalid Tools", str(summary.get("invalid_tools", "Unknown")))
        table.add_row("Disabled by User", str(summary.get("disabled_by_user", 0)))
        table.add_row(
            "Disabled by Validation", str(summary.get("disabled_by_validation", 0))
        )
        table.add_row(
            "Auto-fix Enabled",
            "Yes" if summary.get("auto_fix_enabled", False) else "No",
        )
        table.add_row("Last Provider", str(summary.get("provider", "None")))

        output.print(table)

    except Exception as e:
        output.error(f"Error getting status: {e}")

    return True


async def tools_disabled_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """List disabled tools."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "get_disabled_tools"):
        output.error("Enhanced ToolManager not available.")
        return True

    try:
        disabled_tools = tm.get_disabled_tools()

        if not disabled_tools:
            output.success("No disabled tools")
        else:
            from rich.table import Table

            table = Table(title="Disabled Tools")
            table.add_column("Tool Name")
            table.add_column("Reason")

            for tool, reason in disabled_tools.items():
                table.add_row(tool, reason)

            output.print(table)

    except Exception as e:
        output.error(f"Error listing disabled tools: {e}")

    return True


async def tools_details_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Show tool validation details."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "get_tool_validation_details"):
        output.error("Enhanced ToolManager not available.")
        return True

    if len(parts) < 2:
        output.error("Tool name required")
        return True

    tool_name = parts[1]

    try:
        details = tm.get_tool_validation_details(tool_name)
        if not details:
            output.error(f"Tool '{tool_name}' not found")
            return True

        from rich.panel import Panel

        status = (
            "Enabled"
            if details["is_enabled"]
            else f"Disabled ({details['disabled_reason']})"
        )
        content = f"Status: {status}\n"

        if details["validation_error"]:
            content += f"Validation Error: {details['validation_error']}\n"

        if details["can_auto_fix"]:
            content += "Auto-fix: Available\n"

        output.print(Panel(content, title=f"Tool Details: {tool_name}"))

    except Exception as e:
        output.error(f"Error getting tool details: {e}")

    return True


async def tools_autofix_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Enable/disable auto-fix."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "set_auto_fix_enabled"):
        output.error("Enhanced ToolManager not available.")
        return True

    if len(parts) > 1:
        setting = parts[1].lower() in ("on", "enable", "true", "yes")
        tm.set_auto_fix_enabled(setting)
        status = "enabled" if setting else "disabled"
        output.info(f"Auto-fix {status}")
    else:
        current = tm.is_auto_fix_enabled()
        output.info(f"Auto-fix is currently {'enabled' if current else 'disabled'}")

    return True


async def tools_clear_validation_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Clear validation-disabled tools."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "clear_validation_disabled_tools"):
        output.error("Enhanced ToolManager not available.")
        return True

    try:
        tm.clear_validation_disabled_tools()
        output.success("Cleared all validation-disabled tools")

        # Refresh chat context if needed
        # Note: chat_context refresh might be handled elsewhere

    except Exception as e:
        output.error(f"Error clearing validation: {e}")

    return True


async def tools_errors_command(parts: List[str]) -> bool:
    # Use global context manager
    context = get_context()

    """Show validation errors."""

    tm = context.tool_manager
    if not tm or not hasattr(tm, "get_validation_summary"):
        output.error("Enhanced ToolManager not available.")
        return True

    try:
        summary = tm.get_validation_summary()
        errors = summary.get("validation_errors", [])

        if not errors:
            output.success("No validation errors")
        else:
            output.error(f"Found {len(errors)} validation errors:")
            for error in errors[:10]:
                output.print(f"  • {error['tool']}: {error['error']}")
            if len(errors) > 10:
                output.print(f"  ... and {len(errors) - 10} more errors")

    except Exception as e:
        output.error(f"Error getting validation errors: {e}")

    return True


# Register all tool management commands
register_command("/tools-enable", tools_enable_command)
register_command("/tools-disable", tools_disable_command)
register_command("/tools-validate", tools_validate_command)
register_command("/tools-status", tools_status_command)
register_command("/tools-disabled", tools_disabled_command)
register_command("/tools-details", tools_details_command)
register_command("/tools-autofix", tools_autofix_command)
register_command("/tools-clear-validation", tools_clear_validation_command)
register_command("/tools-errors", tools_errors_command)
