# mcp_cli/chat/commands/tools_manage.py
"""
Chat commands for tool management.
"""

from __future__ import annotations

from typing import Any, Dict, List

from chuk_term.ui import output
from mcp_cli.chat.commands import register_command


async def tools_enable_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Enable a disabled tool."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "enable_tool"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    if len(parts) < 2:
        output.print("[red]Error:[/red] Tool name required")
        return True

    tool_name = parts[1]

    try:
        tm.enable_tool(tool_name)
        output.print(f"[green]✓ Enabled tool:[/green] {tool_name}")

        chat_ctx = ctx.get("chat_context")
        if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
            await chat_ctx.refresh_after_model_change()

    except Exception as e:
        output.print(f"[red]Error enabling tool:[/red] {e}")

    return True


async def tools_disable_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Disable a tool."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "disable_tool"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    if len(parts) < 2:
        output.print("[red]Error:[/red] Tool name required")
        return True

    tool_name = parts[1]

    try:
        tm.disable_tool(tool_name, reason="user")
        output.print(f"[yellow]✗ Disabled tool:[/yellow] {tool_name}")

        chat_ctx = ctx.get("chat_context")
        if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
            await chat_ctx.refresh_after_model_change()

    except Exception as e:
        output.print(f"[red]Error disabling tool:[/red] {e}")

    return True


async def tools_validate_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Validate tool schemas."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "validate_single_tool"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    tool_name = parts[1] if len(parts) > 1 else None
    provider = ctx.get("provider", "openai")

    try:
        if tool_name:
            is_valid, error_msg = await tm.validate_single_tool(tool_name, provider)
            if is_valid:
                output.print(f"[green]✓ Tool '{tool_name}' is valid[/green]")
            else:
                output.print(f"[red]✗ Tool '{tool_name}' is invalid:[/red] {error_msg}")
        else:
            output.print(f"[cyan]Validating all tools for {provider}...[/cyan]")

            summary = await tm.revalidate_tools(provider)
            output.print("[green]Validation complete:[/green]")
            output.print(f"  • Total tools: {summary.get('total_tools', 0)}")
            output.print(f"  • Valid: {summary.get('valid_tools', 0)}")
            output.print(f"  • Invalid: {summary.get('invalid_tools', 0)}")

            chat_ctx = ctx.get("chat_context")
            if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
                await chat_ctx.refresh_after_model_change()

    except Exception as e:
        output.print(f"[red]Error during validation:[/red] {e}")

    return True


async def tools_status_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Show tool management status."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "get_validation_summary"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    try:
        summary = tm.get_validation_summary()

        from rich.table import Table

        table = Table(title="Tool Management Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

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
        output.print(f"[red]Error getting status:[/red] {e}")

    return True


async def tools_disabled_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """List disabled tools."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "get_disabled_tools"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    try:
        disabled_tools = tm.get_disabled_tools()

        if not disabled_tools:
            output.print("[green]No disabled tools[/green]")
        else:
            from rich.table import Table

            table = Table(title="Disabled Tools")
            table.add_column("Tool Name", style="yellow")
            table.add_column("Reason", style="red")

            for tool, reason in disabled_tools.items():
                table.add_row(tool, reason)

            output.print(table)

    except Exception as e:
        output.print(f"[red]Error listing disabled tools:[/red] {e}")

    return True


async def tools_details_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Show tool validation details."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "get_tool_validation_details"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    if len(parts) < 2:
        output.print("[red]Error:[/red] Tool name required")
        return True

    tool_name = parts[1]

    try:
        details = tm.get_tool_validation_details(tool_name)
        if not details:
            output.print(f"[red]Tool '{tool_name}' not found[/red]")
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
        output.print(f"[red]Error getting tool details:[/red] {e}")

    return True


async def tools_autofix_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Enable/disable auto-fix."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "set_auto_fix_enabled"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    if len(parts) > 1:
        setting = parts[1].lower() in ("on", "enable", "true", "yes")
        tm.set_auto_fix_enabled(setting)
        status = "enabled" if setting else "disabled"
        output.print(f"[cyan]Auto-fix {status}[/cyan]")
    else:
        current = tm.is_auto_fix_enabled()
        output.print(
            f"[cyan]Auto-fix is currently {'enabled' if current else 'disabled'}[/cyan]"
        )

    return True


async def tools_clear_validation_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Clear validation-disabled tools."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "clear_validation_disabled_tools"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    try:
        tm.clear_validation_disabled_tools()
        output.print("[green]Cleared all validation-disabled tools[/green]")

        chat_ctx = ctx.get("chat_context")
        if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
            await chat_ctx.refresh_after_model_change()

    except Exception as e:
        output.print(f"[red]Error clearing validation:[/red] {e}")

    return True


async def tools_errors_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Show validation errors."""

    tm = ctx.get("tool_manager")
    if not tm or not hasattr(tm, "get_validation_summary"):
        output.print("[red]Error:[/red] Enhanced ToolManager not available.")
        return True

    try:
        summary = tm.get_validation_summary()
        errors = summary.get("validation_errors", [])

        if not errors:
            output.print("[green]No validation errors[/green]")
        else:
            output.print(f"[red]Found {len(errors)} validation errors:[/red]")
            for error in errors[:10]:
                output.print(f"  • {error['tool']}: {error['error']}")
            if len(errors) > 10:
                output.print(f"  ... and {len(errors) - 10} more errors")

    except Exception as e:
        output.print(f"[red]Error getting validation errors:[/red] {e}")

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
