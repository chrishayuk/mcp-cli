# mcp_cli/chat/commands/tools.py
from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output

# Shared helpers
from mcp_cli.commands.tools import tools_action_async
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# ENHANCED: Tool management commands
# ════════════════════════════════════════════════════════════════════════════
async def tools_manage_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Manage tools (enable/disable/validate).

    Usage:
    /tools-enable <tool_name>           - Enable a disabled tool
    /tools-disable <tool_name>          - Disable a tool
    /tools-validate [tool_name]         - Validate tool(s)
    /tools-status                       - Show tool management status
    /tools-disabled                     - List disabled tools
    /tools-details <tool_name>          - Show tool validation details
    /tools-autofix <on|off>            - Enable/disable auto-fix
    /tools-clear-validation            - Clear validation-disabled tools
    /tools-errors                      - Show validation errors
    """

    tm: ToolManager = ctx.get("tool_manager")
    if not tm:
        output.print("[red]Error:[/red] ToolManager not available.")
        return True

    # Check if enhanced features are available
    if not hasattr(tm, "disable_tool"):
        output.print("[red]Error:[/red] Tool management requires enhanced ToolManager.")
        output.print(
            "[dim]Note: Your ToolManager doesn't support validation features[/dim]"
        )
        return True

    if len(parts) < 1:
        output.print("[red]Error:[/red] No action specified")
        return True

    command = parts[0]  # e.g., "/tools-enable"
    args = parts[1:] if len(parts) > 1 else []

    # Parse command
    if command == "/tools-enable":
        if not args:
            output.print("[red]Error:[/red] Tool name required")
            return True

        tool_name = args[0]
        tm.enable_tool(tool_name)
        output.print(f"[green]✓ Enabled tool:[/green] {tool_name}")

        # Refresh chat context if available
        chat_ctx = ctx.get("chat_context")
        if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
            await chat_ctx.refresh_after_model_change()

    elif command == "/tools-disable":
        if not args:
            output.print("[red]Error:[/red] Tool name required")
            return True

        tool_name = args[0]
        tm.disable_tool(tool_name, reason="user")
        output.print(f"[yellow]✗ Disabled tool:[/yellow] {tool_name}")

        # Refresh chat context if available
        chat_ctx = ctx.get("chat_context")
        if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
            await chat_ctx.refresh_after_model_change()

    elif command == "/tools-validate":
        if args:
            # Validate single tool
            tool_name = args[0]
            try:
                is_valid, error_msg = await tm.validate_single_tool(tool_name)
                if is_valid:
                    output.print(f"[green]✓ Tool '{tool_name}' is valid[/green]")
                else:
                    output.print(
                        f"[red]✗ Tool '{tool_name}' is invalid:[/red] {error_msg}"
                    )
            except Exception as e:
                output.print(f"[red]Error validating tool:[/red] {e}")
        else:
            # Validate all tools
            provider = ctx.get("provider", "openai")
            output.print(f"[cyan]Validating all tools for {provider}...[/cyan]")

            try:
                summary = await tm.revalidate_tools(provider)
                output.print("[green]Validation complete:[/green]")
                output.print(f"  • Total tools: {summary.get('total_tools', 0)}")
                output.print(f"  • Valid: {summary.get('valid_tools', 0)}")
                output.print(f"  • Invalid: {summary.get('invalid_tools', 0)}")

                # Refresh chat context
                chat_ctx = ctx.get("chat_context")
                if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
                    await chat_ctx.refresh_after_model_change()

            except Exception as e:
                output.print(f"[red]Error during validation:[/red] {e}")

    elif command == "/tools-status":
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

    elif command == "/tools-disabled":
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

    elif command == "/tools-details":
        if not args:
            output.print("[red]Error:[/red] Tool name required")
            return True

        tool_name = args[0]
        details = tm.get_tool_validation_details(tool_name)
        if not details:
            output.print(f"[red]Tool '{tool_name}' not found[/red]")
            return True

        # Display details panel
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

    elif command == "/tools-autofix":
        if args:
            setting = args[0].lower() in ("on", "enable", "true", "yes")
            tm.set_auto_fix_enabled(setting)
            status = "enabled" if setting else "disabled"
            output.print(f"[cyan]Auto-fix {status}[/cyan]")
        else:
            current = tm.is_auto_fix_enabled()
            output.print(
                f"[cyan]Auto-fix is currently {'enabled' if current else 'disabled'}[/cyan]"
            )

    elif command == "/tools-clear-validation":
        tm.clear_validation_disabled_tools()
        output.print("[green]Cleared all validation-disabled tools[/green]")

        # Refresh chat context
        chat_ctx = ctx.get("chat_context")
        if chat_ctx and hasattr(chat_ctx, "refresh_after_model_change"):
            await chat_ctx.refresh_after_model_change()

    elif command == "/tools-errors":
        summary = tm.get_validation_summary()
        errors = summary.get("validation_errors", [])

        if not errors:
            output.print("[green]No validation errors[/green]")
        else:
            output.print(f"[red]Found {len(errors)} validation errors:[/red]")
            for error in errors[:10]:  # Show first 10
                output.print(f"  • {error['tool']}: {error['error']}")
            if len(errors) > 10:
                output.print(f"  ... and {len(errors) - 10} more errors")

    else:
        output.print(f"[red]Unknown tool management command: {command}[/red]")
        output.print(
            "[dim]Available commands: /tools-enable, /tools-disable, /tools-validate, /tools-status, /tools-disabled, /tools-details, /tools-autofix, /tools-clear-validation, /tools-errors[/dim]"
        )

    return True


# ════════════════════════════════════════════════════════════════════════════
# Command handler (ENHANCED)
# ════════════════════════════════════════════════════════════════════════════
async def tools_command(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    List available tools (or call one interactively).

    This chat-command shows every server-side tool exposed by the connected
    MCP servers and can also launch a mini-wizard that walks you through
    executing a tool with JSON arguments.

    ENHANCED: Now supports validation flags and filtering.

    Usage
    -----
    /tools              - list tools
    /tools --all        - include parameter schemas
    /tools --raw        - dump raw JSON definitions
    /tools --validation - show validation report
    /tools call         - interactive "call tool" helper
    /t                  - short alias
    """

    tm: ToolManager | None = ctx.get("tool_manager")
    if tm is None:
        output.print("[red]Error:[/red] ToolManager not available.")
        return True  # command handled

    args = parts[1:]  # drop the command itself

    # ── Interactive call helper ────────────────────────────────────────────
    if args and args[0].lower() == "call":
        await tools_call_action(tm)
        return True

    # ── Tool listing with enhanced options ─────────────────────────────────
    show_details = "--all" in args
    show_raw = "--raw" in args
    show_validation = "--validation" in args
    provider = ctx.get("provider", "openai")

    await tools_action_async(
        tm,
        show_details=show_details,
        show_raw=show_raw,
        show_validation=show_validation,
        provider=provider,
    )
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration (ENHANCED)
# ════════════════════════════════════════════════════════════════════════════
register_command("/tools", tools_command)

# Register all tool management commands
register_command("/tools-enable", tools_manage_command)
register_command("/tools-disable", tools_manage_command)
register_command("/tools-validate", tools_manage_command)
register_command("/tools-status", tools_manage_command)
register_command("/tools-disabled", tools_manage_command)
register_command("/tools-details", tools_manage_command)
register_command("/tools-autofix", tools_manage_command)
register_command("/tools-clear-validation", tools_manage_command)
register_command("/tools-errors", tools_manage_command)
