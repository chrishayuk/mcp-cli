# mcp_cli/commands/tools_manage.py
"""
Tool management commands for enabling/disabling tools and validation.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from rich.table import Table
from rich.panel import Panel
from chuk_term.ui import output

from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


async def tools_manage_action_async(
    tm: ToolManager, action: str, tool_name: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Manage tools (enable/disable/validate).

    Args:
        tm: Tool manager
        action: Action to perform (enable, disable, validate, status, list-disabled)
        tool_name: Tool name for specific actions

    Returns:
        Action result dictionary
    """

    if action == "enable":
        if not tool_name:
            output.print("[red]Error: Tool name required for enable action[/red]")
            return {"success": False, "error": "Tool name required"}

        tm.enable_tool(tool_name)
        output.print(f"[green]✓ Enabled tool:[/green] {tool_name}")
        return {"success": True, "action": "enable", "tool": tool_name}

    elif action == "disable":
        if not tool_name:
            output.print("[red]Error: Tool name required for disable action[/red]")
            return {"success": False, "error": "Tool name required"}

        tm.disable_tool(tool_name, reason="user")
        output.print(f"[yellow]✗ Disabled tool:[/yellow] {tool_name}")
        return {"success": True, "action": "disable", "tool": tool_name}

    elif action == "validate":
        if tool_name:
            # Validate single tool
            is_valid, error_msg = await tm.validate_single_tool(tool_name)
            if is_valid:
                output.print(f"[green]✓ Tool '{tool_name}' is valid[/green]")
            else:
                output.print(f"[red]✗ Tool '{tool_name}' is invalid:[/red] {error_msg}")

            return {
                "success": True,
                "action": "validate",
                "tool": tool_name,
                "is_valid": is_valid,
                "error": error_msg,
            }
        else:
            # Validate all tools
            provider = kwargs.get("provider", "openai")
            output.print(f"[cyan]Validating all tools for {provider}...[/cyan]")

            summary = await tm.revalidate_tools(provider)

            output.print("[green]Validation complete:[/green]")
            output.print(f"  • Total tools: {summary.get('total_tools', 0)}")
            output.print(f"  • Valid: {summary.get('valid_tools', 0)}")
            output.print(f"  • Invalid: {summary.get('invalid_tools', 0)}")

            return {"success": True, "action": "validate_all", "summary": summary}

    elif action == "status":
        summary = tm.get_validation_summary()

        # Create status table
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
        return {"success": True, "action": "status", "summary": summary}

    elif action == "list-disabled":
        disabled_tools = tm.get_disabled_tools()

        if not disabled_tools:
            output.print("[green]No disabled tools[/green]")
        else:
            table = Table(title="Disabled Tools")
            table.add_column("Tool Name", style="yellow")
            table.add_column("Reason", style="red")

            for tool, reason in disabled_tools.items():
                table.add_row(tool, reason)

            output.print(table)

        return {
            "success": True,
            "action": "list_disabled",
            "disabled_tools": disabled_tools,
        }

    elif action == "details":
        if not tool_name:
            output.print("[red]Error: Tool name required for details action[/red]")
            return {"success": False, "error": "Tool name required"}

        details = tm.get_tool_validation_details(tool_name)
        if not details:
            output.print(f"[red]Tool '{tool_name}' not found[/red]")
            return {"success": False, "error": "Tool not found"}

        # Display details panel
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
        return {
            "success": True,
            "action": "details",
            "tool": tool_name,
            "details": details,
        }

    elif action == "auto-fix":
        setting = kwargs.get("enabled", True)
        tm.set_auto_fix_enabled(setting)
        status = "enabled" if setting else "disabled"
        output.print(f"[cyan]Auto-fix {status}[/cyan]")
        return {"success": True, "action": "auto_fix", "enabled": setting}

    elif action == "clear-validation":
        tm.clear_validation_disabled_tools()
        output.print("[green]Cleared all validation-disabled tools[/green]")
        return {"success": True, "action": "clear_validation"}

    elif action == "validation-errors":
        summary = tm.get_validation_summary()
        errors = summary.get("validation_errors", [])

        if not errors:
            output.print("[green]No validation errors[/green]")
        else:
            output.print(f"[red]Found {len(errors)} validation errors:[/red]")
            for error in errors:
                output.print(f"  • {error['tool']}: {error['error']}")

        return {"success": True, "action": "validation_errors", "errors": errors}

    else:
        output.print(f"[red]Unknown action: {action}[/red]")
        return {"success": False, "error": f"Unknown action: {action}"}


def tools_manage_action(
    tm: ToolManager, action: str, tool_name: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """Sync wrapper for tool management actions."""
    return asyncio.run(tools_manage_action_async(tm, action, tool_name, **kwargs))
