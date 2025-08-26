from __future__ import annotations

"""
Enhanced /confirm command with ability to manage the tool call confirmation preferences (as applied to the current config):

Usage Examples
--------------
/confirm tool_calls         - Toggle tool call confirmations globally (on by default). Previous individual tool call preferences will be cleared.
/confirm tool_calls list    - List current tool call confirmation preferences (ALWAYS|NEVER)
/confirm tool_calls toggle TOOL_NAME - Toggles whether a tool will always or never be confirmed before execution

"""

from typing import Any, Dict, List

from chuk_term.ui import output
from mcp_cli.tools.manager import ToolManager
from mcp_cli.commands.tools import tools_action_async
from mcp_cli.chat.commands import register_command


async def confirm_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Command to control confirmations, for example durin tool calls."""

    # Get UI manager from context
    ui_manager = ctx.get("ui_manager")
    if not ui_manager:
        # Fallback: look for context object that might have UI manager
        context_obj = ctx.get("context")
        if context_obj and hasattr(context_obj, "ui_manager"):
            ui_manager = context_obj.ui_manager

    if not ui_manager:
        output.print("[red]Error:[/red] UI manager not available.")
        return True

    # Parse arguments
    args = parts[1:]  # Remove command name
    feature = "tool_calls"  # Default area for confirmations
    if len(args) == 0:
        # No arguments, toggle global tool call confirmations
        output.print(
            "[green][dim]No feature specified for confirmations. Defaulting to 'tool_calls' feature.[/dim][/green]"
        )
        args = ["tool_calls"]

    # Parse flags
    if len(args) == 1 and args[0] != "tool_calls":
        output.print(
            f"[red]Error:[/red] Unsupported feature '{args[0]}'. Currently only 'tool_calls' is supported."
        )
        output.print(
            "[dim]Use '/confirm tool_calls' to toggle tool call confirmations.[/dim]"
        )
        return True

    if len(args) == 1:
        # Toggle tool call confirmations globally
        current_mode = getattr(ui_manager, "confirm_tool_execution", True)
        new_mode = not current_mode
        ui_manager.confirm_tool_execution = new_mode
        output.print(
            f"[green]Tool call confirmations {'enabled' if new_mode else 'disabled'} globally.[/green]"
        )
        return True

    elif len(args) == 2:
        # Handle subcommands like verbose or list
        subcommand = args[1]

        if subcommand == "list":
            # List current tool call confirmation preferences
            confirmations = ui_manager.get_tool_call_confirmations()
            if not confirmations:
                output.print("[dim]No tool call confirmations set.[/dim]")
            else:
                output.print("[cyan]Current tool call confirmations:[/cyan]")
                for tool, mode in confirmations.items():
                    output.print(f"  {tool}: {mode}")
            return True
        else:
            output.print(
                f"[red]Error:[/red] Unknown subcommand '{subcommand}'. Use 'verbose' or 'list'."
            )
            return True
    elif len(args) == 3:
        # Toggle specific tool confirmation
        tool_name = args[2]
        if not tool_name:
            output.print("[red]Error:[/red] Tool name must be provided for toggle.")
            return True

        tm: ToolManager | None = ctx.get("tool_manager")
        if tm is None:
            output.print("[red]Error:[/red] ToolManager not available.")
            return True  # command handled

        await tools_action_async(
            tm,
            show_details=show_details,
            show_raw=show_raw,
        )

        # Toggle the confirmation for the specified tool
        current_confirmations = ui_manager.get_tool_call_confirmations()
        if tool_name in current_confirmations:
            del current_confirmations[tool_name]
            output.print(f"[green]Tool '{tool_name}' confirmation removed.[/green]")
        else:
            current_confirmations[tool_name] = "ALWAYS"
            output.print(
                f"[green]Tool '{tool_name}' confirmation set to ALWAYS.[/green]"
            )

        ui_manager.set_tool_call_confirmations(current_confirmations)

        return True

    return True


# Register main command and alias
register_command("/confirm", confirm_command)
