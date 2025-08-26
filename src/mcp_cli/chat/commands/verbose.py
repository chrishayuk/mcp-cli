# mcp_cli/chat/commands/verbose.py
"""
Chat-mode "/verbose" command for MCP-CLI
=======================================

This module implements the **/verbose** (alias **/v**) slash-command that
toggles between verbose and compact display modes for tool execution and
streaming responses.

Display Modes
-------------
* **Verbose mode** - shows full details of each tool call with JSON arguments
* **Compact mode** - shows condensed, animated progress view (default)

The mode affects:
- Tool execution display
- Streaming response formatting
- Progress indicators

Features
--------
* **Persistent setting** - mode is remembered for the session
* **Real-time switching** - can toggle during tool execution
* **Status display** - shows current mode when toggled
* **Context integration** - works with both UI manager and streaming handler

Examples
--------
>>> /verbose      # toggle between verbose/compact
>>> /v            # short alias
"""

from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output

# Chat-command registry
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Command handler
# ════════════════════════════════════════════════════════════════════════════
async def verbose_command(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Toggle between verbose and compact display modes.

    Usage
    -----
      /verbose    - toggle display mode
      /v          - short alias

    Modes
    -----
    * **Verbose**: Shows full tool call details with JSON arguments
    * **Compact**: Shows condensed progress with animations (default)
    """

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

    # Toggle verbose mode
    current_mode = getattr(ui_manager, "verbose_mode", False)
    new_mode = not current_mode
    ui_manager.verbose_mode = new_mode

    # Update any streaming handlers if they exist
    streaming_handler = getattr(ui_manager, "streaming_handler", None)
    if streaming_handler and hasattr(streaming_handler, "verbose_mode"):
        streaming_handler.verbose_mode = new_mode

    # Show status
    mode_name = "verbose" if new_mode else "compact"
    mode_desc = (
        "full tool details and expanded responses"
        if new_mode
        else "condensed progress and animations"
    )

    output.print(f"[green]Display mode:[/green] {mode_name}")
    output.print(f"[dim]Shows {mode_desc}[/dim]")

    # If tools are currently running, show how the change affects them
    if getattr(ui_manager, "tools_running", False):
        if new_mode:
            output.print("[cyan]Future tool calls will show full details.[/cyan]")
        else:
            output.print("[cyan]Switched to compact tool progress display.[/cyan]")

    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/verbose", verbose_command)
register_command("/v", verbose_command)
