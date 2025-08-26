# mcp_cli/chat/commands/interrupt.py
"""
Chat-mode "/interrupt" command for MCP-CLI with streaming support
================================================================

This module implements the **/interrupt**, **/stop**, and **/cancel**
slash-commands that allow users to gracefully interrupt:

1. **Streaming responses** - stops the live text generation
2. **Tool execution** - cancels running tool calls
3. **Long-running operations** - general cancellation

The command is streaming-aware and provides appropriate feedback based
on what's currently running.

Features
--------
* **Streaming-aware** - detects and interrupts streaming responses
* **Tool-aware** - cancels running tool executions
* **Graceful handling** - provides clear feedback about what was interrupted
* **Multiple aliases** - `/interrupt`, `/stop`, `/cancel` all work

Examples
--------
>>> /interrupt    # stops whatever is currently running
>>> /stop         # alias for interrupt
>>> /cancel       # another alias
"""

from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output

# Chat-command registry
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Command handlers
# ════════════════════════════════════════════════════════════════════════════
async def interrupt_command(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Interrupt currently running operations (streaming or tools).

    Usage
    -----
      /interrupt    - interrupt streaming response or tool execution
      /stop         - same as interrupt
      /cancel       - same as interrupt
    """

    # Get UI manager from context if available
    ui_manager = ctx.get("ui_manager")
    if not ui_manager:
        # Fallback: look for context object that might have UI manager
        context_obj = ctx.get("context")
        if context_obj and hasattr(context_obj, "ui_manager"):
            ui_manager = context_obj.ui_manager

    interrupted_something = False

    # Check for streaming response
    if ui_manager and getattr(ui_manager, "is_streaming_response", False):
        ui_manager.interrupt_streaming()
        output.print("[yellow]Streaming response interrupted.[/yellow]")
        interrupted_something = True

    # Check for running tools
    elif ui_manager and getattr(ui_manager, "tools_running", False):
        if hasattr(ui_manager, "_interrupt_now"):
            ui_manager._interrupt_now()
        output.print("[yellow]Tool execution interrupted.[/yellow]")
        interrupted_something = True

    # Check for any tool processor that might be running
    elif "tool_processor" in ctx:
        tool_processor = ctx["tool_processor"]
        if hasattr(tool_processor, "cancel_running_tasks"):
            try:
                tool_processor.cancel_running_tasks()
                output.print("[yellow]Running tasks cancelled.[/yellow]")
                interrupted_something = True
            except Exception as e:
                output.print(f"[red]Error cancelling tasks: {e}[/red]")

    # Nothing to interrupt
    if not interrupted_something:
        output.print("[yellow]Nothing currently running to interrupt.[/yellow]")
        output.print(
            "[dim]Use this command while streaming responses or tool execution are active.[/dim]"
        )

    return True


async def stop_command(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Stop currently running operations (alias for interrupt).

    Usage
    -----
      /stop    - stop streaming response or tool execution
    """
    return await interrupt_command(parts, ctx)


async def cancel_command(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Cancel currently running operations (alias for interrupt).

    Usage
    -----
      /cancel    - cancel streaming response or tool execution
    """
    return await interrupt_command(parts, ctx)


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/interrupt", interrupt_command)
register_command("/stop", stop_command)
register_command("/cancel", cancel_command)
