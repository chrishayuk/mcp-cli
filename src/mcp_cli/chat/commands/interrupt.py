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

from typing import List

# Cross-platform Rich console helper
from chuk_term.ui import output

# Chat-command registry
from mcp_cli.chat.commands import register_command
from mcp_cli.context import get_context


# ════════════════════════════════════════════════════════════════════════════
# Command handlers
# ════════════════════════════════════════════════════════════════════════════
async def interrupt_command(_parts: List[str]) -> bool:  # noqa: D401
    """
    Interrupt currently running operations (streaming or tools).

    Usage
    -----
      /interrupt    - interrupt streaming response or tool execution
      /stop         - same as interrupt
      /cancel       - same as interrupt
    """

    # Use global context manager
    context = get_context()

    # Try to get UI manager from somewhere
    # Note: UI manager might not be stored in ApplicationContext yet
    ui_manager = getattr(context, "ui_manager", None)
    if not ui_manager and hasattr(context, "_extra"):
        ui_manager = context._extra.get("ui_manager")

    interrupted_something = False

    # Check for streaming response
    if ui_manager and getattr(ui_manager, "is_streaming_response", False):
        ui_manager.interrupt_streaming()
        output.warning("Streaming response interrupted.")
        interrupted_something = True

    # Check for running tools
    elif ui_manager and getattr(ui_manager, "tools_running", False):
        if hasattr(ui_manager, "_interrupt_now"):
            ui_manager._interrupt_now()
        output.warning("Tool execution interrupted.")
        interrupted_something = True

    # Check for any tool processor that might be running
    elif hasattr(context, "tool_processor"):
        tool_processor = context.tool_processor
        if hasattr(tool_processor, "cancel_running_tasks"):
            try:
                tool_processor.cancel_running_tasks()
                output.warning("Running tasks cancelled.")
                interrupted_something = True
            except Exception as e:
                output.error(f"Error cancelling tasks: {e}")

    # Nothing to interrupt
    if not interrupted_something:
        output.warning("Nothing currently running to interrupt.")
        output.print(
            "Use this command while streaming responses or tool execution are active."
        )

    return True


async def stop_command(parts: List[str]) -> bool:  # noqa: D401
    """
    Stop currently running operations (alias for interrupt).

    Usage
    -----
      /stop    - stop streaming response or tool execution
    """
    return await interrupt_command(parts)


async def cancel_command(parts: List[str]) -> bool:  # noqa: D401
    """
    Cancel currently running operations (alias for interrupt).

    Usage
    -----
      /cancel    - cancel streaming response or tool execution
    """
    return await interrupt_command(parts)


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/interrupt", interrupt_command)
register_command("/stop", stop_command)
register_command("/cancel", cancel_command)
