# src/mcp_cli/commands/definitions/interrupt.py
"""
Unified interrupt command implementation (chat mode only).
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
)


class InterruptCommand(UnifiedCommand):
    """Interrupt currently running operations."""

    @property
    def name(self) -> str:
        return "interrupt"

    @property
    def aliases(self) -> List[str]:
        return ["stop", "cancel"]

    @property
    def description(self) -> str:
        return "Interrupt currently running operations"

    @property
    def help_text(self) -> str:
        return """
Interrupt currently running operations like streaming responses or tool execution.

Usage:
  /interrupt    - Stop current operation
  /stop         - Alias for interrupt
  /cancel       - Another alias
  
This command is streaming-aware and will gracefully interrupt:
- Streaming text generation
- Running tool executions
- Long-running operations

Note: This command is only available in chat mode.
"""

    @property
    def modes(self) -> CommandMode:
        """This is a chat-only command."""
        return CommandMode.CHAT

    @property
    def requires_context(self) -> bool:
        """Interrupt needs chat context."""
        return True

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the interrupt command."""
        # Get chat context
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Interrupt command requires chat context.",
            )

        # Check if there's anything to interrupt
        interrupted_something = False

        # Try to interrupt streaming
        if hasattr(chat_context, "is_streaming") and chat_context.is_streaming:
            if hasattr(chat_context, "interrupt_streaming"):
                chat_context.interrupt_streaming()
                interrupted_something = True
                message = "Streaming response interrupted."

        # Try to interrupt tool execution
        if (
            not interrupted_something
            and hasattr(chat_context, "is_executing_tool")
            and chat_context.is_executing_tool
        ):
            if hasattr(chat_context, "interrupt_tool_execution"):
                chat_context.interrupt_tool_execution()
                interrupted_something = True
                message = "Tool execution interrupted."

        # Try general cancellation
        if not interrupted_something and hasattr(
            chat_context, "cancel_current_operation"
        ):
            chat_context.cancel_current_operation()
            interrupted_something = True
            message = "Current operation cancelled."

        if interrupted_something:
            return CommandResult(
                success=True,
                output=message,
            )
        else:
            return CommandResult(
                success=True,
                output="Nothing to interrupt.",
            )
