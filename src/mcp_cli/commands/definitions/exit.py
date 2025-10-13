# src/mcp_cli/commands/definitions/exit.py
"""
Unified exit command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
)


class ExitCommand(UnifiedCommand):
    """Exit the application."""

    @property
    def name(self) -> str:
        return "exit"

    @property
    def aliases(self) -> List[str]:
        return ["quit", "q", "bye"]

    @property
    def description(self) -> str:
        return "Exit the application"

    @property
    def help_text(self) -> str:
        return """
Exit the current session.

Usage:
  /exit     - Exit chat mode
  exit      - Exit interactive mode
  
Aliases: quit, q, bye

Note: In CLI mode, commands exit automatically after completion.
"""

    @property
    def modes(self) -> CommandMode:
        """Exit is only for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def requires_context(self) -> bool:
        """Exit doesn't need tool manager context."""
        return False

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the exit command."""
        # Import the exit action
        from mcp_cli.commands.actions.exit import exit_action

        # Execute the action (interactive=True for chat/interactive modes)
        exit_action(interactive=True)

        return CommandResult(
            success=True,
            output="Goodbye!",
            should_exit=True,
        )
