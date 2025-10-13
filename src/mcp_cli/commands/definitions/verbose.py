# src/mcp_cli/commands/definitions/verbose.py
"""
Unified verbose command implementation (chat mode only).
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)


class VerboseCommand(UnifiedCommand):
    """Toggle verbose output mode."""

    @property
    def name(self) -> str:
        return "verbose"

    @property
    def aliases(self) -> List[str]:
        return []

    @property
    def description(self) -> str:
        return "Toggle verbose output mode"

    @property
    def help_text(self) -> str:
        return """
Toggle verbose output mode for debugging.

Usage:
  /verbose [on|off]    - Set verbose mode
  /verbose             - Toggle verbose mode
  
Examples:
  /verbose         - Toggle current state
  /verbose on      - Enable verbose output
  /verbose off     - Disable verbose output
"""

    @property
    def modes(self) -> CommandMode:
        """This is primarily for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="state",
                type=str,
                required=False,
                help="Set verbose state",
                choices=["on", "off", "true", "false"],
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the verbose command."""
        # Get UI manager or context
        ui_manager = kwargs.get("ui_manager")
        chat_handler = kwargs.get("chat_handler")

        # Get current state
        current_verbose = False
        if ui_manager and hasattr(ui_manager, "verbose_mode"):
            current_verbose = ui_manager.verbose_mode
        elif chat_handler and hasattr(chat_handler, "verbose_mode"):
            current_verbose = chat_handler.verbose_mode

        # Get desired state
        state = kwargs.get("state")
        if not state and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                state = args_val[0]
            elif isinstance(args_val, str):
                state = args_val

        # Determine new state
        if state:
            state_lower = state.lower()
            if state_lower in ["on", "true", "1", "yes"]:
                new_verbose = True
            elif state_lower in ["off", "false", "0", "no"]:
                new_verbose = False
            else:
                return CommandResult(
                    success=False,
                    error=f"Invalid state: {state}. Use 'on' or 'off'.",
                )
        else:
            # Toggle
            new_verbose = not current_verbose

        # Apply new state
        if ui_manager and hasattr(ui_manager, "verbose_mode"):
            ui_manager.verbose_mode = new_verbose
        if chat_handler and hasattr(chat_handler, "verbose_mode"):
            chat_handler.verbose_mode = new_verbose

        # Also try to set logging level
        if new_verbose:
            import logging

            logging.getLogger().setLevel(logging.DEBUG)
            status = "enabled"
        else:
            import logging

            logging.getLogger().setLevel(logging.WARNING)
            status = "disabled"

        return CommandResult(
            success=True,
            output=f"Verbose mode {status}.",
        )
