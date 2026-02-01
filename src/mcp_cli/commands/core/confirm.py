# src/mcp_cli/commands/core/confirm.py
"""
Unified confirm command implementation (chat/interactive mode).

Toggles tool call confirmation mode between always, never, and smart.
"""

from __future__ import annotations

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.utils.preferences import get_preference_manager, ConfirmationMode


class ConfirmCommand(UnifiedCommand):
    """Toggle or set tool call confirmation mode."""

    @property
    def name(self) -> str:
        return "confirm"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "Toggle tool call confirmation"

    @property
    def help_text(self) -> str:
        return """
Toggle or set tool call confirmation mode.

Usage:
  /confirm                - Toggle confirmation on/off
  /confirm always         - Always confirm before tool execution
  /confirm never          - Never confirm (auto-approve all tools)
  /confirm smart          - Smart mode: confirm based on risk level

Modes:
  always  - Ask for confirmation before every tool call
  never   - Execute all tools without confirmation
  smart   - Only confirm high-risk tool calls (default)
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="mode",
                type=str,
                required=False,
                help="Confirmation mode: always, never, or smart",
                choices=["always", "never", "smart", "on", "off"],
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the confirm command."""
        pref_manager = get_preference_manager()

        # Get current mode
        current_mode = pref_manager.get_tool_confirmation_mode()

        # Get desired mode from args
        mode_arg = kwargs.get("mode")
        if not mode_arg and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                mode_arg = args_val[0]
            elif isinstance(args_val, str):
                mode_arg = args_val

        if mode_arg:
            # Map on/off to always/never
            mode_lower = mode_arg.lower()
            if mode_lower in ("on", "true", "1", "yes"):
                mode_lower = "always"
            elif mode_lower in ("off", "false", "0", "no"):
                mode_lower = "never"

            if mode_lower not in ("always", "never", "smart"):
                return CommandResult(
                    success=False,
                    error=f"Invalid mode: {mode_arg}. Use 'always', 'never', or 'smart'.",
                )

            pref_manager.set_tool_confirmation_mode(mode_lower)
            new_mode = ConfirmationMode(mode_lower)
        else:
            # Toggle: always -> never -> smart -> always
            if current_mode == ConfirmationMode.ALWAYS:
                new_mode = ConfirmationMode.NEVER
            elif current_mode == ConfirmationMode.NEVER:
                new_mode = ConfirmationMode.SMART
            else:
                new_mode = ConfirmationMode.ALWAYS

            pref_manager.set_tool_confirmation_mode(new_mode.value)

        return CommandResult(
            success=True,
            output=f"Tool confirmation mode: {new_mode.value}",
        )
