# src/mcp_cli/commands/definitions/token.py
"""
Unified token command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
)


class TokenCommand(UnifiedCommand):
    """Manage OAuth and authentication tokens."""

    @property
    def name(self) -> str:
        return "token"

    @property
    def aliases(self) -> List[str]:
        return []

    @property
    def description(self) -> str:
        return "Manage OAuth and authentication tokens"

    @property
    def help_text(self) -> str:
        return """
Manage OAuth and authentication tokens.

Usage:
  /token              - List all stored tokens (chat/interactive mode)
  /token list         - List all stored tokens
  /token clear        - Clear all tokens (with confirmation)
  /token clear --force - Clear all tokens without confirmation
  /token delete <name> - Delete a specific OAuth token

Examples:
  /token              # Show all tokens
  /token list         # Show all tokens
  /token clear        # Clear all tokens (asks for confirmation)
  /token delete linkedin  # Delete LinkedIn OAuth token
"""

    @property
    def modes(self) -> CommandMode:
        """Token is for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def requires_context(self) -> bool:
        """Token doesn't need tool manager context."""
        return False

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the token command."""
        from chuk_term.ui import output
        from mcp_cli.commands.actions.token import (
            token_list_action_async,
            token_clear_action_async,
            token_delete_action_async,
        )

        # Get args from kwargs
        args = kwargs.get("args", [])
        if isinstance(args, str):
            args = [args]

        # Default action is list if no args provided
        if not args or len(args) == 0:
            await token_list_action_async()
            return CommandResult(success=True)

        # Parse subcommand (first arg)
        subcommand = args[0].lower()

        if subcommand == "list":
            await token_list_action_async()
            return CommandResult(success=True)

        elif subcommand == "clear":
            # Check for --force flag
            force = "--force" in args or "-f" in args
            await token_clear_action_async(force=force)
            return CommandResult(success=True)

        elif subcommand == "delete":
            if len(args) < 2:
                output.error("Token name required for delete command")
                output.hint("Usage: /token delete <name>")
                return CommandResult(success=False)

            token_name = args[1]
            # OAuth tokens are the most common use case in chat
            await token_delete_action_async(name=token_name, oauth=True)
            return CommandResult(success=True)

        else:
            output.error(f"Unknown token subcommand: {subcommand}")
            output.hint("Available: list, clear, delete")
            output.hint("Type /help token for more information")
            return CommandResult(success=False)
