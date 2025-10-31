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
from mcp_cli.constants import OAUTH_NAMESPACE, GENERIC_NAMESPACE


class TokenCommand(UnifiedCommand):
    """Manage OAuth and authentication tokens."""

    @property
    def name(self) -> str:
        return "token"

    @property
    def aliases(self) -> List[str]:
        return ["tokens"]

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
  /token set <name> <value> - Store a bearer token
  /token get <name>   - Get details for a specific token
  /token clear        - Clear all tokens (with confirmation)
  /token clear --force - Clear all tokens without confirmation
  /token delete <name> - Delete a specific token

Examples:
  /token              # Show all tokens
  /token list         # Show all tokens
  /token set my-api secret-token  # Store a bearer token
  /token get my-api   # Show token details
  /token get notion   # Show notion OAuth token details
  /token clear        # Clear all tokens (asks for confirmation)
  /token delete my-api # Delete the token
"""

    @property
    def modes(self) -> CommandMode:
        """Token is for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def requires_context(self) -> bool:
        """Token needs tool manager context to get server list."""
        return True

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the token command."""
        from chuk_term.ui import output
        from mcp_cli.commands.actions.token import (
            token_list_action_async,
            token_clear_action_async,
            token_delete_action_async,
            token_get_action_async,
            token_set_action_async,
        )

        # Get args and tool_manager from kwargs
        args = kwargs.get("args", [])
        if isinstance(args, str):
            args = [args]

        tool_manager = kwargs.get("tool_manager")
        server_names = tool_manager.servers if tool_manager else []

        # Default action is list if no args provided
        if not args or len(args) == 0:
            from mcp_cli.commands.models import TokenListParams

            params = TokenListParams(server_names=server_names)
            await token_list_action_async(params)
            return CommandResult(success=True)

        # Parse subcommand (first arg)
        subcommand = args[0].lower()

        if subcommand == "list":
            from mcp_cli.commands.models import TokenListParams

            params = TokenListParams(server_names=server_names)
            await token_list_action_async(params)
            return CommandResult(success=True)

        elif subcommand == "clear":
            # Check for --force flag
            force = "--force" in args or "-f" in args
            from mcp_cli.commands.models import TokenClearParams

            clear_params = TokenClearParams(force=force)
            await token_clear_action_async(clear_params)
            return CommandResult(success=True)

        elif subcommand == "set":
            if len(args) < 3:
                output.error("Token name and value required for set command")
                output.hint("Usage: /token set <name> <value>")
                return CommandResult(success=False)

            token_name = args[1]
            token_value = args[2]
            # Store as bearer token in generic namespace
            from mcp_cli.commands.models import TokenSetParams

            set_params = TokenSetParams(
                name=token_name,
                value=token_value,
                token_type="bearer",
                namespace=GENERIC_NAMESPACE,
            )
            await token_set_action_async(set_params)
            return CommandResult(success=True)

        elif subcommand == "get":
            if len(args) < 2:
                output.error("Token name required for get command")
                output.hint("Usage: /token get <name>")
                return CommandResult(success=False)

            token_name = args[1]
            # Try OAuth namespace first, then generic
            await token_get_action_async(token_name, namespace=OAUTH_NAMESPACE)
            # If not found in OAuth, try generic
            await token_get_action_async(token_name, namespace=GENERIC_NAMESPACE)
            return CommandResult(success=True)

        elif subcommand == "delete":
            if len(args) < 2:
                output.error("Token name required for delete command")
                output.hint("Usage: /token delete <name>")
                return CommandResult(success=False)

            token_name = args[1]
            # OAuth tokens are the most common use case in chat
            from mcp_cli.commands.models import TokenDeleteParams

            delete_params = TokenDeleteParams(name=token_name, oauth=True)
            await token_delete_action_async(delete_params)
            return CommandResult(success=True)

        else:
            output.error(f"Unknown token subcommand: {subcommand}")
            output.hint("Available: list, set, get, clear, delete")
            output.hint("Type /help token for more information")
            return CommandResult(success=False)
