# src/mcp_cli/commands/definitions/server_singular.py
"""
Singular server command - shows details about a specific server.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandResult,
)


class ServerSingularCommand(UnifiedCommand):
    """Show details about a specific MCP server."""

    @property
    def name(self) -> str:
        return "server"

    @property
    def aliases(self) -> List[str]:
        return []  # No aliases for singular form

    @property
    def description(self) -> str:
        return "Show details about a specific MCP server"

    @property
    def help_text(self) -> str:
        return """
Show detailed information about a specific MCP server.

Usage:
  /server <name>      - Show details about a specific server
  
Examples:
  /server echo        - Show details about the echo server
  /server sqlite      - Show details about the sqlite server
  /server filesystem  - Show details about the filesystem server
"""

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the server command."""
        from mcp_cli.commands.actions.servers import server_details_async
        from chuk_term.ui import output

        # Get args - server name is required
        args = kwargs.get("args", [])

        if not args:
            output.error("Server name required")
            output.hint("Usage: /server <name>")
            output.hint("Use /servers to list all available servers")
            return CommandResult(success=False, error="Server name required")

        # Get server name
        server_name = args[0] if isinstance(args, list) else str(args)

        try:
            # Show detailed info about the specific server
            await server_details_async(server_name)
            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to get server details: {str(e)}"
            )
