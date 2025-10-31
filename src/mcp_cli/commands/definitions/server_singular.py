# src/mcp_cli/commands/definitions/server_singular.py
"""
Server command - manages MCP servers (add, remove, enable, disable) and shows server details.
Supports both project servers (server_config.json) and user servers (~/.mcp-cli/preferences.json).
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandResult,
)


class ServerSingularCommand(UnifiedCommand):
    """Manage MCP servers - add, remove, enable, disable, or show details."""

    @property
    def name(self) -> str:
        return "server"

    @property
    def aliases(self) -> List[str]:
        return []  # No aliases for singular form

    @property
    def description(self) -> str:
        return "Manage MCP servers or show server details"

    @property
    def help_text(self) -> str:
        return """
Manage MCP servers or show details about a specific server.

Usage:
  /server                                         - List all servers
  /server <name>                                  - Show server details
  /server list                                    - List all servers
  /server list all                                - Include disabled servers
  
Server Management:
  /server add <name> stdio <command> [args...]    - Add STDIO server
  /server add <name> --transport http <url>       - Add HTTP server  
  /server add <name> --transport sse <url>        - Add SSE server
  /server remove <name>                           - Remove user-added server
  /server enable <name>                           - Enable disabled server
  /server disable <name>                          - Disable server
  /server ping <name>                             - Test server connectivity
  
Examples:
  /server                                          - List all servers
  /server sqlite                                   - Show sqlite server details
  /server add time stdio uvx mcp-server-time      - Add time server
  /server add myapi --transport http --header "Authorization: Bearer token" -- https://api.example.com
  /server disable sqlite                          - Disable sqlite server
  /server remove time                             - Remove time server
  
Note: User-added servers persist in ~/.mcp-cli/preferences.json
"""

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the server command."""
        from mcp_cli.commands.actions.servers import servers_action_async

        # Get args - handle both string and list
        args = kwargs.get("args", [])
        if isinstance(args, str):
            args = [args]
        elif not args:
            args = []

        if not args:
            # No args - show list of servers
            from mcp_cli.commands.models import ServerActionParams

            await servers_action_async(ServerActionParams())
            return CommandResult(success=True)

        try:
            from mcp_cli.commands.models import ServerActionParams

            # Pass args to the enhanced servers action which handles all management
            params = ServerActionParams(args=args)
            await servers_action_async(params)
            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to execute server command: {str(e)}"
            )
