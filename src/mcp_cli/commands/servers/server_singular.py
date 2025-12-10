# src/mcp_cli/commands/definitions/server_singular.py
"""
Server command - manages MCP servers (add, remove, enable, disable) and shows server details.
Supports both project servers (server_config.json) and user servers (~/.mcp-cli/preferences.json).
"""

from __future__ import annotations


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
    def aliases(self) -> list[str]:
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
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table

        # Get args - handle both string and list
        args = kwargs.get("args", [])
        if isinstance(args, str):
            args = [args]
        elif not args:
            args = []

        # Get context and tool manager
        context = get_context()
        if not context or not context.tool_manager:
            return CommandResult(
                success=False,
                error="No tool manager available. Please connect to a server first.",
            )

        if not args:
            # No args - show list of servers
            try:
                servers = await context.tool_manager.get_server_info()

                if not servers:
                    return CommandResult(
                        success=True,
                        output="No servers connected.",
                    )

                # Build table data
                table_data = []
                for server in servers:
                    table_data.append(
                        {
                            "Server": server.name,
                            "Status": "✓ Connected"
                            if server.connected
                            else "✗ Disconnected",
                            "Tools": str(server.tool_count),
                        }
                    )

                # Display table
                table = format_table(
                    table_data,
                    title=f"{len(servers)} Servers",
                    columns=["Server", "Status", "Tools"],
                )
                output.print_table(table)

                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to list servers: {str(e)}"
                )

        # Has arguments - handle server management commands
        first_arg = args[0].lower()

        if first_arg == "list":
            # List all servers
            try:
                servers = await context.tool_manager.get_server_info()
                # Same as no args case
                if not servers:
                    return CommandResult(success=True, output="No servers connected.")

                table_data = []
                for server in servers:
                    table_data.append(
                        {
                            "Server": server.name,
                            "Status": "✓ Connected"
                            if server.connected
                            else "✗ Disconnected",
                            "Tools": str(server.tool_count),
                        }
                    )

                table = format_table(
                    table_data,
                    title=f"{len(servers)} Servers",
                    columns=["Server", "Status", "Tools"],
                )
                output.print_table(table)

                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to list servers: {str(e)}"
                )

        elif first_arg in ["add", "remove", "enable", "disable", "ping"]:
            # These commands need more complex implementation
            return CommandResult(
                success=False,
                error=f"Server {first_arg} command not yet implemented in this version.",
            )
        else:
            # Treat as server name - show server details
            server_name = first_arg
            try:
                servers = await context.tool_manager.get_server_info()
                from mcp_cli.tools.models import ServerInfo

                found_server: ServerInfo | None = next(
                    (s for s in servers if s.name == server_name), None
                )

                if not found_server:
                    return CommandResult(
                        success=False,
                        error=f"Server '{server_name}' not found.",
                    )

                # Show server details
                output.panel(
                    f"Name: {found_server.name}\n"
                    f"Status: {'Connected' if found_server.connected else 'Disconnected'}\n"
                    f"Tools: {found_server.tool_count}\n"
                    f"Transport: {found_server.transport.value if found_server.transport else 'unknown'}",
                    title=f"Server: {found_server.name}",
                )

                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to get server details: {str(e)}"
                )
