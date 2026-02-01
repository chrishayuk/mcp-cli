# src/mcp_cli/commands/definitions/servers.py
"""
Unified servers command implementation.

This single implementation works across all modes (chat, CLI, interactive).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)
from mcp_cli.config.enums import OutputFormat

if TYPE_CHECKING:
    from mcp_cli.commands.models.server import ServerStatusInfo


class ServersCommand(UnifiedCommand):
    """List and manage MCP servers."""

    @property
    def name(self) -> str:
        return "servers"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "List connected MCP servers and their status"

    @property
    def help_text(self) -> str:
        return """
List connected MCP servers and their status.

Usage:
  /servers              - List all connected servers
  /servers --detailed   - Show detailed server information
  /servers --ping       - Test server connectivity
  
Options:
  --detailed            - Show detailed server information
  --format [table|json] - Output format (default: table)
  --ping                - Test server connectivity

Examples:
  /servers              - Show server status table
  /servers --detailed   - Show full server details
  /servers --ping       - Check server connectivity

Note: For server management (add/remove/enable/disable), use /server command
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="detailed",
                type=bool,
                default=False,
                help="Show detailed server information",
                is_flag=True,
            ),
            CommandParameter(
                name="format",
                type=str,
                default="table",
                help="Output format",
                choices=["table", "json"],
            ),
            CommandParameter(
                name="ping",
                type=bool,
                default=False,
                help="Test server connectivity",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the servers command."""
        import json

        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table

        # Extract parameters
        _ = kwargs.get("detailed", False)  # Reserved for future use
        ping_servers = kwargs.get("ping", False)
        output_format = kwargs.get("format", "table")

        # Get context and tool manager
        context = get_context()
        if not context or not context.tool_manager:
            return CommandResult(
                success=False,
                error="No tool manager available. Please connect to a server first.",
            )

        try:
            # Get server information (Pydantic ServerInfo models)
            servers = await context.tool_manager.get_server_info()

            if not servers:
                return CommandResult(
                    success=True,
                    output="No servers connected.",
                )

            # Output as JSON if requested
            if output_format == OutputFormat.JSON:
                server_data = [s.model_dump() for s in servers]
                output.print(json.dumps(server_data, indent=2, default=str))
                return CommandResult(success=True, data=server_data)

            # Build table rows from Pydantic models
            table_data = []
            for server in servers:
                status = self._get_server_status(server)

                # Get connection info based on transport type
                connection = self._get_connection_info(server)

                row = {
                    "Server": server.name,
                    "Type": server.transport.value.upper(),
                    "Status": f"{status.icon} {status.status}",
                    "Tools": str(server.tool_count),
                    "Connection": connection,
                }

                table_data.append(row)

            # Display table with all columns
            columns = ["Server", "Type", "Status", "Tools", "Connection"]

            table = format_table(
                table_data,
                title=f"{len(servers)} Connected Servers",
                columns=columns,
            )
            output.print_table(table)

            if ping_servers:
                output.info("\n🏓 Pinging servers...")
                output.hint("Use /ping for detailed connectivity testing")

            return CommandResult(success=True)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list servers: {str(e)}",
            )

    def _get_server_status(self, server) -> "ServerStatusInfo":
        """Get status info for a server."""
        from mcp_cli.commands.models.server import ServerStatusInfo

        if server.connected:
            return ServerStatusInfo(
                icon="✅",
                status="Connected",
                reason="Server is online and responding",
            )
        return ServerStatusInfo(
            icon="❌",
            status="Disconnected",
            reason="Server is not responding",
        )

    def _get_connection_info(self, server) -> str:
        """Get connection info string for display."""
        from mcp_cli.tools.models import TransportType

        if server.transport == TransportType.STDIO:
            if server.command:
                # Show command with truncated args
                cmd = str(server.command)
                if server.args:
                    args_str = " ".join(str(a) for a in server.args[:2])
                    if len(server.args) > 2:
                        args_str += " ..."
                    return f"{cmd} {args_str}"
                return cmd
            return "stdio"
        elif server.transport in (TransportType.HTTP, TransportType.SSE):
            if server.url:
                # Truncate long URLs
                url = str(server.url)
                if len(url) > 40:
                    return url[:37] + "..."
                return url
            return str(server.transport.value)
        return "unknown"
