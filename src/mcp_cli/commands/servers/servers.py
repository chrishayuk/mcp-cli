# src/mcp_cli/commands/definitions/servers.py
"""
Unified servers command implementation.

This single implementation works across all modes (chat, CLI, interactive).
"""

from __future__ import annotations


from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)


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
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table
        import json

        # Extract parameters
        detailed = kwargs.get("detailed", False)
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
            # Get server information
            servers = await context.tool_manager.get_server_info()

            if not servers:
                return CommandResult(
                    success=True,
                    output="No servers connected.",
                )

            # Output as JSON if requested
            if output_format == "json":
                server_data = [
                    {
                        "name": s.name,
                        "status": s.status,
                        "tool_count": s.tool_count,
                        "transport": s.transport.value if s.transport else "unknown",
                    }
                    for s in servers
                ]
                output.print(json.dumps(server_data, indent=2))
                return CommandResult(success=True, data=server_data)

            # Build table data
            table_data = []
            for server in servers:
                row = {
                    "Server": server.name,
                    "Status": "✓ Connected" if server.connected else "✗ Disconnected",
                    "Tools": str(server.tool_count),
                }

                if detailed:
                    row["Transport"] = (
                        server.transport.value if server.transport else "unknown"
                    )
                    row["Namespace"] = server.namespace or "default"

                table_data.append(row)

            # Display table
            columns = ["Server", "Status", "Tools"]
            if detailed:
                columns.extend(["Transport", "Namespace"])

            table = format_table(
                table_data,
                title=f"{len(servers)} Connected Servers",
                columns=columns,
            )
            output.print_table(table)

            if ping_servers:
                output.info("\n🏓 Pinging servers...")
                output.hint("Use /ping for detailed connectivity testing")

            return CommandResult(success=True, data=table_data)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list servers: {str(e)}",
            )
