# src/mcp_cli/commands/definitions/servers.py
"""
Unified servers command implementation.

This single implementation works across all modes (chat, CLI, interactive).
"""

from __future__ import annotations

from typing import List

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
    def aliases(self) -> List[str]:
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
    def parameters(self) -> List[CommandParameter]:
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
        # Import the servers action and models
        from mcp_cli.commands.actions.servers import servers_action_async
        from mcp_cli.commands.models import ServerActionParams

        # Extract parameters for the existing implementation
        detailed = kwargs.get("detailed", False)
        show_raw = kwargs.get("raw", False)
        ping_servers = kwargs.get("ping", False)
        output_format = kwargs.get("format", "table")

        # Check if there are additional arguments for management commands
        args = kwargs.get("args", [])

        try:
            # Create Pydantic model from parameters
            params = ServerActionParams(
                args=args if args else [],
                detailed=detailed,
                show_capabilities=show_raw,
                output_format=output_format,
                ping_servers=ping_servers,
            )

            # Use the existing enhanced implementation
            # It handles all the display internally
            server_info = await servers_action_async(params)

            # The existing implementation handles all output directly via output.print
            # Just return success
            return CommandResult(success=True, data=server_info)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to execute server command: {str(e)}",
            )
