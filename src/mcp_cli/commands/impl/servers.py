# src/mcp_cli/commands/impl/servers.py
"""
Unified servers command implementation.

This single implementation works across all modes (chat, CLI, interactive).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context
from chuk_term.ui import output, format_table


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
  /servers              - List all servers (chat mode)
  servers               - List all servers (interactive mode)
  mcp-cli servers       - List all servers (CLI mode)
  
Options:
  --detailed            - Show detailed server information
  --format [table|json] - Output format (default: table)

Examples:
  /servers --detailed
  servers --format json
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
        # Import the existing enhanced servers implementation
        from mcp_cli.commands.servers import servers_action_async
        
        # Extract parameters for the existing implementation
        detailed = kwargs.get("detailed", False)
        show_raw = kwargs.get("raw", False)
        
        try:
            # Use the existing enhanced implementation
            # It handles all the display internally
            server_info = await servers_action_async(
                detailed=detailed,
                show_capabilities=detailed,
                show_transport=detailed,
                output_format="json" if show_raw else "table"
            )
            
            # The existing implementation handles all output directly via output.print
            # Just return success
            return CommandResult(
                success=True,
                data=server_info
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get server information: {str(e)}",
            )
