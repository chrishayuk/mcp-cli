# src/mcp_cli/commands/definitions/ping.py
"""
Unified ping command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context


class PingCommand(UnifiedCommand):
    """Test connectivity to MCP servers."""

    @property
    def name(self) -> str:
        return "ping"

    @property
    def aliases(self) -> List[str]:
        return []

    @property
    def description(self) -> str:
        return "Test connectivity to MCP servers"

    @property
    def help_text(self) -> str:
        return """
Test connectivity to MCP servers.

Usage:
  /ping [server_index]    - Test server connectivity (chat mode)
  ping [server_index]     - Test server connectivity (interactive mode)
  mcp-cli ping            - Test all servers (CLI mode)
  
Options:
  --all     - Test all servers
  --timeout - Timeout in seconds (default: 5)

Examples:
  /ping           - Test all servers
  /ping 0         - Test first server
  ping --all      - Test all servers
"""

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="server_index",
                type=int,
                required=False,
                help="Index of server to ping (omit for all)",
            ),
            CommandParameter(
                name="all",
                type=bool,
                default=False,
                help="Test all servers",
                is_flag=True,
            ),
            CommandParameter(
                name="timeout",
                type=float,
                default=5.0,
                help="Timeout in seconds",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the ping command."""
        # Import the ping action from the actions module
        from mcp_cli.commands.actions.ping import ping_action_async

        # Get tool manager
        tool_manager = kwargs.get("tool_manager")
        if not tool_manager:
            try:
                context = get_context()
                if context:
                    tool_manager = context.tool_manager
            except Exception:
                pass

        if not tool_manager:
            return CommandResult(
                success=False,
                error="No active tool manager. Please connect to a server first.",
            )

        # Get parameters
        server_index = kwargs.get("server_index")
        targets = []

        # Handle positional argument
        if server_index is None and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                # Could be server names or indices
                targets = args_val
            elif isinstance(args_val, str):
                targets = [args_val]
        elif server_index is not None:
            targets = [str(server_index)]

        try:
            # Use the existing ping action
            success = await ping_action_async(tool_manager, targets=targets)

            return CommandResult(success=success)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to ping servers: {str(e)}",
            )
