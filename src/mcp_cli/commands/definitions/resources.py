# src/mcp_cli/commands/definitions/resources.py
"""
Unified resources command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)


class ResourcesCommand(UnifiedCommand):
    """List available MCP resources."""

    @property
    def name(self) -> str:
        return "resources"

    @property
    def aliases(self) -> List[str]:
        return []

    @property
    def description(self) -> str:
        return "List available MCP resources"

    @property
    def help_text(self) -> str:
        return """
List available MCP resources from connected servers.

Usage:
  /resources [options]    - List resources (chat mode)
  resources [options]     - List resources (interactive mode)
  mcp-cli resources       - List resources (CLI mode)
  
Options:
  --server <index>  - Show resources from specific server
  --raw             - Output as JSON
  --uri <pattern>   - Filter by URI pattern

Examples:
  /resources              - List all resources
  /resources --server 0   - List resources from first server
  resources --raw         - Output as JSON
  resources --uri "file://*" - Filter file resources
"""

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="server",
                type=int,
                required=False,
                help="Server index to list resources from",
            ),
            CommandParameter(
                name="raw",
                type=bool,
                default=False,
                help="Output as JSON",
                is_flag=True,
            ),
            CommandParameter(
                name="uri",
                type=str,
                required=False,
                help="Filter by URI pattern",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the resources command."""
        # Import the resources action from the actions module
        from mcp_cli.commands.actions.resources import resources_action_async

        # The existing implementation doesn't take parameters,
        # so we'll use it as-is
        try:
            # Use the existing implementation
            # It handles all the display internally
            resources = await resources_action_async()

            # The existing implementation handles all output directly
            # Just return success
            return CommandResult(success=True, data=resources)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list resources: {str(e)}",
            )
