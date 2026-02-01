# src/mcp_cli/commands/definitions/resources.py
"""
Unified resources command implementation.
"""

from __future__ import annotations


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
    def aliases(self) -> list[str]:
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
    def parameters(self) -> list[CommandParameter]:
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
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table

        try:
            # Get context and tool manager
            context = get_context()
            if not context or not context.tool_manager:
                return CommandResult(
                    success=False,
                    error="No tool manager available. Please connect to a server first.",
                )

            # Get resources from tool manager
            resources = await context.tool_manager.list_resources()

            if not resources:
                return CommandResult(
                    success=True,
                    output="No resources available.",
                )

            # Build table data
            table_data = []
            for resource in resources:
                # ResourceInfo has id, name, type, and extra dict
                # URI might be in extra or id field
                uri = resource.id or resource.extra.get("uri", "unknown")
                name = resource.name or "Unnamed"
                type_val = resource.type or resource.extra.get("mime_type", "unknown")

                table_data.append(
                    {
                        "URI": uri,
                        "Name": name,
                        "Type": type_val,
                    }
                )

            # Display table
            table = format_table(
                table_data,
                title=f"{len(resources)} Available Resources",
                columns=["URI", "Name", "Type"],
            )
            output.print_table(table)

            return CommandResult(success=True, data=resources)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list resources: {str(e)}",
            )
