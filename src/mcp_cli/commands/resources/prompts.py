# src/mcp_cli/commands/definitions/prompts.py
"""
Unified prompts command implementation.
"""

from __future__ import annotations


from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output


class PromptsCommand(UnifiedCommand):
    """List and manage MCP prompts."""

    @property
    def name(self) -> str:
        return "prompts"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "List and manage MCP prompts"

    @property
    def help_text(self) -> str:
        return """
List and manage MCP prompts from connected servers.

Usage:
  /prompts [options]      - List prompts (chat mode)
  prompts [options]       - List prompts (interactive mode)
  mcp-cli prompts         - List prompts (CLI mode)
  
Options:
  --server <index>  - Show prompts from specific server
  --raw             - Output as JSON
  --get <name>      - Get a specific prompt

Examples:
  /prompts                - List all prompts
  /prompts --server 0     - List prompts from first server
  prompts --raw           - Output as JSON
  prompts --get "summarize" - Get the summarize prompt
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="server",
                type=int,
                required=False,
                help="Server index to list prompts from",
            ),
            CommandParameter(
                name="raw",
                type=bool,
                default=False,
                help="Output as JSON",
                is_flag=True,
            ),
            CommandParameter(
                name="get",
                type=str,
                required=False,
                help="Get a specific prompt by name",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the prompts command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import format_table

        try:
            # Get context and tool manager
            context = get_context()
            if not context or not context.tool_manager:
                return CommandResult(
                    success=False,
                    error="No tool manager available. Please connect to a server first.",
                )

            # Get prompts from tool manager
            prompts = await context.tool_manager.list_prompts()

            if not prompts:
                return CommandResult(
                    success=True,
                    output="No prompts available.",
                )

            # Build table data
            table_data = []
            for prompt in prompts:
                table_data.append(
                    {
                        "Name": prompt.name,
                        "Description": prompt.description or "No description",
                    }
                )

            # Display table
            table = format_table(
                table_data,
                title=f"{len(prompts)} Available Prompts",
                columns=["Name", "Description"],
            )
            output.print_table(table)

            return CommandResult(success=True, data=prompts)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list prompts: {str(e)}",
            )
