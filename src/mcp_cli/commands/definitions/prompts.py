# src/mcp_cli/commands/definitions/prompts.py
"""
Unified prompts command implementation.
"""

from __future__ import annotations

from typing import List

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
    def aliases(self) -> List[str]:
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
    def parameters(self) -> List[CommandParameter]:
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
        # Import the existing prompts implementation
        from mcp_cli.commands.actions.prompts import prompts_action_async

        try:
            # Use the existing enhanced implementation
            # It handles all the display internally
            prompts = await prompts_action_async()

            # Add count info if we have prompts
            if prompts:
                output.print(f"\nTotal prompts: {len(prompts)}")

            # The existing implementation handles all output directly
            # Just return success
            return CommandResult(success=True, data=prompts)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list prompts: {str(e)}",
            )
