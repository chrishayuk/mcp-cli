# src/mcp_cli/commands/definitions/playbook.py
"""
Unified playbook command implementation.

This single implementation works across all modes (chat, CLI, interactive).
"""

from __future__ import annotations

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)


class PlaybookCommand(UnifiedCommand):
    """Manage playbook integration and query playbooks."""

    @property
    def name(self) -> str:
        return "playbook"

    @property
    def aliases(self) -> list[str]:
        return ["pb"]

    @property
    def description(self) -> str:
        return "Manage playbook integration and query procedural knowledge"

    @property
    def help_text(self) -> str:
        return """
Manage playbook integration (for querying playbooks, just ask the LLM).

Usage:
  /playbook                    - Show playbook status
  /playbook enable             - Enable playbook integration
  /playbook disable            - Disable playbook integration
  /playbook list               - List all available playbooks

Examples:
  /playbook                    - Show current playbook settings
  /playbook enable             - Enable playbook server binding
  /playbook disable            - Disable playbook server binding
  /playbook list               - Browse available playbooks

Note:
  - To QUERY playbooks, just ask the LLM (e.g., "How do I get sunset times?")
  - The LLM will automatically use the query_playbook tool when appropriate
  - Enabling/disabling playbook requires restarting the session
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="action",
                type=str,
                default=None,
                help="Action: enable, disable, list, or status",
                choices=["enable", "disable", "list", "status"],
                required=False,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the playbook command."""
        from mcp_cli.commands.actions.playbook import playbook_action_async
        from mcp_cli.commands.models import PlaybookActionParams

        # Extract arguments
        args = kwargs.get("args", [])

        # Determine action from first arg if not specified
        action = kwargs.get("action")
        if not action and args:
            action = args[0]
            args = args[1:]

        # Default to status if no action
        if not action:
            action = "status"

        try:
            # Create params model
            params = PlaybookActionParams(
                action=action,
                args=args,
            )

            # Execute action
            result = await playbook_action_async(params)

            return CommandResult(
                success=True,
                output=result if isinstance(result, str) else None,
                data=result if not isinstance(result, str) else None,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Error executing playbook command: {e}",
            )
