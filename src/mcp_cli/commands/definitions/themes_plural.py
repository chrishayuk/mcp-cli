# src/mcp_cli/commands/definitions/themes_plural.py
"""
Plural themes command - lists all available themes.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandResult,
)


class ThemesPluralCommand(UnifiedCommand):
    """List all available themes."""

    @property
    def name(self) -> str:
        return "themes"

    @property
    def aliases(self) -> List[str]:
        return []  # No aliases

    @property
    def description(self) -> str:
        return "List all available themes"

    @property
    def help_text(self) -> str:
        return """
List all available UI themes.

Usage:
  /themes             - List all available themes
  
Examples:
  /themes             - Show all themes with descriptions
"""

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the themes command."""
        from mcp_cli.commands.actions.theme import theme_action_async

        try:
            # Always show the list
            from mcp_cli.commands.models import ThemeActionParams

            params = ThemeActionParams()
            await theme_action_async(params)  # Empty args = show list
            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to list themes: {str(e)}"
            )
