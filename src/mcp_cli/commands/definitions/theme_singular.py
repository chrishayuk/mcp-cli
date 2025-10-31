# src/mcp_cli/commands/definitions/theme_singular.py
"""
Singular theme command - shows current theme with preview.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandResult,
)


class ThemeSingularCommand(UnifiedCommand):
    """Show current theme or switch themes."""

    @property
    def name(self) -> str:
        return "theme"

    @property
    def aliases(self) -> List[str]:
        return []  # No aliases for singular form

    @property
    def description(self) -> str:
        return "Show current theme or switch to a different theme"

    @property
    def help_text(self) -> str:
        return """
Show current theme with preview or switch to a different theme.

Usage:
  /theme              - Show current theme with preview
  /theme <name>       - Switch to a different theme
  
Examples:
  /theme              - Show current theme and how it looks
  /theme dark         - Switch to dark theme
  /theme light        - Switch to light theme
  /theme default      - Switch to default theme
"""

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the theme command."""
        from mcp_cli.commands.actions.theme import theme_action_async
        from chuk_term.ui import output
        from chuk_term.ui.theme import get_theme

        # Get args
        args = kwargs.get("args", [])

        if not args:
            # No arguments - show current theme with preview
            try:
                current_theme = get_theme()

                # Display current theme in a panel using theme defaults
                output.panel(
                    f"ℹ Current theme: {current_theme.name}\n"
                    f"ℹ Description: {current_theme.description if hasattr(current_theme, 'description') else 'No description'}",
                    title="Theme Status",
                )

                # Show theme preview
                output.print("\n[bold]Theme Preview:[/bold]")
                output.info("ℹ Information message")
                output.success("✓ Success message")
                output.warning("⚠ Warning message")
                output.error("✗ Error message")
                output.hint("💡 Hint message")
                output.tip("💡 Tip message")

                output.tip(
                    "Use: /theme <name> to switch  |  /themes to see all available themes"
                )

                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to show theme status: {str(e)}"
                )
        else:
            # Has arguments - theme name to switch to
            try:
                from mcp_cli.commands.models import ThemeActionParams

                # Get the theme name from args
                theme_name = args[0] if isinstance(args, list) else str(args)
                params = ThemeActionParams(theme_name=theme_name)
                await theme_action_async(params)
                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to switch theme: {str(e)}"
                )
