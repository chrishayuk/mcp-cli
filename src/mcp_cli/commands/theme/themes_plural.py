# src/mcp_cli/commands/definitions/themes_plural.py
"""
Plural themes command - lists all available themes.
"""

from __future__ import annotations


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
    def aliases(self) -> list[str]:
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
        from chuk_term.ui import output, format_table
        from mcp_cli.utils.preferences import get_preference_manager

        try:
            pref_manager = get_preference_manager()
            current_theme = pref_manager.get_theme()

            # Define available themes with descriptions
            themes = [
                {"name": "default", "description": "Default balanced theme"},
                {"name": "dark", "description": "Dark mode theme"},
                {"name": "light", "description": "Light mode theme"},
                {"name": "minimal", "description": "Minimal styling"},
                {"name": "terminal", "description": "Classic terminal colors"},
                {"name": "monokai", "description": "Monokai color scheme"},
                {"name": "dracula", "description": "Dracula color scheme"},
                {"name": "solarized", "description": "Solarized color scheme"},
            ]

            # Build table data
            table_data = []
            for theme in themes:
                is_current = "✓" if theme["name"] == current_theme else ""
                table_data.append(
                    {
                        "Current": is_current,
                        "Theme": theme["name"],
                        "Description": theme["description"],
                    }
                )

            # Display table
            table = format_table(
                table_data,
                title=f"{len(themes)} Available Themes",
                columns=["Current", "Theme", "Description"],
            )
            output.print_table(table)

            output.tip("Use: /theme <name> to switch themes")

            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to list themes: {str(e)}"
            )
