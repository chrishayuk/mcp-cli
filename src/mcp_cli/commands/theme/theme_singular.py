# src/mcp_cli/commands/definitions/theme_singular.py
"""
Singular theme command - shows current theme with preview.
"""

from __future__ import annotations


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
    def aliases(self) -> list[str]:
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
        from chuk_term.ui import output
        from chuk_term.ui.theme import get_theme, set_theme
        from mcp_cli.utils.preferences import get_preference_manager

        # Get args
        args = kwargs.get("args", [])

        if not args:
            # No arguments - show current theme with preview
            try:
                current_theme = get_theme()
                pref_manager = get_preference_manager()
                theme_name = pref_manager.get_theme()

                # Display current theme in a panel using theme defaults
                output.panel(
                    f"Current theme: {theme_name}\n"
                    f"Description: {getattr(current_theme, 'description', 'No description')}",
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
                theme_name = args[0] if isinstance(args, list) else str(args)

                # Available themes
                available_themes = [
                    "default",
                    "dark",
                    "light",
                    "minimal",
                    "terminal",
                    "monokai",
                    "dracula",
                    "solarized",
                ]

                if theme_name not in available_themes:
                    return CommandResult(
                        success=False,
                        error=f"Invalid theme: {theme_name}. Available themes: {', '.join(available_themes)}",
                    )

                # Apply theme
                set_theme(theme_name)

                # Save preference
                pref_manager = get_preference_manager()
                pref_manager.set_theme(theme_name)

                output.success(f"Switched to theme: {theme_name}")
                return CommandResult(success=True)
            except Exception as e:
                return CommandResult(
                    success=False, error=f"Failed to switch theme: {str(e)}"
                )
