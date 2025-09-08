# src/mcp_cli/commands/definitions/theme.py
"""
Unified theme command implementation.
"""

from __future__ import annotations

from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui.theme import set_theme


class ThemeCommand(UnifiedCommand):
    """Change the UI theme."""

    @property
    def name(self) -> str:
        return "theme"

    @property
    def aliases(self) -> List[str]:
        return ["themes"]

    @property
    def description(self) -> str:
        return "Change the UI theme"

    @property
    def help_text(self) -> str:
        return """
Change the UI theme for better visibility.

Usage:
  /theme [theme_name]    - Set theme (chat mode)
  theme [theme_name]     - Set theme (interactive mode)
  
Available themes:
  default  - Default balanced theme
  dark     - Dark mode theme
  light    - Light mode theme
  minimal  - Minimal styling
  terminal - Classic terminal colors
  monokai  - Monokai color scheme
  dracula  - Dracula color scheme
  solarized - Solarized color scheme

Examples:
  /theme           - Show current theme
  /theme dark      - Switch to dark theme
  theme minimal    - Switch to minimal theme
"""

    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="theme_name",
                type=str,
                required=False,
                help="Name of the theme to set",
                choices=[
                    "default",
                    "dark",
                    "light",
                    "minimal",
                    "terminal",
                    "monokai",
                    "dracula",
                    "solarized",
                ],
            ),
        ]

    @property
    def modes(self) -> CommandMode:
        """Theme is for chat and interactive modes only."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the theme command."""
        from mcp_cli.utils.preferences import get_preference_manager
        from mcp_cli.commands.actions.theme import _interactive_theme_selection

        pref_manager = get_preference_manager()
        theme_name = kwargs.get("theme_name")

        # Handle positional argument
        if not theme_name and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                theme_name = args_val[0] if args_val else None
            elif isinstance(args_val, str):
                theme_name = args_val

        if theme_name:
            # Set new theme
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
            pref_manager.set_theme(theme_name)

            return CommandResult(
                success=True,
                output=f"Theme changed to: {theme_name}",
            )
        else:
            # Show interactive theme selector
            try:
                await _interactive_theme_selection(pref_manager)
                return CommandResult(success=True)
            except Exception:
                # Fallback to showing current theme
                current_theme = pref_manager.get_theme()
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

                output_text = f"Current theme: {current_theme}\n"
                output_text += f"Available themes: {', '.join(available_themes)}"

                return CommandResult(
                    success=True,
                    output=output_text,
                )
