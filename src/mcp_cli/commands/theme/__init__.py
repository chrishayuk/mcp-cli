"""Theme management commands."""

from mcp_cli.commands.theme.theme_singular import ThemeSingularCommand
from mcp_cli.commands.theme.themes_plural import ThemesPluralCommand
from mcp_cli.commands.theme.theme import ThemeCommand

__all__ = [
    "ThemeSingularCommand",
    "ThemesPluralCommand",
    "ThemeCommand",
]
