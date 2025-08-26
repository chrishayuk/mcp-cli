"""Theme command for interactive mode."""

from typing import List, Optional, Any

from chuk_term.ui import output
from chuk_term.ui.theme import set_theme
from chuk_term.ui.prompts import ask

from mcp_cli.utils.preferences import get_preference_manager, Theme
from .base import InteractiveCommand


class ThemeCommand(InteractiveCommand):
    """Interactive command for theme management."""

    def __init__(self):
        super().__init__(
            name="theme",
            help_text="Manage UI themes (theme = select, theme <name> = switch, theme list = show all)",
            aliases=[],
        )

    async def execute(
        self, args: List[str], tool_manager: Any = None, **kwargs
    ) -> Optional[bool]:
        """Execute the theme command."""
        theme_command(args)
        return None  # Don't exit interactive mode


def theme_command(args: List[str]) -> None:
    """Handle theme command in interactive mode.

    Args:
        args: Command arguments
    """
    pref_manager = get_preference_manager()

    # If no args or "select", use interactive selection
    if not args or (args and args[0].lower() in ["select"]):
        interactive_theme_selection(pref_manager)
        return

    # Handle "list" or "show" to display current theme
    if args[0].lower() in ["list", "show"]:
        current = pref_manager.get_theme()
        output.rule("Theme Settings")
        output.print(f"Current theme: [bold cyan]{current}[/bold cyan]")
        output.print()
        output.print("Available themes:")

        themes = [t.value for t in Theme]
        theme_descriptions = {
            "default": "Balanced colors for all terminals",
            "dark": "Dark mode with muted colors",
            "light": "Light mode with bright colors",
            "minimal": "Minimal color usage",
            "terminal": "Uses terminal's default colors",
            "monokai": "Popular dark theme from code editors",
            "dracula": "Dark theme with purple accents",
            "solarized": "Precision colors for readability",
        }

        for theme in themes:
            desc = theme_descriptions.get(theme, "")
            if theme == current:
                output.print(f"  • [bold cyan]{theme}[/bold cyan] (current) - {desc}")
            else:
                output.print(f"  • {theme} - {desc}")

        output.print()
        output.hint("Use 'theme <name>' to switch themes")
        return

    theme_arg = args[0].lower()

    # Handle direct theme switching
    valid_themes = [t.value for t in Theme]
    if theme_arg in valid_themes:
        try:
            # Apply theme immediately
            set_theme(theme_arg)

            # Save preference
            pref_manager.set_theme(theme_arg)

            output.success(f"Theme switched to: {theme_arg}")
            output.print("\nTheme saved to your preferences.")

        except Exception as e:
            output.error(f"Failed to switch theme: {e}")
    else:
        output.error(f"Invalid theme: {theme_arg}")
        output.hint(f"Valid themes are: {', '.join(valid_themes)}")
        output.hint("Use 'theme' for interactive selection")


def interactive_theme_selection(pref_manager) -> None:
    """Interactive theme selection with preview."""

    themes = [t.value for t in Theme]
    current = pref_manager.get_theme()

    # Theme descriptions
    theme_descriptions = {
        "default": "Balanced colors for all terminals",
        "dark": "Dark mode with muted colors",
        "light": "Light mode with bright colors",
        "minimal": "Minimal color usage",
        "terminal": "Uses terminal's default colors",
        "monokai": "Popular dark theme from code editors",
        "dracula": "Dark theme with purple accents",
        "solarized": "Precision colors for readability",
    }

    # Display themes in a nice table format
    from rich.table import Table

    output.rule("Theme Selector")

    table = Table(title="Available Themes", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Theme", style="green", width=12)
    table.add_column("Description", style="white")
    table.add_column("Status", justify="center", width=10)

    for idx, theme in enumerate(themes, 1):
        desc = theme_descriptions.get(theme, "")
        status = "✓ Current" if theme == current else ""
        status_style = "bold green" if theme == current else "dim"

        table.add_row(
            str(idx), theme, desc, f"[{status_style}]{status}[/{status_style}]"
        )

    output.print(table)
    output.print()

    # Get numeric or name input
    current_idx = themes.index(current) + 1 if current in themes else 1
    response = ask(
        "Enter theme number (1-8) or name:", default=str(current_idx), show_default=True
    )

    # Parse the response - could be number or theme name
    theme_name = None
    if response.isdigit():
        idx = int(response)
        if 1 <= idx <= len(themes):
            theme_name = themes[idx - 1]
        else:
            output.error(f"Invalid selection: {idx}. Please choose 1-{len(themes)}")
            return
    else:
        # Try to match theme name
        response_lower = response.lower()
        for theme in themes:
            if theme.lower() == response_lower:
                theme_name = theme
                break

        if not theme_name:
            output.error(f"Unknown theme: {response}")
            output.hint(f"Valid themes: {', '.join(themes)}")
            return

    if theme_name and theme_name != current:
        # Apply and save theme
        set_theme(theme_name)
        pref_manager.set_theme(theme_name)

        output.print()  # Add spacing
        output.rule(f"Theme: {theme_name}")
        output.success(f"Theme switched to: {theme_name}")
        output.print()

        # Show concise preview
        show_theme_preview()
        output.print()
        output.print("Theme saved to your preferences.")
    elif theme_name == current:
        output.info(f"Already using theme: {theme_name}")


def show_theme_preview() -> None:
    """Show a preview of the current theme."""
    output.print("[bold]Preview:[/bold]")
    output.info("Information")
    output.success("Success")
    output.warning("Warning")
    output.error("Error")
    output.hint("Hint")
