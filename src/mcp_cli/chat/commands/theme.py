# mcp_cli/chat/commands/theme.py
"""Enhanced theme command with improved UI/UX for MCP-CLI."""

from typing import List

from chuk_term.ui import output, format_table
from chuk_term.ui.theme import set_theme, get_theme

from mcp_cli.utils.preferences import get_preference_manager, Theme
from mcp_cli.chat.commands import register_command


# Theme metadata with better descriptions and characteristics
THEME_INFO = {
    "default": {
        "name": "Default",
        "description": "Balanced colors for all terminals",
        "best_for": "General use, works everywhere",
        "style": "Professional",
        "preview": "🔵 Blue accents with balanced contrast",
    },
    "dark": {
        "name": "Dark",
        "description": "Dark mode with muted colors",
        "best_for": "Low-light environments",
        "style": "Modern & Easy on eyes",
        "preview": "⚫ Dark background with soft colors",
    },
    "light": {
        "name": "Light",
        "description": "Light mode with bright colors",
        "best_for": "Well-lit environments",
        "style": "Clean & Bright",
        "preview": "⚪ Light background with vibrant colors",
    },
    "minimal": {
        "name": "Minimal",
        "description": "Minimal color usage",
        "best_for": "Distraction-free work",
        "style": "Zen & Focused",
        "preview": "⬜ Black and white only",
    },
    "terminal": {
        "name": "Terminal",
        "description": "Uses terminal's default colors",
        "best_for": "Terminal theme integration",
        "style": "Native & Consistent",
        "preview": "💻 Inherits terminal colors",
    },
    "monokai": {
        "name": "Monokai",
        "description": "Popular dark theme from code editors",
        "best_for": "Developers & coders",
        "style": "Vibrant & Syntax-friendly",
        "preview": "🟣 Purple/green with high contrast",
    },
    "dracula": {
        "name": "Dracula",
        "description": "Dark theme with purple accents",
        "best_for": "Night owls & vampires",
        "style": "Gothic & Elegant",
        "preview": "🦇 Dark purple aesthetic",
    },
    "solarized": {
        "name": "Solarized",
        "description": "Precision colors for readability",
        "best_for": "Long reading sessions",
        "style": "Scientific & Precise",
        "preview": "🟠 Orange/blue optimized contrast",
    },
}


def get_theme_info(theme_key: str) -> dict:
    """Get theme info, dynamically building if needed."""
    # Get from static info if available
    if theme_key in THEME_INFO:
        return THEME_INFO[theme_key]

    # Build dynamically for any new themes
    return {
        "name": theme_key.title(),
        "description": f"Theme: {theme_key}",
        "best_for": "Custom theme",
        "style": "Custom",
        "preview": "🎨 Custom theme",
    }


def show_theme_card(theme_key: str, is_current: bool = False) -> None:
    """Display a theme card with information."""
    info = get_theme_info(theme_key)

    # Create card header - using theme-aware styling
    status = " ✓ [Current]" if is_current else ""
    output.rule(f"{info['name']} Theme{status}")

    # Display theme information
    output.print(f"Description: {info['description']}")
    output.print(f"Best for: {info['best_for']}")
    output.print(f"Style: {info['style']}")
    output.print(f"Preview: {info['preview']}")
    output.print()


def show_theme_preview(theme_name: str) -> None:
    """Show a preview of theme output styles."""
    output.print("Theme Preview:")
    output.info("ℹ️  Information message")
    output.success("✅ Success message")
    output.warning("⚠️  Warning message")
    output.error("❌ Error message")
    output.hint("💡 Hint message")
    output.print("📊 Regular output")
    output.print()


def show_theme_table(current_theme: str) -> None:
    """Display all available themes in a table format."""
    # Build data for chuk_term's format_table
    data = []

    # Get themes from enum instead of hardcoding
    themes = [t.value for t in Theme]

    for idx, theme in enumerate(themes, 1):
        is_current = theme == current_theme
        status = "✓ Current" if is_current else ""

        info = get_theme_info(theme)
        data.append(
            {
                "#": str(idx),
                "Theme": theme,
                "Description": info["description"],
                "Best For": info["best_for"],
                "Status": status,
            }
        )

    # Create and print table using chuk_term's format_table
    table = format_table(
        data=data,
        title="Available Themes",
        columns=["#", "Theme", "Description", "Best For", "Status"],
    )

    # Print table using chuk_term's output (which handles theming)
    output.print(table)
    output.print()


async def interactive_theme_selection(context, pref_manager) -> None:
    """Enhanced interactive theme selection with preview."""
    themes = [t.value for t in Theme]
    current = pref_manager.get_theme()

    # Show theme table
    output.rule("Theme Selector")
    show_theme_table(current)

    output.info(f"Current theme: {get_theme_info(current)['name']}")
    output.print()

    # Get selection
    from chuk_term.ui.prompts import ask

    response = ask(
        f"Enter theme number (1-{len(themes)}), name, or press Enter to keep current:",
        default="",
        show_default=False,
    )

    # Handle empty response
    if not response:
        output.info(f"Keeping current theme: {current}")
        return

    # Parse response
    theme_name = None
    if response.isdigit():
        idx = int(response)
        if 1 <= idx <= len(themes):
            theme_name = themes[idx - 1]
        else:
            output.error(f"Invalid number: {idx}. Please choose 1-{len(themes)}")
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
            output.hint(f"Available themes: {', '.join(themes)}")
            return

    # Check if it's the same theme
    if theme_name == current:
        output.info(f"Already using {theme_name} theme")
        return

    # Show comparison with actual theme previews
    output.print()
    output.rule("Theme Comparison")

    # Save current theme for restoration
    original_theme = get_theme()

    # Show current theme with its actual appearance
    output.print("Current Theme:")
    set_theme(current)  # Apply current theme temporarily
    show_theme_card(current, is_current=True)
    show_theme_preview(current)

    # Show new theme with its actual appearance
    output.print("New Theme:")
    set_theme(theme_name)  # Apply new theme temporarily to show how it looks
    show_theme_card(theme_name, is_current=False)
    show_theme_preview(theme_name)

    # Restore original theme for the prompt
    set_theme(original_theme)

    # Confirm change
    from chuk_term.ui.prompts import confirm

    if confirm(f"Switch to {theme_name} theme?", default=True):
        # Apply with animation
        output.print()
        output.print(f"Applying {theme_name} theme...")

        # Apply theme
        set_theme(theme_name)
        pref_manager.set_theme(theme_name)

        # Show preview
        output.print()
        output.rule(f"Theme: {theme_name}")
        show_theme_preview(theme_name)

        output.success(f"Theme switched to: {theme_name}")
        output.hint(get_theme_info(theme_name)["best_for"])
    else:
        output.info("Theme change cancelled")


async def handle_theme_command(context, args: List[str]) -> None:
    """Handle theme selection and management.

    Usage:
        /theme              - Interactive theme selector with table
        /theme <name>       - Switch to a specific theme directly
        /theme list         - Show all available themes
        /theme preview <n>  - Preview a theme without applying

    Args:
        context: Chat context
        args: Command arguments
    """
    pref_manager = get_preference_manager()

    # Get valid themes from enum
    valid_themes = [t.value for t in Theme]

    # Special commands
    if args and args[0].lower() == "list":
        # Show theme list in table format
        current = pref_manager.get_theme()
        output.rule("Available Themes")
        show_theme_table(current)
        return

    if args and args[0].lower() == "preview":
        # Preview a theme without applying
        if len(args) < 2:
            output.error("Usage: /theme preview <name>")
            return

        theme_name = args[1].lower()

        if theme_name not in valid_themes:
            output.error(f"Invalid theme: {theme_name}")
            output.hint(f"Valid themes: {', '.join(valid_themes)}")
            return

        # Save current theme
        current = get_theme()

        # Preview the theme
        output.rule(f"Preview: {theme_name}")
        set_theme(theme_name)
        show_theme_preview(theme_name)

        # Ask if they want to keep it
        output.print()
        from chuk_term.ui.prompts import confirm

        if confirm(f"Keep {theme_name} theme?", default=False):
            pref_manager.set_theme(theme_name)
            output.success(f"Theme saved: {theme_name}")
        else:
            # Revert to original
            set_theme(current)
            output.info(f"Reverted to {current}")
        return

    # Default behavior - interactive selection or direct setting
    if not args:
        await interactive_theme_selection(context, pref_manager)
    else:
        # Direct theme setting
        theme_arg = args[0].lower()

        if theme_arg in valid_themes:
            try:
                current = pref_manager.get_theme()
                if theme_arg == current:
                    output.info(f"Already using {theme_arg} theme")
                    return

                # Show what's changing
                output.rule("Switching Theme")
                output.print(f"From: {current} → To: {theme_arg}")
                output.print()

                # Apply theme
                set_theme(theme_arg)
                pref_manager.set_theme(theme_arg)

                # Show preview
                show_theme_preview(theme_arg)

                output.success(f"Theme switched to: {theme_arg}")
                output.hint(get_theme_info(theme_arg)["best_for"])

            except Exception as e:
                output.error(f"Failed to switch theme: {e}")
        else:
            output.error(f"Invalid theme: {theme_arg}")
            output.hint(f"Valid themes: {', '.join(valid_themes)}")
            output.hint("Use '/theme' for interactive selection")
            output.hint("Use '/theme list' to see all themes")
            output.hint("Use '/theme preview <name>' to preview without applying")


# Register the command handler
async def cmd_theme(parts: List[str]) -> bool:
    """Manage UI themes and color schemes.

    Usage:
        /theme              - Interactive theme selector with table
        /theme <name>       - Switch to a specific theme directly
        /theme list         - Show all available themes in table
        /theme preview <n>  - Preview a theme without applying

    Available themes are defined in the Theme enum.

    Examples:
        /theme              - Open interactive selector
        /theme dark         - Switch to dark theme
        /theme list         - View all themes in table
        /theme preview monokai - Preview monokai without applying

    Themes are persisted across sessions and affect the entire CLI experience.
    """
    # Use global context manager
    from mcp_cli.context import get_context

    context = get_context()

    # Extract arguments (skip the command itself)
    args = parts[1:] if len(parts) > 1 else []
    await handle_theme_command(context, args)
    return True


# Register the command
register_command("/theme", cmd_theme)

# Register the command help
THEME_HELP = {
    "name": "theme",
    "description": "Manage UI themes",
    "usage": [
        "/theme - Interactive theme selector with table",
        "/theme <name> - Switch to a specific theme directly",
        "/theme list - Show all available themes in table",
        "/theme preview <name> - Preview a theme without applying",
    ],
    "examples": ["/theme", "/theme dark", "/theme list", "/theme preview monokai"],
}
