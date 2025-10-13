# src/mcp_cli/ui/color_converter.py
"""
Color Converter
===============

Converts between Rich color names and prompt_toolkit color formats.
This ensures consistent color handling across different UI components.
"""

from typing import Optional


def rich_to_prompt_toolkit(color_str: Optional[str]) -> str:
    """
    Convert Rich color names to prompt_toolkit format.

    Args:
        color_str: Rich color string (e.g., "bright_white", "dim", "bold yellow")

    Returns:
        prompt_toolkit compatible color string

    Examples:
        >>> rich_to_prompt_toolkit("bright_white")
        "ansiwhite bold"
        >>> rich_to_prompt_toolkit("magenta")
        "ansimagenta"
        >>> rich_to_prompt_toolkit("dim")
        "ansibrightblack"
    """
    if not color_str or color_str == "dim":
        return "ansibrightblack"

    # Map Rich colors to prompt_toolkit ANSI colors
    # Note: prompt_toolkit doesn't have "bright" variants like ansibrightwhite
    # Instead, we use bold to achieve similar effect
    color_map = {
        # Basic colors
        "white": "ansiwhite",
        "black": "ansiblack",
        "red": "ansired",
        "green": "ansigreen",
        "yellow": "ansiyellow",
        "blue": "ansiblue",
        "magenta": "ansimagenta",
        "cyan": "ansicyan",
        # Bright colors (use bold modifier)
        "bright_white": "ansiwhite bold",
        "bright_black": "ansibrightblack",  # This one exists
        "bright_red": "ansired bold",
        "bright_green": "ansigreen bold",
        "bright_yellow": "ansiyellow bold",
        "bright_blue": "ansiblue bold",
        "bright_magenta": "ansimagenta bold",
        "bright_cyan": "ansicyan bold",
        # Dark colors (map to regular colors)
        "dark_red": "ansired",
        "dark_green": "ansigreen",
        "dark_yellow": "ansiyellow",
        "dark_blue": "ansiblue",
        "dark_magenta": "ansimagenta",
        "dark_cyan": "ansicyan",
        "dark_goldenrod": "ansiyellow",
        # Special colors
        "grey50": "ansibrightblack",
        "gray50": "ansibrightblack",
        "dim": "ansibrightblack",
        # Common Rich theme colors
        "default": "ansiwhite",
        "bold": "bold",
        "underline": "underline",
        "italic": "italic",
    }

    # Handle composite styles like "bold yellow"
    parts = color_str.split()

    # If it's a single word, look it up directly
    if len(parts) == 1:
        return color_map.get(color_str, "ansiwhite")

    # Handle composite styles
    result_parts = []
    for part in parts:
        if part in ["bold", "underline", "italic"]:
            result_parts.append(part)
        elif part in color_map:
            mapped = color_map[part]
            # Avoid duplicating modifiers
            if mapped not in result_parts:
                result_parts.append(mapped)
        else:
            # Try to extract just the color
            color = parts[-1]
            if color in color_map:
                result_parts.append(color_map[color])
            else:
                result_parts.append("ansiwhite")

    return " ".join(result_parts) if result_parts else "ansiwhite"


def create_transparent_completion_style(theme_colors, background_color="black"):
    """
    Create a prompt_toolkit Style for autocomplete menu that matches terminal background.

    Args:
        theme_colors: Theme ColorScheme object with color attributes
        background_color: Background color to use (default: "black" for dark terminals)

    Returns:
        Dict for Style.from_dict() constructor
    """
    # Determine background based on theme name if possible
    bg = f"bg:ansi{background_color}" if background_color else ""

    # Return a dict that explicitly overrides all completion styles
    return {
        # Base completion menu
        "completion-menu": f"{bg}",
        "completion-menu.completion": f"{bg} {rich_to_prompt_toolkit(theme_colors.accent)}",
        "completion-menu.completion.current": f"{bg} {rich_to_prompt_toolkit(theme_colors.highlight)} underline",
        # Meta/description column - explicitly set background
        "completion-menu.meta": f"{bg} {rich_to_prompt_toolkit(theme_colors.dim)}",
        "completion-menu.meta.current": f"{bg} {rich_to_prompt_toolkit(theme_colors.normal)}",
        "completion-menu.multi-column-meta": f"{bg} {rich_to_prompt_toolkit(theme_colors.dim)}",
        # Border and other elements
        "completion-menu.border": f"{bg}",
        "scrollbar.background": f"{bg}",
        "scrollbar.button": f"{bg}",
        # Auto-suggestion
        "auto-suggestion": rich_to_prompt_toolkit(theme_colors.dim),
    }
