"""
Unified theme system for MCP CLI.

Centralizes all colors, styles, and visual elements for consistent UI appearance.
Supports theme switching and customization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ColorScheme:
    """Base color scheme definition."""
    
    # Status colors
    success: str = "green"
    error: str = "red"
    warning: str = "yellow"
    info: str = "cyan"
    debug: str = "dim"
    
    # Text styles
    normal: str = "white"
    emphasis: str = "bold"
    dim: str = "dim"
    italic: str = "italic"
    
    # UI element colors
    primary: str = "cyan"
    secondary: str = "blue"
    accent: str = "magenta"
    
    # Semantic colors
    user: str = "yellow"
    assistant: str = "blue"
    tool: str = "magenta"
    system: str = "dim white"
    
    # Component colors
    border: str = "yellow"
    title: str = "bold cyan"
    subtitle: str = "dim"
    prompt: str = "bold cyan"
    
    # Special purpose
    code: str = "green"
    link: str = "blue underline"
    highlight: str = "bold yellow"


@dataclass
class Icons:
    """Icon/symbol definitions for the UI."""
    
    # Status icons
    success: str = "✓"
    error: str = "✗"
    warning: str = "⚠"
    info: str = "ℹ"
    debug: str = "🔍"
    
    # Action icons
    prompt: str = ">"
    loading: str = "⚡"
    spinner: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    
    # UI elements
    bullet: str = "•"
    arrow: str = "→"
    check: str = "✓"
    cross: str = "✗"
    star: str = "★"
    
    # Mode indicators
    chat: str = "💬"
    interactive: str = "⚡"
    diagnostic: str = "🔍"
    
    # Special
    robot: str = "🤖"
    user: str = "👤"
    tool: str = "🔧"
    folder: str = "📁"
    file: str = "📄"


class Theme:
    """
    Main theme class that combines colors, icons, and styles.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize theme.
        
        Args:
            name: Theme name (default, dark, light, minimal, terminal)
        """
        self.name = name
        self.colors = self._load_color_scheme(name)
        self.icons = self._load_icons(name)
        self._style_cache: Dict[str, str] = {}
    
    def _load_color_scheme(self, name: str) -> ColorScheme:
        """
        Load a color scheme by name.
        
        Args:
            name: Theme name
            
        Returns:
            ColorScheme instance
        """
        if name == "dark":
            return DarkColorScheme()
        elif name == "light":
            return LightColorScheme()
        elif name == "minimal":
            return MinimalColorScheme()
        elif name == "terminal":
            return TerminalColorScheme()
        else:
            return ColorScheme()  # Default
    
    def _load_icons(self, name: str) -> Icons:
        """
        Load icons for a theme.
        
        Args:
            name: Theme name
            
        Returns:
            Icons instance
        """
        if name == "minimal" or name == "terminal":
            return MinimalIcons()
        else:
            return Icons()  # Default with emojis
    
    def style(self, *elements: str) -> str:
        """
        Build a Rich style string from theme elements.
        
        Args:
            *elements: Style elements to combine
            
        Returns:
            Rich style string
            
        Example:
            theme.style("error", "emphasis") -> "red bold"
        """
        # Check cache
        cache_key = "|".join(elements)
        if cache_key in self._style_cache:
            return self._style_cache[cache_key]
        
        # Build style
        styles = []
        for element in elements:
            if hasattr(self.colors, element):
                styles.append(getattr(self.colors, element))
            else:
                styles.append(element)  # Pass through unknown styles
        
        result = " ".join(styles)
        self._style_cache[cache_key] = result
        return result
    
    def format(self, text: str, *style_elements: str) -> str:
        """
        Format text with theme styles.
        
        Args:
            text: Text to format
            *style_elements: Style elements to apply
            
        Returns:
            Formatted text with Rich markup
            
        Example:
            theme.format("Error!", "error", "emphasis") -> "[red bold]Error![/]"
        """
        style = self.style(*style_elements)
        return f"[{style}]{text}[/]"
    
    def is_minimal(self) -> bool:
        """Check if this is a minimal theme (no decorations)."""
        return self.name in ("minimal", "terminal")
    
    def should_show_banners(self) -> bool:
        """Check if decorative banners should be shown."""
        return self.name not in ("minimal", "terminal")
    
    def should_show_icons(self) -> bool:
        """Check if icons/emojis should be shown."""
        return self.name not in ("minimal", "terminal")
    
    def should_show_boxes(self) -> bool:
        """Check if boxes/panels should be shown."""
        return self.name != "minimal"  # Terminal can have simple boxes
    
    def get_component_style(self, component: str) -> Dict[str, str]:
        """
        Get style dictionary for a UI component.
        
        Args:
            component: Component name (panel, table, prompt, etc.)
            
        Returns:
            Style dictionary for the component
        """
        # Check if we should include icons in titles
        show_icons = self.should_show_icons()
        
        styles = {
            "panel": {
                "border_style": self.colors.border,
                "title_style": self.colors.title,
                "subtitle_style": self.colors.subtitle,
            },
            "table": {
                "header_style": self.colors.emphasis,
                "row_style": self.colors.normal,
                "title_style": self.colors.title,
            },
            "prompt": {
                "prompt_style": self.colors.prompt,
                "default_style": self.colors.dim,
            },
            "user_message": {
                "border_style": self.colors.user,
                "title": f"{self.icons.user} You" if (show_icons and self.icons.user) else "You",
            },
            "assistant_message": {
                "border_style": self.colors.assistant,
                "title": f"{self.icons.robot} Assistant" if (show_icons and self.icons.robot) else "Assistant",
            },
            "tool_call": {
                "border_style": self.colors.tool,
                "title": f"{self.icons.tool} Tool Invocation" if (show_icons and self.icons.tool) else "Tool Invocation",
            },
        }
        return styles.get(component, {})


class DarkColorScheme(ColorScheme):
    """Dark theme color scheme."""
    
    def __init__(self):
        super().__init__(
            success="bright_green",
            error="bright_red",
            warning="bright_yellow",
            info="bright_cyan",
            normal="bright_white",
            primary="bright_cyan",
            secondary="bright_blue",
            accent="bright_magenta",
        )


class LightColorScheme(ColorScheme):
    """Light theme color scheme (for light terminals)."""
    
    def __init__(self):
        super().__init__(
            success="dark_green",
            error="dark_red",
            warning="dark_goldenrod",
            info="dark_cyan",
            normal="black",
            dim="grey50",
            primary="dark_cyan",
            secondary="dark_blue",
            accent="dark_magenta",
        )


class MinimalColorScheme(ColorScheme):
    """Minimal color scheme with no colors and plain output."""
    
    def __init__(self):
        super().__init__(
            # No colors - all white/default
            success="white",
            error="white",
            warning="white",
            info="white",
            normal="white",
            primary="white",
            secondary="white",
            accent="white",
            user="white",
            assistant="white",
            tool="white",
            border="white",
            title="white",
            subtitle="white",
            prompt="white",
            # No emphasis
            emphasis="",
            dim="",
            italic="",
        )


class TerminalColorScheme(ColorScheme):
    """Terminal color scheme using only basic ANSI colors."""
    
    def __init__(self):
        super().__init__(
            success="green",
            error="red",
            warning="yellow",
            info="blue",
            normal="white",
            primary="cyan",
            secondary="blue",
            accent="magenta",
            user="yellow",
            assistant="blue",
            tool="magenta",
            border="white",
            title="white bold",
            subtitle="white",
            prompt="white bold",
            emphasis="bold",
            dim="",
            italic="",
        )


@dataclass
class MinimalIcons:
    """Minimal icons - just basic ASCII."""
    
    # Status icons - plain text or empty
    success: str = ""
    error: str = ""
    warning: str = ""
    info: str = ""
    debug: str = ""
    
    # Action icons - simple ASCII
    prompt: str = ">"
    loading: str = "..."
    spinner: str = "-\\|/"
    
    # UI elements
    bullet: str = "-"
    arrow: str = "->"
    check: str = "[x]"
    cross: str = "[ ]"
    star: str = "*"
    
    # Mode indicators - no emojis
    chat: str = ""
    interactive: str = ""
    diagnostic: str = ""
    
    # Special - no emojis
    robot: str = ""
    user: str = ""
    tool: str = ""
    folder: str = ""
    file: str = ""


# ─────────────────────────── Global Theme Instance ─────────────────────────

_theme: Optional[Theme] = None


def get_theme() -> Theme:
    """
    Get the global theme instance.
    
    Returns:
        Current theme
    """
    global _theme
    if _theme is None:
        _theme = Theme()
    return _theme


def set_theme(theme_name: str) -> Theme:
    """
    Set the global theme.
    
    Args:
        theme_name: Name of theme to use
        
    Returns:
        New theme instance
    """
    global _theme
    _theme = Theme(theme_name)
    # Notify output system of theme change if it exists
    try:
        from mcp_cli.ui.output import get_output
        output = get_output()
        if hasattr(output, 'set_theme'):
            output.set_theme(_theme)
    except ImportError:
        pass  # Output module not available yet
    return _theme


def use_theme(theme: Theme) -> None:
    """
    Use a custom theme instance.
    
    Args:
        theme: Theme instance to use
    """
    global _theme
    _theme = theme
    # Notify output system of theme change if it exists
    try:
        from mcp_cli.ui.output import get_output
        output = get_output()
        if hasattr(output, 'set_theme'):
            output.set_theme(_theme)
    except ImportError:
        pass  # Output module not available yet


# ─────────────────────────── Legacy Compatibility ──────────────────────────

# Map old color constants to theme access for backward compatibility
def __getattr__(name: str) -> str:
    """Provide backward compatibility for old color constants."""
    theme = get_theme()
    
    # Map old names to new theme attributes
    mappings = {
        "BORDER_PRIMARY": theme.colors.border,
        "BORDER_SECONDARY": theme.colors.secondary,
        "TEXT_NORMAL": theme.colors.normal,
        "TEXT_EMPHASIS": theme.colors.emphasis,
        "TEXT_DEEMPHASIS": theme.colors.dim,
        "TEXT_SUCCESS": theme.colors.success,
        "TEXT_ERROR": theme.colors.error,
        "TEXT_WARNING": theme.colors.warning,
        "TEXT_INFO": theme.colors.info,
        "TEXT_HINT": theme.style("info", "dim", "italic"),
        "PANEL_DEFAULT": "default",
        "SERVER_COLOR": theme.colors.primary,
        "TOOL_COUNT_COLOR": theme.colors.success,
        "STATUS_COLOR": theme.colors.warning,
        "TITLE_COLOR": theme.colors.title,
        "USER_COLOR": theme.colors.user,
        "ASSISTANT_COLOR": theme.colors.assistant,
        "TOOL_COLOR": theme.colors.tool,
    }
    
    if name in mappings:
        return mappings[name]
    
    raise AttributeError(f"module {__name__} has no attribute {name}")


# ─────────────────────────── Convenience Functions ─────────────────────────

def apply_theme_to_output(output_instance) -> None:
    """
    Apply current theme to an Output instance.
    
    Args:
        output_instance: Output instance to configure
    """
    theme = get_theme()
    
    # Update Output's internal theme
    if hasattr(output_instance, 'set_theme'):
        output_instance.set_theme(theme)
    elif hasattr(output_instance, '_theme'):
        output_instance._theme = theme
    
    # Could also update specific style attributes if needed


def get_style_for_level(level: str) -> str:
    """
    Get style for a log/output level.
    
    Args:
        level: Level name (success, error, warning, info, debug)
        
    Returns:
        Style string
    """
    theme = get_theme()
    return theme.style(level.lower())


def format_with_theme(text: str, component: str) -> str:
    """
    Format text according to component style.
    
    Args:
        text: Text to format
        component: Component type
        
    Returns:
        Formatted text
    """
    theme = get_theme()
    
    component_map = {
        "success": ("success", "emphasis"),
        "error": ("error", "emphasis"),
        "warning": ("warning",),
        "info": ("info",),
        "tool": ("tool", "emphasis"),
        "user": ("user", "emphasis"),
        "assistant": ("assistant", "emphasis"),
    }
    
    if component in component_map:
        return theme.format(text, *component_map[component])
    
    return text