"""Unit tests for the theme system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mcp_cli.ui.theme import (
    Theme,
    ColorScheme,
    Icons,
    DarkColorScheme,
    LightColorScheme,
    MinimalColorScheme,
    TerminalColorScheme,
    MinimalIcons,
    get_theme,
    set_theme,
    use_theme,
    apply_theme_to_output,
    get_style_for_level,
    format_with_theme,
)


class TestColorScheme:
    """Test ColorScheme class."""
    
    def test_default_colors(self):
        """Test default color scheme initialization."""
        scheme = ColorScheme()
        
        # Status colors
        assert scheme.success == "green"
        assert scheme.error == "red"
        assert scheme.warning == "yellow"
        assert scheme.info == "cyan"
        assert scheme.debug == "dim"
        
        # Text styles
        assert scheme.normal == "white"
        assert scheme.emphasis == "bold"
        assert scheme.dim == "dim"
        assert scheme.italic == "italic"
        
        # UI element colors
        assert scheme.primary == "cyan"
        assert scheme.secondary == "blue"
        assert scheme.accent == "magenta"
        
        # Semantic colors
        assert scheme.user == "yellow"
        assert scheme.assistant == "blue"
        assert scheme.tool == "magenta"
        assert scheme.system == "dim white"
    
    def test_dark_color_scheme(self):
        """Test dark theme colors."""
        scheme = DarkColorScheme()
        
        assert scheme.success == "bright_green"
        assert scheme.error == "bright_red"
        assert scheme.warning == "bright_yellow"
        assert scheme.info == "bright_cyan"
        assert scheme.normal == "bright_white"
        assert scheme.primary == "bright_cyan"
        assert scheme.secondary == "bright_blue"
        assert scheme.accent == "bright_magenta"
    
    def test_light_color_scheme(self):
        """Test light theme colors."""
        scheme = LightColorScheme()
        
        assert scheme.success == "dark_green"
        assert scheme.error == "dark_red"
        assert scheme.warning == "dark_goldenrod"
        assert scheme.info == "dark_cyan"
        assert scheme.normal == "black"
        assert scheme.dim == "grey50"
        assert scheme.primary == "dark_cyan"
        assert scheme.secondary == "dark_blue"
        assert scheme.accent == "dark_magenta"
    
    def test_minimal_color_scheme(self):
        """Test minimal theme has no colors."""
        scheme = MinimalColorScheme()
        
        # All colors should be white (no color)
        assert scheme.success == "white"
        assert scheme.error == "white"
        assert scheme.warning == "white"
        assert scheme.info == "white"
        assert scheme.normal == "white"
        assert scheme.primary == "white"
        assert scheme.secondary == "white"
        assert scheme.accent == "white"
        
        # No emphasis styles
        assert scheme.emphasis == ""
        assert scheme.dim == ""
        assert scheme.italic == ""
    
    def test_terminal_color_scheme(self):
        """Test terminal theme uses basic ANSI colors."""
        scheme = TerminalColorScheme()
        
        # Basic ANSI colors only
        assert scheme.success == "green"
        assert scheme.error == "red"
        assert scheme.warning == "yellow"
        assert scheme.info == "blue"
        assert scheme.normal == "white"
        assert scheme.primary == "cyan"
        assert scheme.secondary == "blue"
        assert scheme.accent == "magenta"
        
        # Basic styles
        assert scheme.emphasis == "bold"
        assert scheme.dim == ""
        assert scheme.italic == ""


class TestIcons:
    """Test Icons class."""
    
    def test_default_icons(self):
        """Test default icons with emojis."""
        icons = Icons()
        
        # Status icons
        assert icons.success == "✓"
        assert icons.error == "✗"
        assert icons.warning == "⚠"
        assert icons.info == "ℹ"
        assert icons.debug == "🔍"
        
        # Mode indicators
        assert icons.chat == "💬"
        assert icons.interactive == "⚡"
        assert icons.diagnostic == "🔍"
        
        # Special icons
        assert icons.robot == "🤖"
        assert icons.user == "👤"
        assert icons.tool == "🔧"
        assert icons.folder == "📁"
        assert icons.file == "📄"
    
    def test_minimal_icons(self):
        """Test minimal icons are ASCII only."""
        icons = MinimalIcons()
        
        # Status icons should be empty or ASCII
        assert icons.success == ""
        assert icons.error == ""
        assert icons.warning == ""
        assert icons.info == ""
        assert icons.debug == ""
        
        # Action icons - simple ASCII
        assert icons.prompt == ">"
        assert icons.loading == "..."
        assert icons.spinner == "-\\|/"
        
        # UI elements - ASCII only
        assert icons.bullet == "-"
        assert icons.arrow == "->"
        assert icons.check == "[x]"
        assert icons.cross == "[ ]"
        assert icons.star == "*"
        
        # Mode indicators - no emojis
        assert icons.chat == ""
        assert icons.interactive == ""
        assert icons.diagnostic == ""
        
        # Special - no emojis
        assert icons.robot == ""
        assert icons.user == ""
        assert icons.tool == ""


class TestTheme:
    """Test Theme class."""
    
    def test_default_theme_initialization(self):
        """Test default theme initialization."""
        theme = Theme()
        
        assert theme.name == "default"
        assert isinstance(theme.colors, ColorScheme)
        assert isinstance(theme.icons, Icons)
        assert theme._style_cache == {}
    
    @pytest.mark.parametrize("theme_name,color_scheme_class,icons_class", [
        ("default", ColorScheme, Icons),
        ("dark", DarkColorScheme, Icons),
        ("light", LightColorScheme, Icons),
        ("minimal", MinimalColorScheme, MinimalIcons),
        ("terminal", TerminalColorScheme, MinimalIcons),
    ])
    def test_theme_loading(self, theme_name, color_scheme_class, icons_class):
        """Test different themes load correct color schemes and icons."""
        theme = Theme(theme_name)
        
        assert theme.name == theme_name
        assert isinstance(theme.colors, color_scheme_class)
        assert isinstance(theme.icons, icons_class)
    
    def test_style_building(self):
        """Test style string building."""
        theme = Theme()
        
        # Single style element
        style = theme.style("error")
        assert style == "red"
        
        # Multiple style elements
        style = theme.style("error", "emphasis")
        assert style == "red bold"
        
        # Unknown styles pass through
        style = theme.style("custom_style")
        assert style == "custom_style"
    
    def test_style_caching(self):
        """Test that styles are cached."""
        theme = Theme()
        
        # First call builds style
        style1 = theme.style("error", "emphasis")
        assert "error|emphasis" in theme._style_cache
        
        # Second call uses cache
        with patch.object(theme, '_style_cache', {"error|emphasis": "cached"}):
            style2 = theme.style("error", "emphasis")
            assert style2 == "cached"
    
    def test_format_text(self):
        """Test text formatting with theme styles."""
        theme = Theme()
        
        # Format with single style
        formatted = theme.format("Error!", "error")
        assert formatted == "[red]Error![/]"
        
        # Format with multiple styles
        formatted = theme.format("Important Error!", "error", "emphasis")
        assert formatted == "[red bold]Important Error![/]"
    
    def test_theme_capabilities(self):
        """Test theme capability checking methods."""
        # Default theme
        theme = Theme("default")
        assert not theme.is_minimal()
        assert theme.should_show_banners()
        assert theme.should_show_icons()
        assert theme.should_show_boxes()
        
        # Minimal theme
        theme = Theme("minimal")
        assert theme.is_minimal()
        assert not theme.should_show_banners()
        assert not theme.should_show_icons()
        assert not theme.should_show_boxes()
        
        # Terminal theme
        theme = Theme("terminal")
        assert theme.is_minimal()
        assert not theme.should_show_banners()
        assert not theme.should_show_icons()
        assert theme.should_show_boxes()  # Terminal can have simple boxes
    
    def test_get_component_style(self):
        """Test getting component-specific styles."""
        theme = Theme("default")
        
        # Panel styles
        panel_style = theme.get_component_style("panel")
        assert panel_style["border_style"] == theme.colors.border
        assert panel_style["title_style"] == theme.colors.title
        assert panel_style["subtitle_style"] == theme.colors.subtitle
        
        # Table styles
        table_style = theme.get_component_style("table")
        assert table_style["header_style"] == theme.colors.emphasis
        assert table_style["row_style"] == theme.colors.normal
        assert table_style["title_style"] == theme.colors.title
        
        # User message styles
        user_style = theme.get_component_style("user_message")
        assert user_style["border_style"] == theme.colors.user
        assert "You" in user_style["title"]
        
        # Assistant message styles
        assistant_style = theme.get_component_style("assistant_message")
        assert assistant_style["border_style"] == theme.colors.assistant
        assert "Assistant" in assistant_style["title"]
        
        # Tool call styles
        tool_style = theme.get_component_style("tool_call")
        assert tool_style["border_style"] == theme.colors.tool
        assert "Tool" in tool_style["title"]
    
    def test_component_style_icons(self):
        """Test that component styles include/exclude icons based on theme."""
        # Default theme should include icons
        theme = Theme("default")
        user_style = theme.get_component_style("user_message")
        assert "👤" in user_style["title"]
        
        # Minimal theme should not include icons
        theme = Theme("minimal")
        user_style = theme.get_component_style("user_message")
        assert "👤" not in user_style["title"]
        assert user_style["title"] == "You"
    
    def test_unknown_component_style(self):
        """Test getting style for unknown component returns empty dict."""
        theme = Theme()
        style = theme.get_component_style("unknown_component")
        assert style == {}


class TestGlobalThemeFunctions:
    """Test global theme management functions."""
    
    def test_get_theme_default(self):
        """Test get_theme returns default theme when none set."""
        # Reset global theme
        import mcp_cli.ui.theme
        mcp_cli.ui.theme._theme = None
        
        theme = get_theme()
        assert theme.name == "default"
        assert mcp_cli.ui.theme._theme is theme
    
    def test_set_theme(self):
        """Test setting global theme."""
        with patch('mcp_cli.ui.output.get_output') as mock_get_output:
            mock_output = Mock()
            mock_get_output.return_value = mock_output
            
            theme = set_theme("dark")
            
            assert theme.name == "dark"
            assert isinstance(theme.colors, DarkColorScheme)
            
            # Should notify output system if it has set_theme method
            if hasattr(mock_output, 'set_theme'):
                mock_output.set_theme.assert_called_once_with(theme)
    
    def test_set_theme_without_output(self):
        """Test setting theme when output module not available."""
        with patch('mcp_cli.ui.output.get_output', side_effect=ImportError):
            theme = set_theme("minimal")
            assert theme.name == "minimal"
            # Should not raise error
    
    def test_use_custom_theme(self):
        """Test using a custom theme instance."""
        custom_theme = Theme("custom")
        custom_theme.colors.primary = "purple"
        
        with patch('mcp_cli.ui.output.get_output') as mock_get_output:
            mock_output = Mock()
            mock_get_output.return_value = mock_output
            
            use_theme(custom_theme)
            
            # Global theme should be the custom instance
            import mcp_cli.ui.theme
            assert mcp_cli.ui.theme._theme is custom_theme
            
            # Should notify output system
            if hasattr(mock_output, 'set_theme'):
                mock_output.set_theme.assert_called_once_with(custom_theme)


class TestHelperFunctions:
    """Test theme helper functions."""
    
    def test_apply_theme_to_output(self):
        """Test applying theme to output instance."""
        theme = Theme("dark")
        mock_output = Mock()
        
        # With set_theme method
        mock_output.set_theme = Mock()
        apply_theme_to_output(mock_output)
        mock_output.set_theme.assert_called_once()
        
        # With _theme attribute
        mock_output = Mock(spec=[])
        mock_output._theme = None
        apply_theme_to_output(mock_output)
        assert mock_output._theme.name == get_theme().name
    
    def test_get_style_for_level(self):
        """Test getting style for log levels."""
        with patch('mcp_cli.ui.theme.get_theme') as mock_get_theme:
            mock_theme = Mock()
            mock_theme.style.return_value = "test_style"
            mock_get_theme.return_value = mock_theme
            
            style = get_style_for_level("error")
            
            mock_theme.style.assert_called_once_with("error")
            assert style == "test_style"
    
    @pytest.mark.parametrize("component,expected_styles", [
        ("success", ("success", "emphasis")),
        ("error", ("error", "emphasis")),
        ("warning", ("warning",)),
        ("info", ("info",)),
        ("tool", ("tool", "emphasis")),
        ("user", ("user", "emphasis")),
        ("assistant", ("assistant", "emphasis")),
    ])
    def test_format_with_theme(self, component, expected_styles):
        """Test formatting text with theme for different components."""
        with patch('mcp_cli.ui.theme.get_theme') as mock_get_theme:
            mock_theme = Mock()
            mock_theme.format.return_value = "[styled]text[/]"
            mock_get_theme.return_value = mock_theme
            
            result = format_with_theme("text", component)
            
            mock_theme.format.assert_called_once_with("text", *expected_styles)
            assert result == "[styled]text[/]"
    
    def test_format_with_theme_unknown_component(self):
        """Test formatting with unknown component returns unchanged text."""
        result = format_with_theme("text", "unknown_component")
        assert result == "text"


class TestLegacyCompatibility:
    """Test backward compatibility with old color constants."""
    
    def test_legacy_color_mapping(self):
        """Test that old color constants map to theme attributes."""
        import mcp_cli.ui.theme as theme_module
        
        # Set a known theme
        set_theme("default")
        theme = get_theme()
        
        # Test mappings
        assert theme_module.BORDER_PRIMARY == theme.colors.border
        assert theme_module.BORDER_SECONDARY == theme.colors.secondary
        assert theme_module.TEXT_NORMAL == theme.colors.normal
        assert theme_module.TEXT_EMPHASIS == theme.colors.emphasis
        assert theme_module.TEXT_SUCCESS == theme.colors.success
        assert theme_module.TEXT_ERROR == theme.colors.error
        assert theme_module.TEXT_WARNING == theme.colors.warning
        assert theme_module.TEXT_INFO == theme.colors.info
        assert theme_module.USER_COLOR == theme.colors.user
        assert theme_module.ASSISTANT_COLOR == theme.colors.assistant
        assert theme_module.TOOL_COLOR == theme.colors.tool
    
    def test_invalid_legacy_attribute(self):
        """Test that invalid attributes raise AttributeError."""
        import mcp_cli.ui.theme as theme_module
        
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = theme_module.INVALID_ATTRIBUTE


class TestThemeIntegration:
    """Integration tests for theme system."""
    
    def test_theme_switching_preserves_state(self):
        """Test that switching themes preserves application state."""
        # Start with default theme
        theme1 = set_theme("default")
        assert theme1.name == "default"
        
        # Switch to dark theme
        theme2 = set_theme("dark")
        assert theme2.name == "dark"
        assert theme2 is not theme1
        
        # Switch to minimal theme
        theme3 = set_theme("minimal")
        assert theme3.name == "minimal"
        assert theme3.is_minimal()
        assert not theme3.should_show_icons()
    
    def test_theme_affects_component_rendering(self):
        """Test that theme affects how components render."""
        # Default theme with icons
        set_theme("default")
        theme = get_theme()
        user_style = theme.get_component_style("user_message")
        assert theme.icons.user in user_style["title"]
        
        # Minimal theme without icons (empty string)
        set_theme("minimal")
        theme = get_theme()
        user_style = theme.get_component_style("user_message")
        # MinimalIcons.user is empty string, so title should be just "You"
        assert user_style["title"] == "You"
        assert "👤" not in user_style["title"]
    
    def test_custom_theme_creation(self):
        """Test creating and using a custom theme."""
        # Create custom color scheme
        class CustomColorScheme(ColorScheme):
            def __init__(self):
                super().__init__()
                self.primary = "purple"
                self.accent = "orange"
        
        # Create custom theme
        custom_theme = Theme("custom")
        custom_theme.colors = CustomColorScheme()
        
        # Use custom theme
        use_theme(custom_theme)
        
        # Verify custom colors are used
        current_theme = get_theme()
        assert current_theme.colors.primary == "purple"
        assert current_theme.colors.accent == "orange"
        assert current_theme.style("primary") == "purple"


@pytest.fixture(autouse=True)
def reset_global_theme():
    """Reset global theme after each test."""
    yield
    # Reset to None so next test starts fresh
    import mcp_cli.ui.theme
    mcp_cli.ui.theme._theme = None