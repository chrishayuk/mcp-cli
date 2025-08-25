"""Unit tests for the theme system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from mcp_cli.ui.theme import (
    Theme,
    ColorScheme,
    Icons,
    get_theme,
    set_theme,
    reset_theme,
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
        
        # Component colors
        assert scheme.border == "yellow"
        assert scheme.title == "bold cyan"
        assert scheme.subtitle == "dim"
        assert scheme.prompt == "bold cyan"


class TestIcons:
    """Test Icons class."""
    
    def test_default_icons(self):
        """Test default icon set initialization."""
        icons = Icons()
        
        # Status icons
        assert icons.success == "✓"
        assert icons.error == "✗"
        assert icons.warning == "⚠"
        assert icons.info == "ℹ"
        assert icons.debug == "🔍"
        
        # Action icons
        assert icons.prompt == ">"
        assert icons.loading == "⚡"
        assert isinstance(icons.spinner, str)
        
        # UI elements
        assert icons.bullet == "•"
        assert icons.arrow == "→"
        assert icons.check == "✓"
        assert icons.cross == "✗"
        assert icons.star == "★"
        
        # Mode indicators
        assert icons.chat == "💬"
        assert icons.interactive == "⚡"
        assert icons.diagnostic == "🔍"
        
        # Special
        assert icons.robot == "🤖"
        assert icons.user == "👤"
        assert icons.tool == "🔧"
        assert icons.folder == "📁"
        assert icons.file == "📄"


class TestTheme:
    """Test Theme class."""
    
    def test_default_theme_initialization(self):
        """Test default theme initialization."""
        theme = Theme()
        
        assert theme.name == "default"
        assert isinstance(theme.colors, ColorScheme)
        assert isinstance(theme.icons, Icons)
    
    def test_named_theme_initialization(self):
        """Test initialization with specific theme names."""
        # Test each available theme
        theme_names = ["default", "dark", "light", "minimal", "terminal"]
        
        for name in theme_names:
            theme = Theme(name)
            assert theme.name == name
            assert isinstance(theme.colors, ColorScheme)
            assert isinstance(theme.icons, Icons)
    
    def test_minimal_theme_has_ascii_icons(self):
        """Test that minimal theme uses ASCII-only icons."""
        theme = Theme("minimal")
        
        # Check for ASCII representations
        assert theme.icons.success == "[OK]"
        assert theme.icons.error == "[ERROR]"
        assert theme.icons.warning == "[WARN]"
        assert theme.icons.info == "[INFO]"
        assert theme.icons.debug == "[DEBUG]"
        assert theme.icons.chat == "[CHAT]"
        assert theme.icons.interactive == "[INTERACTIVE]"
        assert theme.icons.diagnostic == "[DIAGNOSTIC]"
    
    def test_terminal_theme_has_simple_icons(self):
        """Test that terminal theme uses simple icons."""
        theme = Theme("terminal")
        
        # Check for simple representations
        assert theme.icons.warning == "!"
        assert theme.icons.info == "i"
        assert theme.icons.debug == "?"
        assert theme.icons.chat == "[CHAT]"
        assert theme.icons.interactive == "[LIVE]"
        assert theme.icons.diagnostic == "[DIAG]"
    
    def test_format_method(self):
        """Test format method (compatibility layer)."""
        theme = Theme()
        result = theme.format("test text", "success")
        assert result == "test text"  # Currently just passes through
    
    def test_get_style_method(self):
        """Test get_style method."""
        theme = Theme()
        
        # Test valid styles
        assert theme.get_style("success") == "green"
        assert theme.get_style("error") == "red"
        assert theme.get_style("warning") == "yellow"
        
        # Test invalid style
        assert theme.get_style("nonexistent") == ""


class TestThemeManagement:
    """Test global theme management functions."""
    
    def test_get_theme_returns_singleton(self):
        """Test that get_theme returns a singleton instance."""
        theme1 = get_theme()
        theme2 = get_theme()
        
        assert theme1 is theme2
        assert isinstance(theme1, Theme)
    
    def test_set_theme_changes_global_theme(self):
        """Test that set_theme changes the global theme."""
        # Set a specific theme
        new_theme = set_theme("dark")
        
        assert isinstance(new_theme, Theme)
        assert new_theme.name == "dark"
        
        # Verify it's now the global theme
        current = get_theme()
        assert current is new_theme
        assert current.name == "dark"
    
    def test_reset_theme_returns_to_default(self):
        """Test that reset_theme returns to default theme."""
        # Set a non-default theme
        set_theme("dark")
        
        # Reset
        reset_theme()
        
        # Verify it's back to default
        current = get_theme()
        assert current.name == "default"
    
    def teardown_method(self):
        """Reset theme after each test."""
        reset_theme()


class TestThemeIntegration:
    """Test theme integration with chuk-term."""
    
    @patch('mcp_cli.ui.theme.chuk_set_theme')
    def test_theme_applies_to_chuk_term(self, mock_chuk_set_theme):
        """Test that theme changes are applied to chuk-term."""
        # Create theme with specific name
        theme = Theme("dark")
        
        # Verify chuk_set_theme was called with correct mapping
        mock_chuk_set_theme.assert_called_once_with("dark")
    
    @patch('mcp_cli.ui.theme.chuk_set_theme')
    def test_theme_handles_chuk_term_failure(self, mock_chuk_set_theme):
        """Test that theme handles chuk-term failures gracefully."""
        # Make chuk_set_theme raise an exception
        mock_chuk_set_theme.side_effect = Exception("Failed to set theme")
        
        # Should not raise - continues with default
        theme = Theme("dark")
        assert theme.name == "dark"
        assert isinstance(theme.colors, ColorScheme)
        assert isinstance(theme.icons, Icons)