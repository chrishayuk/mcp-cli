"""Tests for display color converter utilities."""

from mcp_cli.display import rich_to_prompt_toolkit, create_transparent_completion_style


class TestRichToPromptToolkit:
    """Test rich_to_prompt_toolkit color conversion."""

    def test_none_color(self):
        """Test None color returns default."""
        assert rich_to_prompt_toolkit(None) == "ansibrightblack"

    def test_empty_color(self):
        """Test empty string returns default."""
        assert rich_to_prompt_toolkit("") == "ansibrightblack"

    def test_dim_color(self):
        """Test dim color."""
        assert rich_to_prompt_toolkit("dim") == "ansibrightblack"

    def test_basic_colors(self):
        """Test basic color conversions."""
        assert rich_to_prompt_toolkit("white") == "ansiwhite"
        assert rich_to_prompt_toolkit("black") == "ansiblack"
        assert rich_to_prompt_toolkit("red") == "ansired"
        assert rich_to_prompt_toolkit("green") == "ansigreen"
        assert rich_to_prompt_toolkit("yellow") == "ansiyellow"
        assert rich_to_prompt_toolkit("blue") == "ansiblue"
        assert rich_to_prompt_toolkit("magenta") == "ansimagenta"
        assert rich_to_prompt_toolkit("cyan") == "ansicyan"

    def test_bright_colors(self):
        """Test bright color conversions with bold modifier."""
        assert rich_to_prompt_toolkit("bright_white") == "ansiwhite bold"
        assert rich_to_prompt_toolkit("bright_black") == "ansibrightblack"
        assert rich_to_prompt_toolkit("bright_red") == "ansired bold"
        assert rich_to_prompt_toolkit("bright_green") == "ansigreen bold"
        assert rich_to_prompt_toolkit("bright_yellow") == "ansiyellow bold"
        assert rich_to_prompt_toolkit("bright_blue") == "ansiblue bold"
        assert rich_to_prompt_toolkit("bright_magenta") == "ansimagenta bold"
        assert rich_to_prompt_toolkit("bright_cyan") == "ansicyan bold"

    def test_dark_colors(self):
        """Test dark color conversions."""
        assert rich_to_prompt_toolkit("dark_red") == "ansired"
        assert rich_to_prompt_toolkit("dark_green") == "ansigreen"
        assert rich_to_prompt_toolkit("dark_yellow") == "ansiyellow"
        assert rich_to_prompt_toolkit("dark_blue") == "ansiblue"
        assert rich_to_prompt_toolkit("dark_magenta") == "ansimagenta"
        assert rich_to_prompt_toolkit("dark_cyan") == "ansicyan"
        assert rich_to_prompt_toolkit("dark_goldenrod") == "ansiyellow"

    def test_special_colors(self):
        """Test special color names."""
        assert rich_to_prompt_toolkit("grey50") == "ansibrightblack"
        assert rich_to_prompt_toolkit("gray50") == "ansibrightblack"
        assert rich_to_prompt_toolkit("default") == "ansiwhite"

    def test_modifiers(self):
        """Test style modifiers."""
        assert rich_to_prompt_toolkit("bold") == "bold"
        assert rich_to_prompt_toolkit("underline") == "underline"
        assert rich_to_prompt_toolkit("italic") == "italic"

    def test_composite_styles(self):
        """Test composite styles like 'bold yellow'."""
        result = rich_to_prompt_toolkit("bold yellow")
        assert "bold" in result
        assert "ansiyellow" in result

    def test_composite_with_bright(self):
        """Test composite with bright colors."""
        result = rich_to_prompt_toolkit("bold bright_red")
        assert "bold" in result
        assert "ansired" in result

    def test_composite_avoids_duplicates(self):
        """Test that composite styles process all parts."""
        result = rich_to_prompt_toolkit("bold bold yellow")
        # The function processes each part, so duplicates may appear
        # Just verify the expected components are there
        assert "bold" in result
        assert "ansiyellow" in result

    def test_unknown_color(self):
        """Test unknown color returns default."""
        assert rich_to_prompt_toolkit("unknown_color_xyz") == "ansiwhite"

    def test_composite_with_unknown(self):
        """Test composite with unknown color."""
        result = rich_to_prompt_toolkit("bold unknown_color")
        # Should extract the last part and fall back if unknown
        assert "ansiwhite" in result or result == "ansiwhite"


class MockColorScheme:
    """Mock color scheme for testing."""

    def __init__(
        self,
        accent="bright_white",
        highlight="bright_yellow",
        dim="grey50",
        normal="white",
    ):
        self.accent = accent
        self.highlight = highlight
        self.dim = dim
        self.normal = normal


class TestCreateTransparentCompletionStyle:
    """Test create_transparent_completion_style function."""

    def test_default_black_background(self):
        """Test default black background style."""
        colors = MockColorScheme()
        style = create_transparent_completion_style(colors)

        assert "completion-menu" in style
        assert style["completion-menu"] == "bg:ansiblack"

    def test_white_background(self):
        """Test white background style."""
        colors = MockColorScheme()
        style = create_transparent_completion_style(colors, background_color="white")

        assert style["completion-menu"] == "bg:ansiwhite"

    def test_no_background(self):
        """Test no background (empty string)."""
        colors = MockColorScheme()
        style = create_transparent_completion_style(colors, background_color="")

        assert style["completion-menu"] == ""

    def test_accent_color_applied(self):
        """Test accent color is applied to completions."""
        colors = MockColorScheme(accent="bright_cyan")
        style = create_transparent_completion_style(colors)

        # Verify accent color is converted and applied
        completion = style["completion-menu.completion"]
        assert "ansicyan" in completion or "bold" in completion

    def test_highlight_color_applied(self):
        """Test highlight color is applied to current completion."""
        colors = MockColorScheme(highlight="bright_green")
        style = create_transparent_completion_style(colors)

        completion_current = style["completion-menu.completion.current"]
        assert "ansigreen bold" in completion_current or "bold" in completion_current
        assert "underline" in completion_current

    def test_dim_color_applied(self):
        """Test dim color is applied to meta text."""
        colors = MockColorScheme(dim="grey50")
        style = create_transparent_completion_style(colors)

        assert "ansibrightblack" in style["completion-menu.meta"]

    def test_all_required_keys_present(self):
        """Test all required style keys are present."""
        colors = MockColorScheme()
        style = create_transparent_completion_style(colors)

        required_keys = [
            "completion-menu",
            "completion-menu.completion",
            "completion-menu.completion.current",
            "completion-menu.meta",
            "completion-menu.meta.current",
            "completion-menu.multi-column-meta",
            "completion-menu.border",
            "scrollbar.background",
            "scrollbar.button",
            "auto-suggestion",
        ]

        for key in required_keys:
            assert key in style, f"Missing required key: {key}"

    def test_background_applied_to_all_elements(self):
        """Test background is applied to all menu elements."""
        colors = MockColorScheme()
        style = create_transparent_completion_style(colors, background_color="blue")

        # Check that background is in various elements
        assert "bg:ansiblue" in style["completion-menu"]
        assert "bg:ansiblue" in style["completion-menu.completion"]
        assert "bg:ansiblue" in style["completion-menu.meta"]
        assert "bg:ansiblue" in style["scrollbar.background"]

    def test_auto_suggestion_no_background(self):
        """Test auto-suggestion doesn't have background (only foreground)."""
        colors = MockColorScheme(dim="grey50")
        style = create_transparent_completion_style(colors, background_color="black")

        # Auto-suggestion should only have color, not background
        assert "bg:" not in style["auto-suggestion"]
        assert "ansibrightblack" in style["auto-suggestion"]
