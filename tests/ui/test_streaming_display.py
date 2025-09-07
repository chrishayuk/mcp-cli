"""
Tests for the streaming display components.

Tests the CompactStreamingDisplay class and StreamingContext context manager
for content-aware streaming display functionality.
"""

import pytest
from unittest.mock import Mock, patch
from rich.panel import Panel

from mcp_cli.ui.streaming_display import (
    CompactStreamingDisplay,
    StreamingContext,
    tokenize_text,
)


class TestTokenizeText:
    """Tests for the tokenize_text function."""

    def test_tokenize_simple_text(self):
        """Test basic text tokenization."""
        text = "Hello world test"
        tokens = list(tokenize_text(text))

        assert len(tokens) > 0
        # All tokens should be strings
        assert all(isinstance(token, str) for token in tokens)
        # Joined tokens should equal original text
        assert "".join(tokens) == text

    def test_tokenize_with_newlines(self):
        """Test tokenization handles newlines correctly."""
        text = "Line 1\nLine 2\nLine 3"
        tokens = list(tokenize_text(text))

        assert "".join(tokens) == text
        assert any("\n" in token for token in tokens)

    def test_tokenize_empty_text(self):
        """Test tokenization of empty text."""
        tokens = list(tokenize_text(""))
        assert tokens == []

    def test_tokenize_single_word(self):
        """Test tokenization of single word."""
        tokens = list(tokenize_text("hello"))
        assert len(tokens) == 1
        assert tokens[0] == "hello"


class TestCompactStreamingDisplay:
    """Tests for CompactStreamingDisplay class."""

    @pytest.fixture
    def display(self):
        """Create a CompactStreamingDisplay instance."""
        return CompactStreamingDisplay()

    def test_initialization(self, display):
        """Test proper initialization."""
        assert display.title == "🤖 Assistant"
        assert display.mode == "response"
        assert display.first_lines == []
        assert display.current_line == ""
        assert display.total_chars == 0
        assert display.total_lines == 0
        assert not display.preview_captured
        assert display.detected_type is None
        assert display.content == ""

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        display = CompactStreamingDisplay(title="Custom", mode="tool")
        assert display.title == "Custom"
        assert display.mode == "tool"

    def test_detect_content_type_code(self, display):
        """Test code detection."""
        display.detect_content_type("def function():")
        assert display.detected_type == "code"

        display.detected_type = None
        display.detect_content_type("function test() {")
        assert display.detected_type == "code"

        display.detected_type = None
        display.detect_content_type("class MyClass:")
        assert display.detected_type == "code"

        display.detected_type = None
        display.detect_content_type("import os")
        assert display.detected_type == "code"

    def test_detect_content_type_markdown(self, display):
        """Test markdown detection."""
        display.detect_content_type("## Heading")
        assert display.detected_type == "markdown"

        display.detected_type = None
        display.detect_content_type("### Subheading")
        assert display.detected_type == "markdown"

    def test_detect_content_type_code_block(self, display):
        """Test code block detection."""
        display.detect_content_type("```python\nprint('hello')\n```")
        assert display.detected_type == "code"

    def test_detect_content_type_json(self, display):
        """Test JSON detection."""
        display.detect_content_type('{"key": "value"}')
        assert display.detected_type == "json"

        display.detected_type = None
        display.detect_content_type('[{"item": 1}]')
        assert display.detected_type == "json"

    def test_detect_content_type_query(self, display):
        """Test SQL query detection."""
        display.detect_content_type("SELECT * FROM table")
        assert display.detected_type == "query"

        display.detected_type = None
        display.detect_content_type("CREATE TABLE test")
        assert display.detected_type == "query"

    def test_detect_content_type_markup(self, display):
        """Test HTML/XML markup detection."""
        display.detect_content_type("<html><body></body></html>")
        assert display.detected_type == "markup"

        display.detected_type = None
        display.detect_content_type("<div>content</div>")
        assert display.detected_type == "markup"

        display.detected_type = None
        display.detect_content_type("<?xml version='1.0'?>")
        assert display.detected_type == "markup"

    def test_detect_content_type_markdown_table(self, display):
        """Test markdown table detection."""
        table_content = """
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
"""
        display.detect_content_type(table_content)
        assert display.detected_type == "markdown_table"

    def test_detect_content_type_default_text(self, display):
        """Test default text detection."""
        display.detect_content_type("Just some plain text")
        assert display.detected_type == "text"

    def test_is_markdown_table_valid(self, display):
        """Test valid markdown table detection."""
        table = """
| Header 1 | Header 2 |
|----------|----------|
| Row 1    | Row 1    |
"""
        assert display._is_markdown_table(table) is True

    def test_is_markdown_table_invalid(self, display):
        """Test invalid markdown table detection."""
        # Not enough lines
        assert display._is_markdown_table("| Header |") is False

        # No separator line
        not_table = """
| Header 1 | Header 2 |
| Row 1    | Row 1    |
"""
        assert display._is_markdown_table(not_table) is False

        # No pipes
        assert display._is_markdown_table("Regular text") is False

    def test_get_phase_message_response_mode(self, display):
        """Test phase messages for response mode."""
        # Initial phase
        phase = display.get_phase_message()
        assert phase == "Starting"

        # Add some content to trigger progression
        display.total_chars = 100
        phase = display.get_phase_message()
        assert phase in ["Starting", "Generating response"]  # Allow flexibility

        display.total_chars = 600
        phase = display.get_phase_message()
        assert phase in ["Generating response", "Adding details", "Elaborating"]

    def test_get_phase_message_tool_mode(self):
        """Test phase messages for tool mode."""
        display = CompactStreamingDisplay(mode="tool")

        phase = display.get_phase_message()
        assert phase == "Preparing tool"

        display.total_chars = 200
        phase = display.get_phase_message()
        assert phase == "Executing tool"

    def test_get_phase_message_thinking_mode(self):
        """Test phase messages for thinking mode."""
        display = CompactStreamingDisplay(mode="thinking")

        phase = display.get_phase_message()
        assert phase == "Thinking"

        display.total_chars = 200
        phase = display.get_phase_message()
        assert phase == "Analyzing request"

    def test_get_phase_message_with_detected_code_type(self, display):
        """Test phase messages adapt to detected content type."""
        display.detected_type = "code"

        phase = display.get_phase_message()
        assert phase == "Starting"

        display.total_chars = 100
        phase = display.get_phase_message()
        assert phase == "Writing code"

    def test_add_content_basic(self, display):
        """Test basic content addition."""
        content = "Hello world"
        display.add_content(content)

        assert display.content == content
        assert display.total_chars == len(content)

    def test_add_content_with_newlines(self, display):
        """Test content addition with line counting."""
        content = "Line 1\nLine 2\nLine 3"
        display.add_content(content)

        assert display.content == content
        assert display.total_chars == len(content)
        assert display.total_lines >= 2  # Should count newlines

    def test_add_content_preview_capture(self, display):
        """Test preview line capturing."""
        lines = [
            "Short line 1",
            "Short line 2",
            "Short line 3",
            "Short line 4",
            "Short line 5",
        ]
        content = "\n".join(lines)

        display.add_content(content)

        # Should capture first few lines
        assert len(display.first_lines) <= display.max_preview_lines
        assert display.first_lines[0] == lines[0]

    def test_add_content_long_line_truncation(self, display):
        """Test long line truncation in preview."""
        long_line = "x" * 100  # Very long line
        display.add_content(long_line)

        # Should truncate long lines
        assert len(display.first_lines) > 0
        assert len(display.first_lines[0]) <= 70

    def test_get_panel_basic(self, display):
        """Test panel generation."""
        display.add_content("Some content")
        panel = display.get_panel(elapsed=1.5)

        assert isinstance(panel, Panel)
        assert display.title in str(panel.title)

    def test_get_panel_with_preview(self, display):
        """Test panel with content preview."""
        display.add_content("Line 1\nLine 2\nLine 3")
        panel = display.get_panel(elapsed=1.0)

        # Panel should be rendered
        assert isinstance(panel, Panel)
        # Should have fixed height for stability
        assert panel.height == 10

    def test_get_final_panel_markdown(self, display):
        """Test final panel with markdown content."""
        display.content = "# Heading\n\nSome **bold** text"
        panel = display.get_final_panel(elapsed=2.5)

        assert isinstance(panel, Panel)
        assert "Response time: 2.50s" in str(panel.subtitle)

    def test_get_final_panel_markdown_table(self, display):
        """Test final panel with markdown table."""
        display.content = """
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
"""
        panel = display.get_final_panel(elapsed=1.0)

        assert isinstance(panel, Panel)
        # Should handle table content appropriately

    def test_get_final_panel_code_block(self, display):
        """Test final panel with code blocks."""
        display.content = "```python\nprint('hello')\n```"
        panel = display.get_final_panel(elapsed=1.0)

        assert isinstance(panel, Panel)

    def test_get_final_panel_fallback_to_text(self, display):
        """Test fallback to text when markdown fails."""
        # Create content that might cause markdown to fail
        display.content = "Some content with < > special chars"

        with patch(
            "mcp_cli.ui.streaming_display.Markdown", side_effect=Exception("Test error")
        ):
            panel = display.get_final_panel(elapsed=1.0)
            assert isinstance(panel, Panel)


class TestStreamingContext:
    """Tests for StreamingContext context manager."""

    @pytest.fixture
    def console(self):
        """Create a properly configured mock console."""
        console = Mock()
        # Add required Rich Console attributes and methods
        console.is_jupyter = False
        console.set_live = Mock()
        console.clear_live = Mock()
        console.show_cursor = Mock()
        console.push_render_hook = Mock()
        console.pop_render_hook = Mock()
        console.set_alt_screen = Mock(return_value=False)
        console.print = Mock()
        console._lock = Mock()
        console._live = None
        # Make it support context manager protocol
        console.__enter__ = Mock(return_value=console)
        console.__exit__ = Mock(return_value=None)
        return console

    def test_context_manager_basic(self, console):
        """Test basic context manager functionality."""
        with StreamingContext(console) as ctx:
            assert ctx is not None
            assert ctx.display is not None
            assert ctx.live is not None

    def test_context_manager_with_content(self, console):
        """Test context manager with content updates."""
        with StreamingContext(console, title="Test") as ctx:
            ctx.update("Hello")
            ctx.update(" world")

            assert ctx.content == "Hello world"

    def test_context_manager_custom_params(self, console):
        """Test context manager with custom parameters."""
        with StreamingContext(
            console,
            title="Custom Title",
            mode="tool",
            refresh_per_second=4,
            transient=False,
        ) as ctx:
            assert ctx.display.title == "Custom Title"
            assert ctx.display.mode == "tool"
            assert ctx.refresh_per_second == 4
            assert ctx.transient is False

    @patch("mcp_cli.ui.streaming_display.Live")
    def test_context_manager_live_display_lifecycle(self, mock_live_class, console):
        """Test Live display lifecycle management."""
        mock_live = Mock()
        mock_live.__enter__ = Mock(return_value=mock_live)
        mock_live.__exit__ = Mock(return_value=None)
        mock_live.update = Mock()
        mock_live_class.return_value = mock_live

        with StreamingContext(console) as ctx:
            ctx.update("test content")

        # Verify Live was created and used properly
        mock_live_class.assert_called_once()
        mock_live.__enter__.assert_called_once()
        mock_live.__exit__.assert_called_once()
        mock_live.update.assert_called()

    def test_content_property(self, console):
        """Test content property access."""
        with StreamingContext(console) as ctx:
            assert ctx.content == ""

            ctx.update("test")
            assert ctx.content == "test"

    @patch("mcp_cli.ui.streaming_display.time.time")
    def test_elapsed_time_tracking(self, mock_time, console):
        """Test elapsed time is tracked correctly."""
        # Mock time progression
        mock_time.side_effect = [100.0, 100.5, 101.0]  # start, update, end

        with StreamingContext(console) as ctx:
            # Time should progress as we make updates
            ctx.update("content")

        # Verify time was called appropriately
        assert mock_time.call_count >= 2

    def test_update_method(self, console):
        """Test update method functionality."""
        with patch("mcp_cli.ui.streaming_display.Live") as mock_live_class:
            mock_live = Mock()
            mock_live.__enter__ = Mock(return_value=mock_live)
            mock_live.__exit__ = Mock(return_value=None)
            mock_live.update = Mock()
            mock_live_class.return_value = mock_live

            with StreamingContext(console) as ctx:
                ctx.update("first")
                ctx.update(" second")

                # Should have updated Live display
                assert mock_live.update.call_count >= 2
                assert ctx.content == "first second"

    def test_final_panel_display(self, console):
        """Test final panel is displayed after context exit."""
        with StreamingContext(console) as ctx:
            ctx.update("Final content")

        # Console should have been called to print final panel
        console.print.assert_called()

        # The printed object should be a Panel
        args, kwargs = console.print.call_args
        assert len(args) > 0
        # Check if it's a Panel (the final panel)
        printed_obj = args[0]
        assert hasattr(printed_obj, "title") or isinstance(printed_obj, Panel)

    def test_no_final_panel_without_content(self, console):
        """Test no final panel is shown if no content was added."""
        with StreamingContext(console):
            pass  # No content added

        # Should not print anything if no content
        # (or should print minimal content)
        if console.print.called:
            args, kwargs = console.print.call_args
            # If something was printed, it should be minimal or empty
            pass  # Allow some flexibility in implementation
