"""Tests for display formatters module."""

from mcp_cli.display.formatters import (
    format_args_preview,
    format_reasoning_preview,
    format_content_preview,
)


class TestFormatArgsPreview:
    """Tests for format_args_preview function."""

    def test_empty_arguments(self):
        """Test with empty dict."""
        result = format_args_preview({})
        assert result == ""

    def test_single_string_argument(self):
        """Test with single string argument."""
        result = format_args_preview({"name": "test"})
        assert result == "name=test"

    def test_multiple_arguments_within_limit(self):
        """Test with 2 arguments (default limit)."""
        result = format_args_preview({"host": "localhost", "port": "8080"})
        assert "host=localhost" in result
        assert "port=8080" in result

    def test_more_than_max_args(self):
        """Test that only first N args shown with indicator."""
        args = {"a": "1", "b": "2", "c": "3", "d": "4"}
        result = format_args_preview(args, max_args=2)

        # Should have first 2 args
        assert "a=1" in result
        assert "b=2" in result

        # Should have indicator for more
        assert "+2 more" in result

        # Should NOT show c or d
        assert "c=3" not in result

    def test_long_string_value_truncated(self):
        """Test that long string values are truncated."""
        long_str = "x" * 100
        result = format_args_preview({"data": long_str}, max_len=40)

        assert "data=" in result
        assert "..." in result
        assert len(result) < 100

    def test_dict_value(self):
        """Test formatting of dict value."""
        result = format_args_preview({"config": {"key": "value"}})
        assert "config=" in result
        assert "key" in result
        assert "value" in result

    def test_list_value(self):
        """Test formatting of list value."""
        result = format_args_preview({"items": [1, 2, 3]})
        assert "items=" in result
        assert "[1, 2, 3]" in result or "[1,2,3]" in result

    def test_large_dict_value_truncated(self):
        """Test that large dict values are truncated."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        result = format_args_preview({"data": large_dict}, max_len=40)

        assert "data=" in result
        assert "..." in result

    def test_large_list_value_truncated(self):
        """Test that large list values are truncated."""
        large_list = list(range(50))
        result = format_args_preview({"numbers": large_list}, max_len=40)

        assert "numbers=" in result
        assert "..." in result

    def test_integer_value(self):
        """Test formatting of integer value."""
        result = format_args_preview({"count": 42})
        assert "count=42" in result

    def test_boolean_value(self):
        """Test formatting of boolean value."""
        result = format_args_preview({"enabled": True})
        assert "enabled=True" in result

    def test_none_value(self):
        """Test formatting of None value."""
        result = format_args_preview({"optional": None})
        assert "optional=None" in result

    def test_custom_max_args(self):
        """Test with custom max_args parameter."""
        args = {"a": "1", "b": "2", "c": "3"}
        result = format_args_preview(args, max_args=3)

        # All 3 should be shown
        assert "a=1" in result
        assert "b=2" in result
        assert "c=3" in result
        assert "more" not in result

    def test_custom_max_len(self):
        """Test with custom max_len parameter."""
        result = format_args_preview({"data": "x" * 100}, max_len=10)

        # Should be truncated to ~10 chars
        assert "..." in result
        assert len(result) < 50


class TestFormatReasoningPreview:
    """Tests for format_reasoning_preview function."""

    def test_empty_reasoning(self):
        """Test with empty string."""
        result = format_reasoning_preview("")
        assert result == ""

    def test_short_reasoning(self):
        """Test with reasoning shorter than max_len."""
        short = "This is short"
        result = format_reasoning_preview(short)
        assert result == short

    def test_long_reasoning_from_end(self):
        """Test showing last N chars (default)."""
        long_text = "The quick brown fox jumps over the lazy dog. This is the end part."
        result = format_reasoning_preview(long_text, max_len=30, from_end=True)

        # Should start with ...
        assert result.startswith("...")

        # Should contain end part
        assert "end part" in result

        # Should NOT contain beginning
        assert "quick" not in result

    def test_long_reasoning_from_start(self):
        """Test showing first N chars."""
        long_text = "This is the start. More text here. And even more at the end."
        result = format_reasoning_preview(long_text, max_len=30, from_end=False)

        # Should end with ...
        assert result.endswith("...")

        # Should contain start
        assert "start" in result

        # Should NOT contain end
        assert "end" not in result

    def test_word_boundary_from_end(self):
        """Test that it tries to break at word boundary from end."""
        text = "word1 word2 word3 word4 word5 word6"
        result = format_reasoning_preview(text, max_len=20, from_end=True)

        # Should have ellipsis
        assert "..." in result

        # Should not break in middle of word (should find space)
        # Result should not have partial words at start
        words_in_result = result.replace("...", "").strip().split()
        assert all(word in text for word in words_in_result)

    def test_word_boundary_from_start(self):
        """Test that it tries to break at word boundary from start."""
        text = "word1 word2 word3 word4 word5 word6"
        result = format_reasoning_preview(text, max_len=20, from_end=False)

        # Should have ellipsis
        assert "..." in result

        # Should not break in middle of word
        words_in_result = result.replace("...", "").strip().split()
        assert all(word in text for word in words_in_result)

    def test_no_spaces_from_end(self):
        """Test text with no spaces when showing from end."""
        text = "x" * 100
        result = format_reasoning_preview(text, max_len=30, from_end=True)

        # Should have ellipsis
        assert result.startswith("...")

        # Should have ~30 chars (plus ellipsis)
        assert len(result) <= 35

    def test_no_spaces_from_start(self):
        """Test text with no spaces when showing from start."""
        text = "x" * 100
        result = format_reasoning_preview(text, max_len=30, from_end=False)

        # Should have ellipsis
        assert result.endswith("...")

        # Should have ~30 chars (plus ellipsis)
        assert len(result) <= 35

    def test_exact_length(self):
        """Test with text exactly at max_len."""
        text = "x" * 50
        result = format_reasoning_preview(text, max_len=50)

        # Should return as-is
        assert result == text

    def test_custom_max_len(self):
        """Test with custom max_len."""
        text = "x" * 100
        result = format_reasoning_preview(text, max_len=10)

        assert "..." in result
        assert len(result) <= 15


class TestFormatContentPreview:
    """Tests for format_content_preview function."""

    def test_empty_content(self):
        """Test with empty string."""
        result = format_content_preview("")
        assert result == ""

    def test_short_content(self):
        """Test with content shorter than max_len."""
        short = "Short text"
        result = format_content_preview(short)
        assert result == short

    def test_long_content(self):
        """Test with content longer than max_len."""
        long_text = "x" * 200
        result = format_content_preview(long_text, max_len=100)

        # Should have ellipsis
        assert result.endswith("...")

        # Should be truncated
        assert len(result) <= 110

    def test_word_boundary_breaking(self):
        """Test that it breaks at word boundaries when possible."""
        text = "The quick brown fox jumps over the lazy dog. More text here."
        result = format_content_preview(text, max_len=30)

        # Should have ellipsis
        assert result.endswith("...")

        # Should not have partial words at end (should break at space)
        words = result.replace("...", "").strip().split()
        assert all(word in text for word in words)

    def test_word_boundary_too_early(self):
        """Test that word boundary is only used if reasonably far."""
        # Create text where first space is very early
        text = "a " + "x" * 200
        result = format_content_preview(text, max_len=100)

        # Should still truncate at max_len, not use early space
        # (space is at position 1, which is < max_len/2 = 50)
        assert len(result) > 50

    def test_no_spaces(self):
        """Test content with no spaces."""
        text = "x" * 200
        result = format_content_preview(text, max_len=100)

        # Should have ellipsis
        assert result.endswith("...")

        # Should truncate at max_len
        assert len(result) <= 105

    def test_exact_length(self):
        """Test with content exactly at max_len."""
        text = "x" * 100
        result = format_content_preview(text, max_len=100)

        # Should return as-is
        assert result == text

    def test_custom_max_len(self):
        """Test with custom max_len."""
        text = "x" * 200
        result = format_content_preview(text, max_len=50)

        assert result.endswith("...")
        assert len(result) <= 55

    def test_multiline_content(self):
        """Test with multiline content."""
        text = "Line 1\nLine 2\nLine 3\n" + "More text " * 20
        result = format_content_preview(text, max_len=50)

        # Should truncate
        assert result.endswith("...")
        assert len(result) <= 55
