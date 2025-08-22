"""Unit tests for code display module."""

import pytest
from unittest.mock import patch, Mock, call

from mcp_cli.ui.code import (
    display_code,
    display_diff,
    display_code_review,
    display_code_analysis,
    display_side_by_side,
    format_code_snippet,
    display_file_tree
)
from mcp_cli.ui.theme import Theme, set_theme


@pytest.fixture
def minimal_theme():
    """Set minimal theme for testing."""
    set_theme("minimal")
    return Theme("minimal")


@pytest.fixture
def default_theme():
    """Set default theme for testing."""
    set_theme("default")
    return Theme("default")


@pytest.fixture
def mock_ui():
    """Mock the ui output object."""
    with patch('mcp_cli.ui.code.ui') as mock:
        yield mock


class TestDisplayCode:
    """Test code display functionality."""
    
    def test_display_code_minimal(self, minimal_theme, mock_ui):
        """Test code display in minimal theme."""
        code = "def hello():\n    print('world')"
        display_code(code, "python", line_numbers=True)
        
        # Should print with line numbers
        calls = mock_ui.print.call_args_list
        assert any("1 |" in str(call) for call in calls)
        assert any("def hello" in str(call) for call in calls)
    
    def test_display_code_with_title(self, minimal_theme, mock_ui):
        """Test code display with title."""
        code = "print('test')"
        display_code(code, title="Example Code")
        
        calls = mock_ui.print.call_args_list
        assert any("Example Code" in str(call) for call in calls)
    
    def test_display_code_terminal_theme(self, mock_ui):
        """Test code display in terminal theme."""
        set_theme("terminal")
        code = "# Comment\ncode_line"
        display_code(code, line_numbers=True)
        
        # Terminal theme should handle comments differently
        calls = mock_ui.print.call_args_list
        assert any("# Comment" in str(call) or "[dim]" in str(call) for call in calls)
    
    def test_display_code_rich_theme(self, default_theme, mock_ui):
        """Test code display in rich theme."""
        code = "import sys\nprint(sys.version)"
        display_code(code, "python", line_numbers=True, highlight_lines=[2])
        
        # Rich theme should use Syntax object
        calls = mock_ui.print.call_args_list
        assert len(calls) > 0


class TestDisplayDiff:
    """Test diff display functionality."""
    
    def test_display_diff_basic(self, minimal_theme, mock_ui):
        """Test basic diff display."""
        old = "line1\nline2"
        new = "line1\nmodified"
        
        display_diff(old, new)
        
        calls = mock_ui.print.call_args_list
        # Should show diff output
        assert len(calls) > 0
    
    def test_display_diff_no_changes(self, minimal_theme, mock_ui):
        """Test diff with no changes."""
        text = "same content"
        display_diff(text, text)
        
        # Should indicate no changes
        mock_ui.info.assert_called_with("No changes detected")
    
    def test_display_diff_with_title(self, minimal_theme, mock_ui):
        """Test diff with title."""
        old = "old"
        new = "new"
        
        display_diff(old, new, title="Changes", file_path="test.py")
        
        calls = mock_ui.print.call_args_list
        assert any("Changes" in str(call) for call in calls)
    
    def test_display_diff_terminal_theme(self, mock_ui):
        """Test diff in terminal theme."""
        set_theme("terminal")
        old = "removed"
        new = "added"
        
        display_diff(old, new)
        
        calls = mock_ui.print.call_args_list
        # Terminal theme should color diff lines
        assert any("[green]" in str(call) or "[red]" in str(call) for call in calls)


class TestDisplayCodeReview:
    """Test code review display."""
    
    def test_display_code_review_minimal(self, minimal_theme, mock_ui):
        """Test code review in minimal theme."""
        code = "def func():\n    pass"
        comments = [
            {
                "line": 1,
                "type": "error",
                "message": "Missing docstring",
                "suggestion": "Add a docstring"
            }
        ]
        
        display_code_review(code, comments)
        
        calls = mock_ui.print.call_args_list
        assert any("REVIEW COMMENTS" in str(call) for call in calls)
        assert any("Missing docstring" in str(call) for call in calls)
    
    def test_display_code_review_multiple_types(self, minimal_theme, mock_ui):
        """Test code review with multiple comment types."""
        code = "x = 1"
        comments = [
            {"line": 1, "type": "error", "message": "Error here"},
            {"line": 1, "type": "warning", "message": "Warning here"},
            {"line": 1, "type": "info", "message": "Info here"},
            {"line": 1, "type": "suggestion", "message": "Suggestion here"}
        ]
        
        display_code_review(code, comments)
        
        calls = mock_ui.print.call_args_list
        assert any("ERROR" in str(call) or "error" in str(call) for call in calls)
        assert any("WARNING" in str(call) or "warning" in str(call) for call in calls)


class TestDisplayCodeAnalysis:
    """Test code analysis display."""
    
    def test_display_code_analysis_minimal(self, minimal_theme, mock_ui):
        """Test code analysis in minimal theme."""
        metrics = {
            "lines": 100,
            "functions": 10,
            "complexity": 5,
            "coverage": 85
        }
        
        display_code_analysis(metrics)
        
        calls = mock_ui.print.call_args_list
        assert any("Code Analysis" in str(call) for call in calls)
        assert any("100" in str(call) for call in calls)
        assert any("85" in str(call) for call in calls)
    
    def test_display_code_analysis_with_issues(self, minimal_theme, mock_ui):
        """Test code analysis with issues."""
        metrics = {
            "complexity": 25,  # High complexity
            "coverage": 50,    # Low coverage
            "issues": [
                {"severity": "high"},
                {"severity": "high"},
                {"severity": "medium"}
            ]
        }
        
        display_code_analysis(metrics, show_recommendations=True)
        
        calls = mock_ui.print.call_args_list
        # Should show recommendations
        assert any("Recommendations" in str(call) for call in calls)
        assert any("refactor" in str(call).lower() for call in calls)
    
    def test_display_code_analysis_terminal_theme(self, mock_ui):
        """Test code analysis in terminal theme."""
        set_theme("terminal")
        metrics = {"test": "value"}
        
        display_code_analysis(metrics)
        
        calls = mock_ui.print.call_args_list
        assert any("[cyan]" in str(call) or "[bold]" in str(call) for call in calls)


class TestDisplaySideBySide:
    """Test side-by-side code display."""
    
    def test_display_side_by_side_minimal(self, minimal_theme, mock_ui):
        """Test side-by-side in minimal theme (sequential)."""
        left = "original"
        right = "modified"
        
        display_side_by_side(left, right)
        
        calls = mock_ui.print.call_args_list
        # Minimal theme shows sequentially
        assert any("Before" in str(call) for call in calls)
        assert any("After" in str(call) for call in calls)
        assert any("original" in str(call) for call in calls)
        assert any("modified" in str(call) for call in calls)
    
    def test_display_side_by_side_rich(self, default_theme, mock_ui):
        """Test side-by-side in rich theme."""
        left = "def old():\n    pass"
        right = "def new():\n    return None"
        
        display_side_by_side(
            left, right,
            left_title="Original",
            right_title="Updated"
        )
        
        # Rich theme creates a table
        calls = mock_ui.print.call_args_list
        assert len(calls) > 0


class TestFormatCodeSnippet:
    """Test code snippet formatting."""
    
    def test_format_code_snippet_inline(self, minimal_theme):
        """Test inline code snippet."""
        result = format_code_snippet("print()", inline=True)
        assert result == "`print()`"
    
    def test_format_code_snippet_block(self, minimal_theme):
        """Test block code snippet."""
        result = format_code_snippet("def func():\n    pass", inline=False)
        assert "```" in result
        assert "def func" in result
    
    def test_format_code_snippet_rich(self, default_theme):
        """Test code snippet in rich theme."""
        result = format_code_snippet("code", inline=True)
        assert "[cyan]" in result or "`code`" in result


class TestDisplayFileTree:
    """Test file tree display."""
    
    def test_display_file_tree_minimal(self, minimal_theme, mock_ui):
        """Test file tree in minimal theme."""
        tree_data = {
            "src": {
                "main.py": "1KB",
                "lib": {
                    "utils.py": "2KB"
                }
            }
        }
        
        display_file_tree(tree_data)
        
        calls = mock_ui.print.call_args_list
        assert any("src" in str(call) for call in calls)
        assert any("main.py" in str(call) for call in calls)
        assert any("utils.py" in str(call) for call in calls)
    
    def test_display_file_tree_with_sizes(self, minimal_theme, mock_ui):
        """Test file tree with file sizes."""
        tree_data = {
            "file1.txt": "10KB",
            "file2.txt": "20KB"
        }
        
        display_file_tree(tree_data, show_sizes=True)
        
        calls = mock_ui.print.call_args_list
        assert any("10KB" in str(call) for call in calls)
        assert any("20KB" in str(call) for call in calls)
    
    def test_display_file_tree_terminal(self, mock_ui):
        """Test file tree in terminal theme."""
        set_theme("terminal")
        tree_data = {"folder": {"file": None}}
        
        display_file_tree(tree_data)
        
        calls = mock_ui.print.call_args_list
        # Terminal theme uses colors
        assert any("[cyan]" in str(call) for call in calls)
    
    def test_display_file_tree_rich_with_icons(self, default_theme, mock_ui):
        """Test file tree in rich theme with icons."""
        tree_data = {
            "docs": {
                "readme.md": None,
                "images": {}
            }
        }
        
        display_file_tree(tree_data, show_icons=True)
        
        # Rich theme creates Tree object
        calls = mock_ui.print.call_args_list
        assert len(calls) > 0