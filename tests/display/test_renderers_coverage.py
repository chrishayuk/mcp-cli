"""Additional tests for renderers module to increase coverage."""

import json
import time

from mcp_cli.chat.models import ToolExecutionState
from mcp_cli.display.renderers import (
    _sanitize_for_display,
    show_tool_execution_result,
)


class TestSanitizeForDisplay:
    """Tests for _sanitize_for_display function."""

    def test_sanitize_newlines(self):
        """Test sanitizing newline characters."""
        result = _sanitize_for_display("line1\nline2\nline3")
        assert "\\n" in result
        assert "\n" not in result

    def test_sanitize_carriage_return(self):
        """Test sanitizing carriage return characters."""
        result = _sanitize_for_display("text\rmore")
        assert "\\r" in result
        assert "\r" not in result

    def test_sanitize_crlf(self):
        """Test sanitizing CRLF sequences."""
        result = _sanitize_for_display("line1\r\nline2")
        assert "\\r\\n" in result

    def test_sanitize_tab(self):
        """Test sanitizing tab characters."""
        result = _sanitize_for_display("col1\tcol2")
        assert "\\t" in result
        assert "\t" not in result

    def test_sanitize_escape(self):
        """Test sanitizing ESC character."""
        result = _sanitize_for_display("text\x1b[31mred")
        assert "\\x1b" in result
        assert "\x1b" not in result

    def test_sanitize_clean_text(self):
        """Test that clean text is unchanged."""
        clean = "Hello world! 123"
        result = _sanitize_for_display(clean)
        assert result == clean

    def test_sanitize_mixed_control_chars(self):
        """Test sanitizing mixed control characters."""
        result = _sanitize_for_display("a\nb\tc\rd")
        assert "\\n" in result
        assert "\\t" in result
        assert "\\r" in result


class TestShowToolExecutionResultCoverage:
    """Additional tests for show_tool_execution_result to cover all branches."""

    def test_result_dict_with_many_keys(self, capsys):
        """Test result with dict having more than 5 keys."""
        # Lines 157-158 - dict with more than 5 keys
        result_data = {f"key{i}": f"value{i}" for i in range(10)}

        tool = ToolExecutionState(
            name="dict_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should mention total keys
        assert "10 keys total" in output

    def test_result_dict_with_few_keys(self, capsys):
        """Test result with dict having 5 or fewer keys."""
        result_data = {"a": 1, "b": 2, "c": 3}

        tool = ToolExecutionState(
            name="small_dict",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should show keys without "total"
        assert "keys total" not in output

    def test_result_dict_with_nested_values(self, capsys):
        """Test result dict with nested dict/list values."""
        # Lines 163-169 - nested values with JSON formatting
        result_data = {
            "nested_dict": {"inner": "value"},
            "nested_list": [1, 2, 3],
            "simple": "string",
        }

        tool = ToolExecutionState(
            name="nested_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "nested_dict" in output or "nested_list" in output

    def test_result_dict_with_long_values(self, capsys):
        """Test result dict with values that need truncation."""
        # Lines 165-166 - value truncation
        result_data = {"long_value": "x" * 100}

        tool = ToolExecutionState(
            name="long_value_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should be truncated
        assert "..." in output

    def test_result_dict_with_control_chars_in_values(self, capsys):
        """Test result dict with control characters in values."""
        # Line 168 - sanitization of values
        result_data = {"message": "line1\nline2\ttab"}

        tool = ToolExecutionState(
            name="control_chars_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should have sanitized control chars
        assert "\\n" in output or "\\t" in output

    def test_result_list(self, capsys):
        """Test result that is a JSON list."""
        # Lines 173-188 - list handling
        result_data = ["item1", "item2", "item3", "item4", "item5"]

        tool = ToolExecutionState(
            name="list_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "5 items" in output
        assert "[0]" in output  # First item index

    def test_result_list_with_many_items(self, capsys):
        """Test result list with more than 3 items shows 'more' message."""
        # Lines 187-188 - "and X more" message
        result_data = list(range(10))

        tool = ToolExecutionState(
            name="big_list",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "7 more" in output

    def test_result_list_with_nested_items(self, capsys):
        """Test result list with nested dict/list items."""
        # Lines 177-180 - nested item formatting
        result_data = [{"key": "value"}, [1, 2, 3], "simple"]

        tool = ToolExecutionState(
            name="nested_list",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "[0]" in output

    def test_result_list_with_long_items(self, capsys):
        """Test result list with items that need truncation."""
        # Lines 182-183 - item truncation
        result_data = ["x" * 100]

        tool = ToolExecutionState(
            name="long_item_list",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "..." in output

    def test_result_list_with_control_chars(self, capsys):
        """Test result list with control chars in items."""
        # Lines 184-186 - sanitization
        result_data = ["line1\nline2"]

        tool = ToolExecutionState(
            name="control_list",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "\\n" in output

    def test_result_simple_json_value(self, capsys):
        """Test result that is a simple JSON value (string, number, etc.)."""
        # Lines 189-194 - simple value handling
        tool = ToolExecutionState(
            name="simple_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps("just a string"),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "just a string" in output

    def test_result_simple_json_long_value(self, capsys):
        """Test simple JSON value that needs truncation."""
        # Lines 192-193 - long simple value
        tool = ToolExecutionState(
            name="long_simple",
            arguments={},
            start_time=time.time(),
            result=json.dumps("x" * 300),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "..." in output
        # Should be truncated to ~200 chars
        assert output.count("x") < 250

    def test_result_json_number(self, capsys):
        """Test result that is a JSON number."""
        tool = ToolExecutionState(
            name="number_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(42),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "42" in output

    def test_result_json_boolean(self, capsys):
        """Test result that is a JSON boolean."""
        tool = ToolExecutionState(
            name="bool_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(True),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "True" in output

    def test_result_invalid_json(self, capsys):
        """Test result that is not valid JSON."""
        # Lines 195-200 - JSONDecodeError handling
        tool = ToolExecutionState(
            name="invalid_json",
            arguments={},
            start_time=time.time(),
            result="not valid json {{{",
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should show as plain string
        assert "not valid json" in output

    def test_result_plain_string_long(self, capsys):
        """Test plain string result that needs truncation."""
        # Lines 198-199 - long non-JSON string
        tool = ToolExecutionState(
            name="long_plain",
            arguments={},
            start_time=time.time(),
            result="x" * 300,
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "..." in output

    def test_result_plain_string_with_control_chars(self, capsys):
        """Test plain string with control characters."""
        # Line 197 - sanitization of non-JSON
        tool = ToolExecutionState(
            name="control_plain",
            arguments={},
            start_time=time.time(),
            result="text\nwith\tnewlines\rand\ttabs",
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "\\n" in output or "\\t" in output

    def test_failed_tool_with_control_chars_in_error(self, capsys):
        """Test failed tool with control chars in error message."""
        # Lines 204-206 - error sanitization
        tool = ToolExecutionState(
            name="failed_tool",
            arguments={},
            start_time=time.time(),
            result="Error:\nStack trace\nline 2",
            success=False,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Error message should have sanitized newlines
        assert "Error:" in output
        assert "\\n" in output

    def test_result_empty_dict(self, capsys):
        """Test result that is an empty dict."""
        tool = ToolExecutionState(
            name="empty_dict",
            arguments={},
            start_time=time.time(),
            result=json.dumps({}),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        # Should not raise
        captured = capsys.readouterr()
        assert "empty_dict" in (captured.out + captured.err)

    def test_result_empty_list(self, capsys):
        """Test result that is an empty list."""
        tool = ToolExecutionState(
            name="empty_list",
            arguments={},
            start_time=time.time(),
            result=json.dumps([]),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "0 items" in output

    def test_result_null_json(self, capsys):
        """Test result that is JSON null."""
        tool = ToolExecutionState(
            name="null_tool",
            arguments={},
            start_time=time.time(),
            result=json.dumps(None),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "None" in output

    def test_result_list_exactly_3_items(self, capsys):
        """Test list with exactly 3 items (no 'more' message)."""
        result_data = ["a", "b", "c"]

        tool = ToolExecutionState(
            name="three_items",
            arguments={},
            start_time=time.time(),
            result=json.dumps(result_data),
            success=True,
            elapsed=1.0,
            completed=True,
        )

        show_tool_execution_result(tool)

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should NOT have "more" message
        assert "more" not in output.lower()
        assert "[0]" in output
        assert "[1]" in output
        assert "[2]" in output
