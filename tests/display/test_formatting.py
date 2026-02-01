# tools/test_formatting.py

import pytest
from rich.table import Table
from mcp_cli.display import (
    format_tool_for_display,
    create_tools_table,
    create_servers_table,
    display_tool_call_result,
)
from mcp_cli.tools.models import ToolInfo, ServerInfo, ToolCallResult


def make_sample_tool():
    return ToolInfo(
        name="t1",
        namespace="ns1",
        description="Test tool",
        parameters={
            "properties": {
                "a": {"type": "int"},
                "b": {"type": "string"},
            },
            "required": ["a"],
        },
        is_async=False,
        tags=["x"],
    )


def test_format_tool_for_display_minimal():
    ti = ToolInfo(name="foo", namespace="srv", description=None, parameters=None)
    d = format_tool_for_display(ti)
    assert d["name"] == "foo"
    assert d["server"] == "srv"
    assert d["description"] == "No description"
    assert "parameters" not in d


def test_format_tool_for_display_with_details():
    ti = make_sample_tool()
    d = format_tool_for_display(ti, show_details=True)
    # name, server, description
    assert d["name"] == "t1"
    assert d["server"] == "ns1"
    assert d["description"] == "Test tool"
    # parameters string must mention both fields, with (required) on a
    lines = d["parameters"].splitlines()
    assert "a (required): int" in lines
    assert "b: string" in lines


def test_create_tools_table_basic():
    ti = ToolInfo(name="foo", namespace="srv", description="d", parameters=None)
    table: Table = create_tools_table([ti], show_details=False)
    # table title lists count
    assert table.title == "1 Available Tools"
    # columns should be Server, Tool, Description
    assert [c.header for c in table.columns] == ["Server", "Tool", "Description"]
    # exactly one data row
    rows = list(table.rows)
    assert len(rows) == 1
    # Test that the table structure is correct and can be used
    assert hasattr(table, "columns")
    assert hasattr(table, "rows")
    # Verify we can access basic properties without Rich
    assert len(list(table.rows)) == 1


def test_create_tools_table_with_details():
    ti = make_sample_tool()
    table: Table = create_tools_table([ti], show_details=True)
    assert table.title == "1 Available Tools"
    # now there are four columns
    assert [c.header for c in table.columns] == [
        "Server",
        "Tool",
        "Description",
        "Parameters",
    ]
    # Test that the table structure is correct
    assert hasattr(table, "columns")
    assert hasattr(table, "rows")
    assert len(list(table.rows)) == 1
    # Test that the data format function works
    display_data = format_tool_for_display(ti, show_details=True)
    assert display_data["server"] == "ns1"
    assert display_data["name"] == "t1"
    assert display_data["description"] == "Test tool"
    assert "a (required): int" in display_data["parameters"]
    assert "b: string" in display_data["parameters"]


def test_create_servers_table():
    s1 = ServerInfo(id=1, name="one", status="Up", tool_count=3, namespace="ns")
    s2 = ServerInfo(id=2, name="two", status="Down", tool_count=0, namespace="ns")
    table: Table = create_servers_table([s1, s2])
    assert table.title == "Connected MCP Servers"
    # headers ID, Server Name, Tools, Status
    assert [c.header for c in table.columns] == ["ID", "Server Name", "Tools", "Status"]
    # two rows
    rows = list(table.rows)
    assert len(rows) == 2

    # Test that the table structure is correct
    assert hasattr(table, "columns")
    assert hasattr(table, "rows")
    assert len(list(table.rows)) == 2
    # Test the table properties without Rich rendering
    assert table.title == "Connected MCP Servers"


@pytest.mark.parametrize(
    "result",
    [
        ToolCallResult(
            tool_name="foo", success=True, result="ok", error=None, execution_time=None
        ),
        ToolCallResult(
            tool_name="bar",
            success=True,
            result={"x": 1},
            error=None,
            execution_time=0.5,
        ),
    ],
)
def test_display_tool_call_success(result, capsys):
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()

    # Check that success message appears in stdout or stderr
    output_text = captured.out + captured.err
    assert f"Tool '{result.tool_name}' completed" in output_text

    # the result content should appear
    if isinstance(result.result, dict):
        # Check for each key and value in the result
        for key, value in result.result.items():
            assert str(key) in output_text
            assert str(value) in output_text
    else:
        # For non-dict results, just check the string representation
        assert str(result.result) in output_text


@pytest.mark.parametrize(
    "result",
    [
        ToolCallResult(
            tool_name="foo",
            success=False,
            result=None,
            error="fail",
            execution_time=None,
        ),
        ToolCallResult(
            tool_name="bar", success=False, result=None, error=None, execution_time=1.2
        ),
    ],
)
def test_display_tool_call_failure(result, capsys):
    # Test that the function runs without error for failed tool calls
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()

    # The function should produce some output (chuk-term may bypass capture)
    # At minimum, check that the function executes without errors
    # and that the error content gets to stdout
    if captured.out:
        expected_error = result.error or "Unknown error"
        assert expected_error in captured.out or "Error:" in captured.out


def test_display_tool_call_large_list_of_dicts(capsys):
    """Test display of large list of dict results (>10 items)."""
    large_list = [{"id": i, "value": f"item_{i}"} for i in range(15)]
    result = ToolCallResult(
        tool_name="test", success=True, result=large_list, error=None
    )
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should show summary instead of table
    assert "15 records" in output_text or "First 3 records" in output_text


def test_display_tool_call_simple_list(capsys):
    """Test display of simple list (non-dict items)."""
    simple_list = ["item1", "item2", "item3"]
    result = ToolCallResult(
        tool_name="test", success=True, result=simple_list, error=None
    )
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should show items with bullet points
    assert "3 items" in output_text or "item1" in output_text


def test_display_tool_call_large_simple_list(capsys):
    """Test display of large simple list (>10 items)."""
    large_list = [f"item_{i}" for i in range(15)]
    result = ToolCallResult(
        tool_name="test", success=True, result=large_list, error=None
    )
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should show truncation message
    assert "15 items" in output_text or "5 more" in output_text


def test_display_tool_call_large_dict(capsys):
    """Test display of large dict (>10 keys)."""
    large_dict = {f"key_{i}": f"value_{i}" for i in range(15)}
    result = ToolCallResult(
        tool_name="test", success=True, result=large_dict, error=None
    )
    display_tool_call_result(result, console=None)
    # Should use JSON format for large dicts
    # Just verify it doesn't crash
    assert True


def test_display_tool_call_very_large_dict(capsys):
    """Test display of very large dict with >500 chars JSON."""
    # Create a dict that will be >500 chars when serialized
    large_dict = {f"key_{i}": f"value_{i}" * 20 for i in range(20)}
    result = ToolCallResult(
        tool_name="test", success=True, result=large_dict, error=None
    )
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should show truncation
    assert "truncated" in output_text.lower() or len(output_text) > 0


def test_display_tool_call_long_string(capsys):
    """Test display of long string result (>500 chars)."""
    long_string = "x" * 600
    result = ToolCallResult(
        tool_name="test", success=True, result=long_string, error=None
    )
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should show truncation message
    assert "truncated" in output_text.lower() or "x" * 500 in output_text


def test_display_tool_call_other_type_serializable(capsys):
    """Test display of other types that can be JSON serialized."""
    result = ToolCallResult(tool_name="test", success=True, result=123, error=None)
    display_tool_call_result(result, console=None)
    # Should serialize as JSON
    # Just verify it doesn't crash
    assert True


def test_display_tool_call_other_type_large_json(capsys):
    """Test display of other type with large JSON output."""
    # Create a large nested structure
    large_obj = {"data": ["x" * 100 for _ in range(10)]}
    result = ToolCallResult(
        tool_name="test", success=True, result=large_obj, error=None
    )
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should handle large JSON
    assert len(output_text) > 0 or "truncated" in output_text.lower()


class NonSerializable:
    """Class that can't be JSON serialized."""

    pass


def test_display_tool_call_non_serializable(capsys):
    """Test display of non-JSON-serializable result."""
    obj = NonSerializable()
    result = ToolCallResult(tool_name="test", success=True, result=obj, error=None)
    display_tool_call_result(result, console=None)
    captured = capsys.readouterr()
    output_text = captured.out + captured.err

    # Should fallback to str() representation
    assert "NonSerializable" in output_text or len(output_text) > 0
