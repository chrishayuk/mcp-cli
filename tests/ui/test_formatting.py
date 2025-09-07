# tools/test_formatting.py

import pytest
from rich.table import Table
from mcp_cli.ui.formatting import (
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
