# tools/test_formatting.py

import pytest
import json
from rich.console import Console
from rich.table import Table
from mcp_cli.tools.formatting import (
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
    assert rows[0].cells == ["srv", "foo", "d"]


def test_create_tools_table_with_details():
    ti = make_sample_tool()
    table: Table = create_tools_table([ti], show_details=True)
    assert table.title == "1 Available Tools"
    # now there are four columns
    assert [c.header for c in table.columns] == ["Server", "Tool", "Description", "Parameters"]
    # inspect the single row
    row = table.rows[0].cells
    assert row[0] == "ns1"
    assert row[1] == "t1"
    assert row[2] == "Test tool"
    # parameters cell is multi-line
    assert "a (required): int" in row[3]
    assert "b: string" in row[3]


def test_create_servers_table():
    s1 = ServerInfo(id=1, name="one", status="Up", tool_count=3, namespace="ns")
    s2 = ServerInfo(id=2, name="two", status="Down", tool_count=0, namespace="ns")
    table: Table = create_servers_table([s1, s2])
    assert table.title == "Connected MCP Servers"
    # headers ID, Server Name, Tools, Status
    assert [c.header for c in table.columns] == ["ID", "Server Name", "Tools", "Status"]
    # two rows
    rows = list(table.rows)
    assert rows[0].cells == ["1", "one", "3", "Up"]
    assert rows[1].cells == ["2", "two", "0", "Down"]


@pytest.mark.parametrize("result", [
    ToolCallResult(tool_name="foo", success=True, result="ok", error=None, execution_time=None),
    ToolCallResult(tool_name="bar", success=True, result={"x": 1}, error=None, execution_time=0.5),
])
def test_display_tool_call_success(result):
    console = Console(record=True)
    display_tool_call_result(result, console=console)
    text = console.export_text()
    # check that the title contains the tool name and "Success"
    assert f"Tool '{result.tool_name}' - Success" in text
    # the result content should appear
    if isinstance(result.result, dict):
        # JSON dict
        assert json.dumps(result.result) in text
    else:
        assert str(result.result) in text


@pytest.mark.parametrize("result", [
    ToolCallResult(tool_name="foo", success=False, result=None, error="fail", execution_time=None),
    ToolCallResult(tool_name="bar", success=False, result=None, error=None, execution_time=1.2),
])
def test_display_tool_call_failure(result):
    console = Console(record=True)
    display_tool_call_result(result, console=console)
    text = console.export_text()
    # title should contain "Failed"
    assert f"Tool '{result.tool_name}' - Failed" in text
    # error message
    expected = result.error or "Unknown error"
    assert expected in text
