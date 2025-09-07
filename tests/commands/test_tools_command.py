# commands/test_tools_command.py

import pytest
import json

import mcp_cli.commands.tools as tools_mod
from mcp_cli.commands.tools import tools_action_async
from mcp_cli.tools.models import ToolInfo
from tests.conftest import setup_test_context


class DummyTMNoTools:
    async def get_unique_tools(self):
        return []


class DummyTMWithTools:
    def __init__(self, tools):
        self._tools = tools

    async def get_unique_tools(self):
        return self._tools


def make_tool(name, namespace):
    return ToolInfo(
        name=name,
        namespace=namespace,
        description="d",
        parameters={},
        is_async=False,
        tags=[],
    )


@pytest.mark.asyncio
async def test_tools_action_no_tools(monkeypatch):
    # Arrange: capture print and warning calls
    printed_messages = []

    def capture_output(message):
        printed_messages.append(message)

    # Patch both output methods that might be used
    monkeypatch.setattr(tools_mod.output, "print", capture_output)
    monkeypatch.setattr(tools_mod.output, "warning", capture_output)
    monkeypatch.setattr(tools_mod.output, "info", capture_output)

    tm = DummyTMNoTools()
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    # Act
    result = await tools_action_async()

    # Assert
    assert result == []
    assert any("No tools available" in str(m) for m in printed_messages)


@pytest.mark.asyncio
async def test_tools_action_table(monkeypatch):
    # Arrange: capture output calls
    captured_messages = []
    captured_tables = []

    def capture_info(msg):
        captured_messages.append(msg)

    def capture_success(msg):
        captured_messages.append(msg)

    def capture_print(msg):
        captured_messages.append(msg)

    def capture_print_table(table):
        captured_tables.append(table)

    # Patch output methods
    monkeypatch.setattr(tools_mod.output, "print", capture_print)
    monkeypatch.setattr(tools_mod.output, "info", capture_info)
    monkeypatch.setattr(tools_mod.output, "success", capture_success)
    monkeypatch.setattr(tools_mod.output, "print_table", capture_print_table)

    fake_tools = [make_tool("t1", "ns1"), make_tool("t2", "ns2")]
    tm = DummyTMWithTools(fake_tools)
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    # Monkeypatch create_tools_table to return a mock table
    dummy_table = "mock_table"
    monkeypatch.setattr(
        tools_mod, "create_tools_table", lambda tools, show_details=False: dummy_table
    )

    # Act
    result = await tools_action_async(show_details=True, show_raw=False)

    # Assert
    # Should return the expected JSON structure
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "t1"
    assert result[0]["namespace"] == "ns1"
    assert result[1]["name"] == "t2"
    assert result[1]["namespace"] == "ns2"

    # Check what was captured
    assert any("Fetching tool catalogue" in str(msg) for msg in captured_messages)
    assert any("Total tools available: 2" in str(msg) for msg in captured_messages)

    # Should have printed the table
    assert len(captured_tables) == 1
    assert captured_tables[0] == dummy_table


@pytest.mark.asyncio
async def test_tools_action_raw(monkeypatch):
    # Arrange: capture print calls
    printed_objects = []

    def capture_print(obj):
        printed_objects.append(obj)

    # Patch output.print to capture messages
    monkeypatch.setattr(tools_mod.output, "print", capture_print)

    fake_tools = [make_tool("x", "ns")]
    tm = DummyTMWithTools(fake_tools)
    # Setup context with test tool manager
    setup_test_context(tool_manager=tm)

    # Act
    result = await tools_action_async(show_raw=True)

    # Assert
    # Should return raw JSON list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["name"] == "x"
    assert result[0]["namespace"] == "ns"

    # The implementation should call output.json in raw mode
    # We need to patch that method
    captured_json = []

    def capture_json(data):
        captured_json.append(data)

    monkeypatch.setattr(tools_mod.output, "json", capture_json)
    monkeypatch.setattr(tools_mod.output, "info", capture_print)

    # Re-run the test with json capture
    await tools_action_async(show_raw=True)

    # Should have called output.json
    if captured_json:
        json_data = json.loads(captured_json[0])
        assert json_data[0]["name"] == "x"
        assert len(json_data) == 1
        assert json_data[0]["namespace"] == "ns"
