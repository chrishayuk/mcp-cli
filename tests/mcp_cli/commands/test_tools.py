# commands/test_tools.py

import pytest
import json
from unittest.mock import Mock

from rich.table import Table
from rich.syntax import Syntax

import mcp_cli.commands.tools as tools_mod
from mcp_cli.commands.tools import tools_action_async
from mcp_cli.tools.models import ToolInfo


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
        tags=[]
    )


@pytest.mark.asyncio
async def test_tools_action_no_tools(monkeypatch):
    # Arrange: mock console to capture print calls
    mock_console = Mock()
    printed_messages = []
    
    def capture_print(message):
        printed_messages.append(message)
    
    mock_console.print = capture_print
    
    # Patch get_console to return our mock
    monkeypatch.setattr(tools_mod, "get_console", lambda: mock_console)
    
    tm = DummyTMNoTools()
    
    # Act
    result = await tools_action_async(tm)
    
    # Assert
    assert result == []
    assert any("No tools available" in str(m) for m in printed_messages)


@pytest.mark.asyncio
async def test_tools_action_table(monkeypatch):
    # Arrange: mock console to capture print calls
    mock_console = Mock()
    printed_objects = []
    
    def capture_print(obj):
        printed_objects.append(obj)
    
    mock_console.print = capture_print
    
    # Patch get_console to return our mock
    monkeypatch.setattr(tools_mod, "get_console", lambda: mock_console)
    
    fake_tools = [make_tool("t1", "ns1"), make_tool("t2", "ns2")]
    tm = DummyTMWithTools(fake_tools)
    
    # Monkeypatch create_tools_table to return a dummy Table
    dummy_table = Table(title="Dummy")
    monkeypatch.setattr(tools_mod, "create_tools_table", lambda tools, show_details=False: dummy_table)
    
    # Act
    result = await tools_action_async(tm, show_details=True, show_raw=False)
    
    # Assert
    # Should return the expected JSON structure
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "t1"
    assert result[0]["namespace"] == "ns1"
    assert result[1]["name"] == "t2"
    assert result[1]["namespace"] == "ns2"
    
    # Check what was printed
    # printed_objects[0] should be the fetching message string
    assert isinstance(printed_objects[0], str)
    assert "Fetching tool catalogue" in str(printed_objects[0])
    
    # Next, the dummy Table
    assert any(obj is dummy_table for obj in printed_objects), f"Expected dummy_table in {printed_objects}"
    
    # And finally the summary string
    assert any("Total tools available: 2" in str(obj) for obj in printed_objects)


@pytest.mark.asyncio
async def test_tools_action_raw(monkeypatch):
    # Arrange: mock console to capture print calls
    mock_console = Mock()
    printed_objects = []
    
    def capture_print(obj):
        printed_objects.append(obj)
    
    mock_console.print = capture_print
    
    # Patch get_console to return our mock
    monkeypatch.setattr(tools_mod, "get_console", lambda: mock_console)
    
    fake_tools = [make_tool("x", "ns")]
    tm = DummyTMWithTools(fake_tools)
    
    # Act
    result = await tools_action_async(tm, show_raw=True)
    
    # Assert
    # Should return raw JSON list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["name"] == "x"
    assert result[0]["namespace"] == "ns"
    
    # Should have printed a Syntax object
    syntax_objects = [obj for obj in printed_objects if isinstance(obj, Syntax)]
    assert len(syntax_objects) == 1, f"Expected exactly one Syntax object, got: {printed_objects}"
    
    # Verify that the JSON inside Syntax matches our tool list
    syntax_obj = syntax_objects[0]
    text = syntax_obj.code  # the raw JSON text
    data = json.loads(text)
    assert len(data) == 1
    assert data[0]["name"] == "x"
    assert data[0]["namespace"] == "ns"
    assert data[0]["description"] == "d"
    assert data[0]["parameters"] == {}
    assert data[0]["is_async"] == False
    assert data[0]["tags"] == []