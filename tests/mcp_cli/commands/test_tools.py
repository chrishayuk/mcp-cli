# commands/test_tools.py

import pytest
import json

from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

import mcp_cli.commands.tools as tools_mod
from mcp_cli.commands.tools import tools_action
from mcp_cli.tools.models import ToolInfo


class DummyTMNoTools:
    def get_unique_tools(self):
        return []

class DummyTMWithTools:
    def __init__(self, tools):
        self._tools = tools
    def get_unique_tools(self):
        return self._tools

@pytest.mark.asyncio
async def test_tools_action_no_tools(monkeypatch):
    tm = DummyTMNoTools()
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(msg))

    out = tools_action(tm)
    assert out == []
    assert any("No tools available" in str(m) for m in printed)

def make_tool(name, namespace):
    return ToolInfo(name=name, namespace=namespace, description="d", parameters={}, is_async=False, tags=[])

def test_tools_action_table(monkeypatch):
    fake_tools = [make_tool("t1", "ns1"), make_tool("t2", "ns2")]
    tm = DummyTMWithTools(fake_tools)

    printed = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: printed.append(obj))

    # Monkeypatch create_tools_table to return a dummy Table
    dummy_table = Table(title="Dummy")
    monkeypatch.setattr(tools_mod, "create_tools_table", lambda tools, show_details=False: dummy_table)

    out = tools_action(tm, show_details=True, show_raw=False)
    # Should return the original tool list
    assert out == fake_tools

    # printed[0] is the fetching message string
    assert isinstance(printed[0], str)

    # Next, the dummy Table
    assert any(o is dummy_table for o in printed), printed

    # And finally the summary string
    assert any("Total tools available: 2" in str(m) for m in printed)

def test_tools_action_raw(monkeypatch):
    fake_tools = [make_tool("x", "ns")]
    tm = DummyTMWithTools(fake_tools)

    printed = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: printed.append(obj))

    # Call action in raw mode
    out = tools_action(tm, show_raw=True)
    # Should return raw JSON list
    assert isinstance(out, list) and isinstance(out[0], dict)
    # And printed a Syntax
    assert any(isinstance(o, Syntax) for o in printed)

    # Verify that the JSON inside Syntax matches our tool list
    syntax_obj = next(o for o in printed if isinstance(o, Syntax))
    text = syntax_obj.code  # the raw JSON text
    data = json.loads(text)
    assert data[0]["name"] == "x"
    assert data[0]["namespace"] == "ns"
