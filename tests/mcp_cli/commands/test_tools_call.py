# commands/test_tools_call.py
import pytest
import asyncio
import builtins
import json

from rich.console import Console
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.models import ToolInfo, ToolCallResult

class DummyTMEmpty:
    def get_all_tools(self):
        return []

class DummyTMOne:
    def __init__(self, tool_info: ToolInfo):
        self._tool = tool_info
        self.executed = None

    def get_all_tools(self):
        return [self._tool]

    async def execute_tool(self, name, args):
        self.executed = (name, args)
        return ToolCallResult(tool_name=name, success=True, result={"ok": True})

@pytest.mark.asyncio
async def test_no_tools(monkeypatch):
    tm = DummyTMEmpty()
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    await tools_call_action(tm)
    assert any("No tools available" in m for m in printed)

@pytest.mark.asyncio
async def test_invalid_selection(monkeypatch):
    tool = ToolInfo(name="t1", namespace="ns", description="", parameters={}, is_async=False, tags=[])
    tm = DummyTMOne(tool)
    # input not a number
    monkeypatch.setattr(builtins, "input", lambda prompt="": "foo")

    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    await tools_call_action(tm)
    assert any("Please enter a valid number" in m for m in printed)

@pytest.mark.asyncio
async def test_full_flow(monkeypatch):
    tool = ToolInfo(
        name="t2",
        namespace="ns",
        description="desc",
        parameters={"properties": {}, "required": []},
        is_async=False,
        tags=[]
    )
    tm = DummyTMOne(tool)
    # inputs: "1" then "{}"
    inputs = iter(["1", "{}"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(msg))

    await tools_call_action(tm)
    # verify execution
    assert tm.executed == ("t2", {})
    # ensure we printed the JSON back out
    assert any(isinstance(m, dict) or (isinstance(m, str) and "{}" in m) for m in printed)
