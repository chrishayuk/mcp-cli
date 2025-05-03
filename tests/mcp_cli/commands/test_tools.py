"""
pytest suite for mcp_cli.commands.tools (updated May 2025)

The CLI sub-module exposes two public coroutines:

* tools_list(tool_manager)
* tools_call(tool_manager)

This suite injects a tiny **FakeToolManager** that fulfils only the methods the
code actually touches:

    • get_all_tools()
    • execute_tool()
    • get_server_for_tool()          (only so the code can print its name)

No real servers or interactive input are needed – we monkey-patch
`asyncio.to_thread` so that the prompts receive canned answers.
"""
from __future__ import annotations

import asyncio
import builtins
from itertools import cycle
from typing import Any, Dict, List

import pytest
from rich.console import Console

from mcp_cli.commands import tools as tools_cmd
from mcp_cli.tools.models import ToolInfo, ToolCallResult


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

class FakeToolManager:
    """Minimal stub compatible with mcp_cli.commands.tools"""

    def __init__(self, tools: List[ToolInfo]):
        self._tools = tools
        self._namespace_map = {t.name: t.namespace for t in tools}

    # --- API expected by commands.tools ----------------------------------- #
    def get_all_tools(self) -> List[ToolInfo]:
        return self._tools

    def get_server_for_tool(self, tool_name: str) -> str:
        return self._namespace_map.get(tool_name, "UnknownSrv")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:  # noqa: D401
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result=f"Echo {tool_name} with {arguments}",
        )


# Monkey-patch helper – replaces asyncio.to_thread so keyboard input is mocked
def _patch_to_thread(monkeypatch, replies: List[str]) -> None:
    answers = cycle(replies)

    async def fake_to_thread(func, *args, **kwargs):  # noqa: D401
        if func is builtins.input:
            return next(answers)
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def single_tool() -> List[ToolInfo]:
    return [
        ToolInfo(
            name="hello_world",
            namespace="UnitSrv",
            description="Simple test tool",
            parameters={
                "type": "object",
                "properties": {
                    "greeting": {
                        "type": "string",
                        "description": "Text to greet with",
                    }
                },
                "required": ["greeting"],
            },
            is_async=False,
            tags=[],
        )
    ]


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tools_list_single_tool(capsys, single_tool):
    tm = FakeToolManager(single_tool)

    # tools_list prints to stdout using Rich, capture it.
    await tools_cmd.tools_list(tm)

    out, _ = capsys.readouterr()
    assert "hello_world" in out
    assert "Simple test tool" in out
    assert "UnitSrv" in out


@pytest.mark.asyncio
async def test_tools_call_success(monkeypatch, capsys, single_tool):
    """
    Simulate a user selecting the first tool and entering empty JSON `{}` for
    arguments.  The FakeToolManager echoes the call; we assert that echo is
    shown.
    """
    tm = FakeToolManager(single_tool)

    # Provide canned answers:  "1" (select first tool)   "{}" (args)
    _patch_to_thread(monkeypatch, ["1", "{}"])

    await tools_cmd.tools_call(tm)
    out, _ = capsys.readouterr()

    # Confirmation lines
    assert "Selected: hello_world from UnitSrv" in out
    # Echo from FakeToolManager.execute_tool
    assert "Echo hello_world" in out
