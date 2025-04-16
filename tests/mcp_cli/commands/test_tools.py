"""pytest test‑suite for mcp_cli.commands.tools

The tests exercise the two public coroutines exported by the module:
  * ``tools_list`` – prints tabular tool information
  * ``tools_call`` – interactive helper that lets the user choose a tool and
    forwards the request to ``StreamManager.call_tool``.

Rather than spinning up real servers we inject a **fake** ``stream_manager``
object that exposes just the methods those coroutines expect:
  • ``get_all_tools``
  • ``get_server_for_tool``
  • ``call_tool``

We also monkey‑patch ``asyncio.to_thread`` so that the code that normally waits
for keyboard input receives pre‑defined responses immediately.
"""
from __future__ import annotations

import asyncio
import builtins
from itertools import cycle
from types import SimpleNamespace
from typing import Any, List, Dict

import pytest

# Import the module under test
from mcp_cli.commands import tools as tools_cmd

###############################################################################
# Helper fakes
###############################################################################

class FakeStreamManager:
    """Very small stub that satisfies the methods used by tools.py"""

    def __init__(self, tools: List[Dict[str, Any]]):
        self._tools = tools
        # map every tool to its declared server (or fallback)
        self._server_map = {
            tool["name"]: tool.get("server", "TestServer") for tool in tools
        }

    # ---- API expected by tools.py ------------------------------------------------
    def get_all_tools(self) -> List[Dict[str, Any]]:  # noqa: D401 – verb form fine
        return self._tools

    def get_server_for_tool(self, name: str) -> str:  # noqa: D401
        return self._server_map[name]

    async def call_tool(self, *, tool_name: str, arguments: Any, server_name: str):
        # echo back for assertion.
        return {
            "isError": False,
            "content": f"Echo {tool_name} on {server_name} with {arguments}",
        }

###############################################################################
# Fixtures
###############################################################################

@pytest.fixture()
def fake_tools_single() -> List[Dict[str, Any]]:
    return [
        {
            "name": "hello_world",
            "description": "Simple test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "greeting": {
                        "type": "string",
                        "description": "text to greet with",
                    }
                },
                "required": ["greeting"],
            },
            "server": "UnitSrv",
        }
    ]

###############################################################################
# Monkey‑patch asyncio.to_thread so we can provide deterministic "input".
###############################################################################

def make_to_thread_patch(responses: List[str]):
    """Return a replacement for asyncio.to_thread that yields *responses*."""

    answers_iter = cycle(responses)  # will repeat after exhaustion – fine for tests

    async def _fake_to_thread(func, *args, **kwargs):  # noqa: D401 – verb form fine
        # The real to_thread passes ``func`` and args to a worker thread.
        # We only care when func is ``input`` – then we deliver a canned answer.
        if func is builtins.input:
            return next(answers_iter)
        # For any other function, preserve behaviour by executing in loop thread.
        return func(*args, **kwargs)

    return _fake_to_thread

###############################################################################
# Actual tests
###############################################################################

@pytest.mark.asyncio
async def test_tools_list_single_tool(capsys, fake_tools_single):
    """``tools_list`` should print a table with the single fake tool."""
    stream_manager = FakeStreamManager(fake_tools_single)
    await tools_cmd.tools_list(stream_manager)

    out, _ = capsys.readouterr()
    assert "hello_world" in out
    assert "Simple test tool" in out


@pytest.mark.asyncio
async def test_tools_call_success(monkeypatch, capsys, fake_tools_single):
    """User selects the first tool, provides empty JSON – call succeeds."""
    stream_manager = FakeStreamManager(fake_tools_single)

    # Patch asyncio.to_thread so that:
    #   – first call returns "1"   (tool selection)
    #   – second call returns "{}"  (arguments)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        make_to_thread_patch(["1", "{}"]),
    )

    await tools_cmd.tools_call(stream_manager)
    out, _ = capsys.readouterr()

    # selections printed
    assert "Selected: hello_world from UnitSrv" in out
    # Response echoed from FakeStreamManager.call_tool
    assert "Echo hello_world" in out
