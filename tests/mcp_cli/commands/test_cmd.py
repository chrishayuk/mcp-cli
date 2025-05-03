"""
Tests for the *non-interactive* command-mode (`mcp_cli.commands.cmd`).

Only the **public interface** expected from a StreamManager is faked – the real
implementation can live in (and be imported from) *any* package.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from pytest import CaptureFixture
from typer import Exit  # Typer’s own abort exception

import mcp_cli.commands.cmd as cmd_mod

###############################################################################
# Fakes / stubs
###############################################################################

class FakeStreamManager:
    """
    Tiny stub exposing just the public surface used by `cmd.py`.
    It now accepts *both* positional **and** keyword arguments so it works
    with direct cmd-calls **and** the tools-handler round-trip.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def call_tool(self, *args, **kwargs):
        """
        Accept either:
            call_tool(tool_name, arguments)
        or:
            call_tool(tool_name=…, arguments=…)
        """
        if args:                     # positional style
            tool_name, arguments = args
        else:                        # keyword style
            tool_name = kwargs["tool_name"]
            arguments = kwargs["arguments"]

        self.calls.append({"tool": tool_name, "args": arguments})
        return {"isError": False, "content": {"echo": arguments}}

    # unchanged
    def get_internal_tools(self):
        return [
            {
                "name": "echo",
                "description": "Return whatever you pass",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ]



class DummyLLM:
    """A **very** small fake of an LLM client used by `cmd.py`."""

    def __init__(self, *, with_tool: bool = False):
        self._with_tool = with_tool
        self.received_messages: List[Dict[str, Any]] = []

    # `cmd.py` only ever calls `create_completion`
    def create_completion(self, *, messages, tools=None):  # noqa: D401
        self.received_messages = messages
        if self._with_tool:  # ask for a fictitious tool-call on first pass
            self._with_tool = False
            return {
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"text":"hi"}',
                        },
                    }
                ]
            }
        # final answer
        return {"response": "LLM-answer"}


###############################################################################
# Helpers
###############################################################################


async def _invoke_cmd(**kwargs):
    """
    Convenience wrapper to call the Typer command *directly* (async).

    The *actual* CLI runner isn’t needed – Typer exposes the coroutine as
    `.callback` on the command object.
    """
    return await cmd_mod.cmd_run.callback(**kwargs)  # type: ignore[attr-defined]


###############################################################################
# Tests
###############################################################################


@pytest.mark.asyncio
async def test_single_tool_stdout(capsys: CaptureFixture[str]):
    """Happy-path for ``--tool`` – result printed to stdout."""
    sm = FakeStreamManager()
    await _invoke_cmd(
        tool="echo",
        tool_args='{"text":"hello"}',
        stream_manager=sm,
    )

    out, _ = capsys.readouterr()
    assert json.loads(out.strip()) == {"echo": {"text": "hello"}}
    assert sm.calls == [{"tool": "echo", "args": {"text": "hello"}}]


@pytest.mark.asyncio
async def test_single_tool_error(monkeypatch):
    """`call_tool` returning ``isError`` aborts the program."""
    sm = FakeStreamManager()

    async def boom(*_args, **_kwargs):
        return {"isError": True, "error": "kaputt"}

    monkeypatch.setattr(sm, "call_tool", boom, raising=True)

    # Typer signals CLI abort with its own Exit exception
    with pytest.raises(Exit):
        await _invoke_cmd(tool="echo", stream_manager=sm)


@pytest.mark.asyncio
async def test_missing_stream_manager():
    """Passing *no* StreamManager raises immediately (dependency check)."""
    with pytest.raises(Exit):
        await _invoke_cmd(tool="echo", stream_manager=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_llm_simple(monkeypatch, capsys: CaptureFixture[str]):
    """
    LLM branch without tool-calls: we patch `get_llm_client` so no network
    traffic is required.
    """
    sm = FakeStreamManager()
    fake_llm = DummyLLM()

    monkeypatch.setattr(
        cmd_mod, "get_llm_client", lambda provider, model: fake_llm, raising=True
    )

    await _invoke_cmd(
        input="-",  # empty stdin
        stream_manager=sm,
        provider="stub",
        model="123",
    )

    out, _ = capsys.readouterr()
    assert out.strip() == "LLM-answer"
    # LLM received the converted tools
    assert fake_llm.received_messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_llm_with_tool_call(monkeypatch):
    """Full round-trip: LLM asks for a tool, cmd handles it and asks again."""
    sm = FakeStreamManager()
    fake_llm = DummyLLM(with_tool=True)

    monkeypatch.setattr(
        cmd_mod, "get_llm_client", lambda provider, model: fake_llm, raising=True
    )

    result = await _invoke_cmd(stream_manager=sm)

    # final answer delivered
    assert result.strip() == "LLM-answer"
    # StreamManager invoked via handle_tool_call
    assert sm.calls[0]["tool"] == "echo"
    assert sm.calls[0]["args"] == {"text": "hi"}
