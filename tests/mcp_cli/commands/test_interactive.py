"""
Unit-tests for mcp_cli.commands.interactive

We stub *all* sub-commands so that the interactive loop can be exercised
without actually talking to servers or requiring user input.
"""
from __future__ import annotations

import asyncio
import builtins
from itertools import cycle
from typing import Any, Dict, List

import pytest
from rich.console import Console

# --------------------------------------------------------------------------- #
# System under test
# --------------------------------------------------------------------------- #
import mcp_cli.commands.interactive as inter_mod


# --------------------------------------------------------------------------- #
# Helpers / stubs
# --------------------------------------------------------------------------- #
class FakeStreamManager:
    """Tiny stub that provides just the attributes used by interactive.py."""

    def __init__(self) -> None:
        self._tools: List[Dict[str, Any]] = [
            {"name": "Server1_echo"},
            {"name": "Server2_sum"},
        ]
        self._server_info = [
            {"id": 1, "name": "One", "tools": 1, "status": "up"},
            {"id": 2, "name": "Two", "tools": 1, "status": "up"},
        ]
        self.tool_to_server_map = {t["name"]: "Srv" for t in self._tools}
        self.server_names = {0: "One", 1: "Two"}

    # API expected by interactive.py
    def get_all_tools(self) -> List[Dict[str, Any]]:  # noqa: D401
        return self._tools

    def get_server_info(self) -> List[Dict[str, Any]]:  # noqa: D401
        return self._server_info


# --------------------------------------------------------------------------- #
# Patching helpers
# --------------------------------------------------------------------------- #
def _patch_prompt(monkeypatch, answers: List[str]) -> None:
    """
    Replace user-prompting with a deterministic generator that yields *answers*.

    After the list is exhausted it keeps repeating the last element.  We have
    to monkey-patch **both**:

    * rich.prompt.Prompt.ask           (some sub-commands still use it)
    * prompt_toolkit.PromptSession.prompt_async (main interactive loop)
    """
    it = cycle(answers)

    # ---- Rich Prompt (legacy paths) ---------------------------------------
    def fake_prompt(*_args, **_kw):  # noqa: D401
        return next(it)

    monkeypatch.setattr("rich.prompt.Prompt.ask", fake_prompt, raising=True)

    # ---- prompt_toolkit prompt_async (interactive loop) -------------------
    async def fake_prompt_async(self, *_a, **_kw):  # noqa: D401
        # `self` is the PromptSession instance – ignore
        return next(it)

    monkeypatch.setattr(
        "prompt_toolkit.PromptSession.prompt_async",
        fake_prompt_async,
        raising=True,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sm() -> FakeStreamManager:
    return FakeStreamManager()


@pytest.fixture(autouse=True)
def no_clear(monkeypatch):
    """Disable clear_screen so test logs stay visible."""
    monkeypatch.setattr("mcp_cli.ui.ui_helpers.clear_screen", lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# Happy-path – immediately exit
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_exit_immediately(monkeypatch, sm, capsys):
    _patch_prompt(monkeypatch, ["exit"])

    # run loop – should quit after first iteration
    ok = await inter_mod.interactive_mode(sm)
    assert ok is True

    out, _ = capsys.readouterr()
    # banner printed
    assert "Interactive Mode" in out
    # goodbye message printed
    assert "Goodbye" in out


# --------------------------------------------------------------------------- #
# Built-in commands are dispatched
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_ping_then_exit(monkeypatch, sm):
    _patch_prompt(monkeypatch, ["/ping", "exit"])

    # substitute ping.ping_run with async-mock
    called = False

    async def fake_ping_run(*_a, **_kw):  # noqa: D401
        nonlocal called
        called = True

    monkeypatch.setattr("mcp_cli.commands.ping.ping_run", fake_ping_run, raising=True)

    ok = await inter_mod.interactive_mode(sm)
    assert ok is True
    assert called is True


# --------------------------------------------------------------------------- #
# Unknown command generates warning
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_unknown_command(monkeypatch, sm, capsys):
    _patch_prompt(monkeypatch, ["foobar", "exit"])
    ok = await inter_mod.interactive_mode(sm)
    assert ok is True

    out, _ = capsys.readouterr()
    assert "Unknown command" in out
    assert "foobar" in out


# --------------------------------------------------------------------------- #
# /servers prints a nice table
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_servers_table(monkeypatch, sm, capsys):
    _patch_prompt(monkeypatch, ["/servers", "exit"])
    ok = await inter_mod.interactive_mode(sm)
    assert ok is True

    out, _ = capsys.readouterr()
    # headings of the table
    assert "Connected Servers" in out
    # server names appear
    assert "One" in out and "Two" in out


# --------------------------------------------------------------------------- #
# /tools-raw delegates to tools_command
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tools_raw_dispatch(monkeypatch, sm):
    _patch_prompt(monkeypatch, ["/tools-raw", "exit"])

    captured_args: List[List[str]] = []

    async def fake_tools_command(args, ctx):  # noqa: D401
        captured_args.append(args)

    # patch the re-export inside *interactive* itself …
    monkeypatch.setattr(
        inter_mod, "tools_command", fake_tools_command, raising=True
    )
    # … and (optionally) the original symbol (harmless if already patched)
    monkeypatch.setattr(
        "mcp_cli.chat.commands.tools.tools_command",
        fake_tools_command,
        raising=False,
    )

    ok = await inter_mod.interactive_mode(sm)
    assert ok is True
    assert captured_args == [["--raw"]]
