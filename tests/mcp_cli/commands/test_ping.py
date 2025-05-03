"""
Unit-tests for the *new* stream-manager version of ``mcp_cli.commands.ping``.
They validate the **table output** produced by the command instead of the
old one-liner messages.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import re
import pytest

from mcp_cli.commands import ping as ping_cmd


# --------------------------------------------------------------------------- #
# 🔧  tiny fake StreamManager
# --------------------------------------------------------------------------- #
class FakeStreamManager:
    """Expose exactly get_streams(); caller never inspects the items."""

    def __init__(self, n: int = 2) -> None:
        self._streams = [(f"r{i}", f"w{i}") for i in range(n)]

    def get_streams(self):
        return list(self._streams)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _patch_send_ping(monkeypatch, responses):
    """
    Replace *send_ping* with an ``AsyncMock`` emitting *responses* sequentially.
    *responses* may contain bools **or** callables raising exceptions.
    """
    async def _next(*_):
        if not _answers:
            raise RuntimeError("ran out of mock answers")
        item = _answers.pop(0)
        if callable(item):
            return await item()
        return item

    _answers = list(responses)
    mock = AsyncMock(side_effect=_next)
    monkeypatch.setattr("mcp_cli.commands.ping.send_ping", mock, raising=True)
    return mock


# --------------------------------------------------------------------------- #
# helpers to parse the pretty table
# --------------------------------------------------------------------------- #
def _extract_rows(output: str):
    """
    Return ``[(server, status)]`` where *status* is True (✓) or False (✗).
    """
    rows = []
    for line in output.splitlines():
        m = re.match(r"│\s*(\S+)\s*│\s*([✓✗])\s*│", line)
        if m:
            rows.append((m.group(1), m.group(2) == "✓"))
    return rows


# --------------------------------------------------------------------------- #
# 🧪  tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_all_up(monkeypatch, capsys):
    _patch_send_ping(monkeypatch, [True, True])
    await ping_cmd.ping_run(
        FakeStreamManager(2), server_names={0: "Alpha", 1: "Beta"}
    )

    out, _ = capsys.readouterr()
    rows = _extract_rows(out)

    assert rows == [("Alpha", True), ("Beta", True)]


@pytest.mark.asyncio
async def test_one_down(monkeypatch, capsys):
    _patch_send_ping(monkeypatch, [True, False])
    await ping_cmd.ping_run(FakeStreamManager(2))

    out, _ = capsys.readouterr()
    rows = _extract_rows(out)

    assert rows == [("server-0", True), ("server-1", False)]


@pytest.mark.asyncio
async def test_exception(monkeypatch, capsys):
    async def boom(*_):
        raise RuntimeError("network down")

    _patch_send_ping(monkeypatch, [True, boom])
    await ping_cmd.ping_run(FakeStreamManager(2))

    out, _ = capsys.readouterr()
    rows = _extract_rows(out)

    assert rows == [("server-0", True), ("server-1", False)]
