# commands/test_ping.py

import pytest
import asyncio
import time
from typing import Any, Dict, List, Sequence

from rich.console import Console
from rich.table import Table

import mcp_cli.commands.ping as ping_mod
from mcp_cli.commands.ping import _display_name, _ping_one, ping_action
from mcp_cli.tools.models import ServerInfo

class DummyToolManager:
    """
    Fake ToolManager exposing .get_streams() and .get_server_info(),
    plus a server_names attribute.
    """
    def __init__(
        self,
        streams: List[Sequence[Any]],
        server_names: Dict[int, str] = None,
        server_info: List[Dict[str, Any]] = None
    ):
        self._streams = streams
        self.server_names = server_names or {}
        info = server_info or []
        # Build real ServerInfo objects
        self._server_info = [
            ServerInfo(
                id=i,
                name=info_item.get("name", f"server-{i}"),
                status=info_item.get("status", ""),
                tool_count=info_item.get("tools", 0),
                namespace=info_item.get("namespace", ""),
            ) for i, info_item in enumerate(info)
        ]

    def get_streams(self):
        return self._streams

    def get_server_info(self):
        # Return list of ServerInfo
        return self._server_info


class FakeStream:
    """Dummy stream passed to send_ping (which we'll monkeypatch)."""
    pass


@pytest.mark.parametrize(
    "mapping, names, info, idx, expected",
    [
        ({0: "X"}, {}, [], 0, "X"),
        (None, {0: "Y"}, [], 0, "Y"),
        (None, {}, [{"name": "Z"}], 0, "Z"),
        (None, {}, [], 5, "server-5"),
    ]
)
def test_display_name(mapping, names, info, idx, expected):
    tm = DummyToolManager(streams=[], server_names=names, server_info=info)
    assert _display_name(idx, tm, mapping) == expected


@pytest.mark.asyncio
async def test_ping_one_success(monkeypatch):
    monkeypatch.setattr(ping_mod, "send_ping", lambda r, w: asyncio.sleep(0, result=True))
    start = time.perf_counter()
    name, ok, ms = await _ping_one(1, "n1", object(), object(), timeout=1.0)
    end = time.perf_counter()

    assert name == "n1"
    assert ok is True
    assert 0 <= ms <= (end - start) * 1000 + 1


@pytest.mark.asyncio
async def test_ping_one_timeout(monkeypatch):
    async def hang(r, w):
        await asyncio.sleep(10)
    monkeypatch.setattr(ping_mod, "send_ping", hang)

    name, ok, ms = await _ping_one(2, "n2", object(), object(), timeout=0.01)
    assert name == "n2"
    assert ok is False
    assert isinstance(ms, float)


@pytest.mark.asyncio
async def test_ping_action_no_streams(monkeypatch):
    tm = DummyToolManager(streams=[])
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(str(msg)))

    result = await ping_action(tm)
    assert result is False
    assert any("No matching servers" in p for p in printed)


@pytest.mark.asyncio
async def test_ping_action_with_streams_and_no_filter(monkeypatch):
    async def fake_ping(idx, name, r, w, timeout=5.0):
        return (name, True, 99.9)
    monkeypatch.setattr(ping_mod, "_ping_one", fake_ping)

    fs = FakeStream()
    tm = DummyToolManager(
        streams=[(fs, fs)],
        server_names={0: "srv0"},
        server_info=[{"name": "ignored"}]
    )

    output = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: output.append(obj))

    result = await ping_action(tm)
    assert result is True

    tables = [o for o in output if isinstance(o, Table)]
    assert tables, f"Expected a Table in output: {output}"
    tbl = tables[0]
    # Only check row_count and headers
    assert tbl.row_count == 1
    headers = [col.header for col in tbl.columns]
    assert headers == ["Server", "Status", "Latency"]


@pytest.mark.asyncio
async def test_ping_action_with_targets(monkeypatch):
    async def fake_ping(idx, name, r, w, timeout=5.0):
        return (name, True, 55.5)
    monkeypatch.setattr(ping_mod, "_ping_one", fake_ping)

    fs = FakeStream()
    tm = DummyToolManager(
        streams=[(fs, fs), (fs, fs)],
        server_names={0: "a", 1: "b"}
    )

    # No matching → False
    assert await ping_action(tm, targets=["z"]) is False

    output = []
    monkeypatch.setattr(Console, "print", lambda self, obj, **kw: output.append(obj))
    result = await ping_action(tm, targets=["b"])
    assert result is True

    tables = [o for o in output if isinstance(o, Table)]
    assert tables, "Expected a Table"
    tbl = tables[0]
    assert tbl.row_count == 1
    # Confirm header only (we trust correct row)
    headers = [col.header for col in tbl.columns]
    assert headers == ["Server", "Status", "Latency"]

