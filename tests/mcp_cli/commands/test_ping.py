# tests/commands/test_ping.py
import pytest

# Component under test
from mcp_cli.commands.ping import ping_action_async

# ---------------------------------------------------------------------------
# Spy / stubs
# ---------------------------------------------------------------------------

class _DummyServerInfo:
    def __init__(self, name: str):
        self.name = name
        self.id = 0
        self.status = "OK"
        self.tool_count = 0


class DummyToolManager:
    """Minimal stand-in that satisfies ping_action_async."""

    def __init__(self):
        # Two mock (read, write) stream pairs – the concrete objects are never
        # touched because we monkey-patch _ping_one.
        self._streams = [(None, None), (None, None)]
        self._server_info = [_DummyServerInfo("ServerA"), _DummyServerInfo("ServerB")]

    # Called *synchronously*
    def get_streams(self):
        return self._streams

    # Awaited inside ping_action_async
    async def get_server_info(self):
        return self._server_info


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_tm():
    """Provide a fresh DummyToolManager for each test."""
    return DummyToolManager()


@pytest.fixture()
def ping_spy(monkeypatch):
    """Replace _ping_one with a deterministic spy recording the calls."""
    calls = []

    async def _dummy_ping(idx, name, _r, _w, *, timeout):  # noqa: WPS430
        calls.append((idx, name))
        # Always report success with constant latency for simplicity
        return name, True, 42.0

    monkeypatch.setattr("mcp_cli.commands.ping._ping_one", _dummy_ping)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ping_all_servers(dummy_tm, ping_spy):
    """/ping with no filters should contact every server."""
    ok = await ping_action_async(dummy_tm)
    assert ok is True
    assert len(ping_spy) == 2
    assert {n for _, n in ping_spy} == {"ServerA", "ServerB"}


@pytest.mark.asyncio
async def test_ping_filtered_by_index(dummy_tm, ping_spy):
    """Filter accepts the numeric *index* of the server."""
    ok = await ping_action_async(dummy_tm, targets=["0"])
    assert ok is True
    assert ping_spy == [(0, "ServerA")]


@pytest.mark.asyncio
async def test_ping_filtered_by_name(dummy_tm, ping_spy):
    """Filter is case-insensitive for *names*."""
    ok = await ping_action_async(dummy_tm, targets=["serverb"])
    assert ok is True
    assert ping_spy == [(1, "ServerB")]


@pytest.mark.asyncio
async def test_ping_no_match_returns_false(dummy_tm, ping_spy):
    """If no targets match, command prints a warning and returns False."""
    ok = await ping_action_async(dummy_tm, targets=["does-not-exist"])
    assert ok is False
    assert ping_spy == []
