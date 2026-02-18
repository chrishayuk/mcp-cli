# tests/apps/test_session.py
"""Session reliability tests for MCP Apps."""

from __future__ import annotations

import pytest

from mcp_cli.apps.bridge import AppBridge
from mcp_cli.apps.models import AppInfo, AppState


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeToolManager:
    async def execute_tool(self, *a, **kw):
        pass

    async def read_resource(self, *a, **kw):
        return {}


class FakeWs:
    def __init__(self):
        self.sent: list[str] = []
        self.closed = False

    async def send(self, msg: str) -> None:
        self.sent.append(msg)

    async def close(self) -> None:
        self.closed = True


def _make_bridge() -> AppBridge:
    info = AppInfo(
        tool_name="session-test",
        resource_uri="ui://test",
        server_name="srv",
        port=9470,
    )
    return AppBridge(info, FakeToolManager())


# ── Tests ──────────────────────────────────────────────────────────────────


class TestSessionReliability:
    def test_set_ws_resets_state_to_initializing(self):
        """On new WS connection, state resets to INITIALIZING."""
        bridge = _make_bridge()
        bridge.app_info.state = AppState.READY
        bridge.set_ws(FakeWs())
        assert bridge.app_info.state == AppState.INITIALIZING

    @pytest.mark.asyncio
    async def test_push_queued_when_disconnected(self):
        """Tool results queued when WS is not connected."""
        bridge = _make_bridge()
        await bridge.push_tool_result({"data": "test"})
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_drain_pending_flushes(self):
        """Pending notifications are flushed when WS reconnects."""
        bridge = _make_bridge()
        await bridge.push_tool_result("result1")
        await bridge.push_tool_result("result2")
        assert len(bridge._pending_notifications) == 2

        ws = FakeWs()
        bridge.set_ws(ws)
        await bridge.drain_pending()

        assert len(ws.sent) == 2
        assert len(bridge._pending_notifications) == 0

    @pytest.mark.asyncio
    async def test_duplicate_launch_closes_previous(self):
        """Launching app twice closes previous instance."""
        from mcp_cli.apps.host import AppHostServer

        tm = FakeToolManager()
        host = AppHostServer(tm)

        info1 = AppInfo(
            tool_name="dup-app",
            resource_uri="ui://dup",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        host._apps["dup-app"] = info1
        host._bridges["dup-app"] = AppBridge(info1, tm)

        # Close the previous instance via close_app
        await host.close_app("dup-app")
        assert "dup-app" not in host._apps
        assert info1.state == AppState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_reconnects_drain_correctly(self):
        """Multiple disconnect/reconnect cycles drain correctly."""
        bridge = _make_bridge()

        # First cycle: queue while disconnected
        await bridge.push_tool_result("batch1")

        # Connect and drain
        ws1 = FakeWs()
        bridge.set_ws(ws1)
        await bridge.drain_pending()
        assert len(ws1.sent) == 1

        # Second cycle: simulate disconnect (ws goes None for test)
        bridge._ws = None
        await bridge.push_tool_result("batch2")
        assert len(bridge._pending_notifications) == 1

        # Reconnect and drain
        ws2 = FakeWs()
        bridge.set_ws(ws2)
        await bridge.drain_pending()
        assert len(ws2.sent) == 1
        assert len(bridge._pending_notifications) == 0
