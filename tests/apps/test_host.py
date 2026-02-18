# tests/apps/test_host.py
"""Tests for MCP Apps host server (no real WS servers)."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from mcp_cli.apps.host import AppHostServer, _SAFE_CSP_SOURCE
from mcp_cli.apps.models import AppInfo, AppState


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeToolManager:
    """Stub for ToolManager — only needs read_resource for launch tests."""

    def __init__(self):
        self._resource: dict[str, Any] = {
            "contents": [
                {
                    "uri": "ui://test/app.html",
                    "text": "<html><body>Hello</body></html>",
                }
            ]
        }

    async def read_resource(self, uri, server_name=None):
        return self._resource


# ── Init Tests ─────────────────────────────────────────────────────────────


class TestAppHostServerInit:
    def test_construction(self):
        tm = FakeToolManager()
        host = AppHostServer(tm)
        assert host.tool_manager is tm
        assert host._apps == {}
        assert host._bridges == {}
        assert host._servers == []

    def test_get_running_apps_empty(self):
        host = AppHostServer(FakeToolManager())
        assert host.get_running_apps() == []

    def test_get_bridge_missing(self):
        host = AppHostServer(FakeToolManager())
        assert host.get_bridge("nonexistent") is None


# ── Extract Helpers ────────────────────────────────────────────────────────


class TestExtractHtml:
    def test_text_content(self):
        resource = {"contents": [{"uri": "ui://test", "text": "<html>OK</html>"}]}
        assert AppHostServer._extract_html(resource) == "<html>OK</html>"

    def test_blob_content(self):
        html = "<html>blob</html>"
        b64 = base64.b64encode(html.encode()).decode()
        resource = {"contents": [{"uri": "ui://test", "blob": b64}]}
        assert AppHostServer._extract_html(resource) == html

    def test_empty_contents(self):
        assert AppHostServer._extract_html({"contents": []}) == ""

    def test_nested_result(self):
        resource = {
            "result": {
                "contents": [{"uri": "ui://test", "text": "<html>nested</html>"}]
            }
        }
        assert AppHostServer._extract_html(resource) == "<html>nested</html>"

    def test_no_contents(self):
        assert AppHostServer._extract_html({}) == ""


class TestExtractCsp:
    def test_csp_present(self):
        resource = {
            "contents": [
                {
                    "uri": "ui://test",
                    "text": "<html></html>",
                    "_meta": {
                        "ui": {"csp": {"connectDomains": ["https://api.example.com"]}}
                    },
                }
            ]
        }
        csp = AppHostServer._extract_csp(resource)
        assert csp == {"connectDomains": ["https://api.example.com"]}

    def test_csp_absent(self):
        resource = {"contents": [{"uri": "ui://test", "text": "<html></html>"}]}
        assert AppHostServer._extract_csp(resource) is None

    def test_csp_empty_contents(self):
        assert AppHostServer._extract_csp({"contents": []}) is None


class TestExtractPermissions:
    def test_permissions_present(self):
        resource = {
            "contents": [
                {
                    "uri": "ui://test",
                    "text": "<html></html>",
                    "_meta": {"ui": {"permissions": {"clipboard": True}}},
                }
            ]
        }
        perms = AppHostServer._extract_permissions(resource)
        assert perms == {"clipboard": True}

    def test_permissions_absent(self):
        resource = {"contents": [{"uri": "ui://test", "text": "<html></html>"}]}
        assert AppHostServer._extract_permissions(resource) is None


# ── CSP Domain Sanitization ───────────────────────────────────────────────


class TestCspDomainSanitization:
    def test_valid_domains(self):
        assert _SAFE_CSP_SOURCE.match("https://api.example.com")
        assert _SAFE_CSP_SOURCE.match("*.example.com")
        assert _SAFE_CSP_SOURCE.match("http://localhost:8080")

    def test_invalid_domains(self):
        assert not _SAFE_CSP_SOURCE.match('https://evil.com"; script-src *')
        assert not _SAFE_CSP_SOURCE.match("https://evil.com' onclick=alert(1)")
        assert not _SAFE_CSP_SOURCE.match("<script>")
        assert not _SAFE_CSP_SOURCE.match(
            "https://example.com; script-src 'unsafe-inline'"
        )


# ── Launch & Close ─────────────────────────────────────────────────────────


class TestCloseApp:
    @pytest.mark.asyncio
    async def test_close_app_removes(self):
        host = AppHostServer(FakeToolManager())
        info = AppInfo(
            tool_name="test-app",
            resource_uri="ui://test",
            server_name="srv",
            port=9470,
        )
        host._apps["test-app"] = info
        host._bridges["test-app"] = object()
        await host.close_app("test-app")
        assert "test-app" not in host._apps
        assert "test-app" not in host._bridges

    @pytest.mark.asyncio
    async def test_close_app_sets_closed(self):
        host = AppHostServer(FakeToolManager())
        info = AppInfo(
            tool_name="test-app",
            resource_uri="ui://test",
            server_name="srv",
            port=9470,
            state=AppState.READY,
        )
        host._apps["test-app"] = info
        await host.close_app("test-app")
        assert info.state == AppState.CLOSED

    @pytest.mark.asyncio
    async def test_close_app_idempotent(self):
        host = AppHostServer(FakeToolManager())
        # Closing non-existent app should not raise
        await host.close_app("nonexistent")


class TestCloseAll:
    @pytest.mark.asyncio
    async def test_close_all_marks_closed_first(self):
        host = AppHostServer(FakeToolManager())
        info1 = AppInfo(
            tool_name="app1",
            resource_uri="ui://1",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        info2 = AppInfo(
            tool_name="app2",
            resource_uri="ui://2",
            server_name="s",
            port=9471,
            state=AppState.READY,
        )
        host._apps["app1"] = info1
        host._apps["app2"] = info2

        await host.close_all()
        assert info1.state == AppState.CLOSED
        assert info2.state == AppState.CLOSED
        assert len(host._apps) == 0
        assert len(host._bridges) == 0


class TestGetRunningApps:
    def test_filters_closed(self):
        host = AppHostServer(FakeToolManager())
        info_ready = AppInfo(
            tool_name="ready-app",
            resource_uri="ui://r",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        info_closed = AppInfo(
            tool_name="closed-app",
            resource_uri="ui://c",
            server_name="s",
            port=9471,
            state=AppState.CLOSED,
        )
        host._apps["ready-app"] = info_ready
        host._apps["closed-app"] = info_closed
        running = host.get_running_apps()
        assert len(running) == 1
        assert running[0].tool_name == "ready-app"
