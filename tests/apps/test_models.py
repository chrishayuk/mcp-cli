# tests/apps/test_models.py
"""Tests for MCP Apps Pydantic models."""

from __future__ import annotations

import pytest

from mcp_cli.apps.models import AppInfo, AppState, HostContext


class TestAppState:
    def test_enum_values(self):
        assert AppState.PENDING == "pending"
        assert AppState.INITIALIZING == "initializing"
        assert AppState.READY == "ready"
        assert AppState.CLOSED == "closed"


class TestAppInfo:
    def test_create_minimal(self):
        info = AppInfo(
            tool_name="get-time",
            resource_uri="ui://get-time/app.html",
            server_name="my-server",
            port=9470,
        )
        assert info.tool_name == "get-time"
        assert info.resource_uri == "ui://get-time/app.html"
        assert info.server_name == "my-server"
        assert info.state == AppState.PENDING
        assert info.port == 9470
        assert info.html_content == ""
        assert info.csp is None
        assert info.permissions is None

    def test_url_property(self):
        info = AppInfo(
            tool_name="test",
            resource_uri="ui://test/app.html",
            server_name="s",
            port=9471,
        )
        assert info.url == "http://localhost:9471"

    def test_state_mutation(self):
        info = AppInfo(
            tool_name="test",
            resource_uri="ui://test/app.html",
            server_name="s",
            port=9470,
        )
        assert info.state == AppState.PENDING
        info.state = AppState.READY
        assert info.state == AppState.READY

    def test_with_html_content(self):
        info = AppInfo(
            tool_name="test",
            resource_uri="ui://test/app.html",
            server_name="s",
            port=9470,
            html_content="<html><body>Hello</body></html>",
        )
        assert "<html>" in info.html_content

    def test_with_csp(self):
        csp = {"connectDomains": ["https://api.example.com"]}
        info = AppInfo(
            tool_name="test",
            resource_uri="ui://test/app.html",
            server_name="s",
            port=9470,
            csp=csp,
        )
        assert info.csp == csp


class TestHostContext:
    def test_defaults(self):
        ctx = HostContext()
        assert ctx.theme == "dark"
        assert ctx.locale == "en"
        assert ctx.platform == "desktop"
        assert ctx.display_mode == "inline"
        assert "inline" in ctx.available_display_modes
        assert "fullscreen" in ctx.available_display_modes

    def test_allows_extra_fields(self):
        ctx = HostContext(custom_field="value")
        assert ctx.custom_field == "value"  # type: ignore[attr-defined]
