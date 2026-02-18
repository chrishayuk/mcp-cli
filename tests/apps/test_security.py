# tests/apps/test_security.py
"""Security tests for MCP Apps."""

from __future__ import annotations

import html as html_mod

from mcp_cli.apps.bridge import _VALID_TOOL_NAME
from mcp_cli.apps.host import _SAFE_CSP_SOURCE


class TestToolNameHtmlEscaping:
    """Verify tool names are HTML-escaped before template injection."""

    def test_script_tag_escaped(self):
        """XSS via <script> in tool_name should be escaped."""
        malicious = '<script>alert("xss")</script>'
        escaped = html_mod.escape(malicious, quote=True)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_html_attributes_escaped(self):
        malicious = '" onload="alert(1)'
        escaped = html_mod.escape(malicious, quote=True)
        assert '"' not in escaped
        assert "&quot;" in escaped

    def test_normal_name_unchanged(self):
        name = "show_chart"
        escaped = html_mod.escape(name, quote=True)
        assert escaped == name


class TestCspDomainInjection:
    """Verify CSP domain values are sanitized."""

    def test_semicolon_injection_rejected(self):
        """Attacker tries to add new CSP directives via semicolon."""
        assert not _SAFE_CSP_SOURCE.match("https://evil.com; script-src *")

    def test_quote_injection_rejected(self):
        assert not _SAFE_CSP_SOURCE.match('https://evil.com" onclick=bad()')

    def test_angle_bracket_rejected(self):
        assert not _SAFE_CSP_SOURCE.match("<script>alert(1)</script>")

    def test_space_injection_rejected(self):
        assert not _SAFE_CSP_SOURCE.match("evil.com script-src *")

    def test_valid_domain_accepted(self):
        assert _SAFE_CSP_SOURCE.match("https://api.example.com")
        assert _SAFE_CSP_SOURCE.match("*.cdn.example.com")
        assert _SAFE_CSP_SOURCE.match("http://localhost:3000")


class TestToolNameValidation:
    """Verify tool name regex rejects dangerous input."""

    def test_valid_names(self):
        assert _VALID_TOOL_NAME.match("get-time")
        assert _VALID_TOOL_NAME.match("my_tool.v2")
        assert _VALID_TOOL_NAME.match("server/tool-name")

    def test_invalid_names(self):
        assert not _VALID_TOOL_NAME.match("rm -rf /")
        assert not _VALID_TOOL_NAME.match("tool; drop table")
        assert not _VALID_TOOL_NAME.match("tool<script>")
        assert not _VALID_TOOL_NAME.match("")
