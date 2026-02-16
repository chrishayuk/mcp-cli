# tests/utils/test_trusted_domains.py
"""Tests for trusted domain bypass in tool confirmation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.utils.preferences import (
    PreferenceManager,
    ToolConfirmationPreferences,
)


# ── ToolConfirmationPreferences defaults ──────────────────────────────────────


class TestTrustedDomainsDefaults:
    def test_default_includes_chukai(self):
        prefs = ToolConfirmationPreferences()
        assert "chukai.io" in prefs.trusted_domains

    def test_custom_trusted_domains(self):
        prefs = ToolConfirmationPreferences(trusted_domains=["example.com"])
        assert "example.com" in prefs.trusted_domains
        assert "chukai.io" not in prefs.trusted_domains


# ── PreferenceManager.is_trusted_domain ───────────────────────────────────────


class TestIsTrustedDomain:
    @pytest.fixture()
    def manager(self, tmp_path):
        return PreferenceManager(config_dir=tmp_path)

    def test_exact_match(self, manager):
        assert manager.is_trusted_domain("https://chukai.io/mcp") is True

    def test_subdomain_match(self, manager):
        assert manager.is_trusted_domain("https://api.chukai.io/v1") is True

    def test_deep_subdomain_match(self, manager):
        assert manager.is_trusted_domain("https://a.b.chukai.io/path") is True

    def test_non_trusted_domain(self, manager):
        assert manager.is_trusted_domain("https://evil.com/mcp") is False

    def test_similar_but_different_domain(self, manager):
        # "notchukai.io" should NOT match "chukai.io"
        assert manager.is_trusted_domain("https://notchukai.io/mcp") is False

    def test_none_url(self, manager):
        assert manager.is_trusted_domain(None) is False

    def test_empty_url(self, manager):
        assert manager.is_trusted_domain("") is False

    def test_no_scheme(self, manager):
        # urlparse needs scheme to parse hostname correctly
        assert manager.is_trusted_domain("chukai.io/path") is False

    def test_http_scheme(self, manager):
        assert manager.is_trusted_domain("http://chukai.io/mcp") is True


# ── PreferenceManager domain management ───────────────────────────────────────


class TestTrustedDomainManagement:
    @pytest.fixture()
    def manager(self, tmp_path):
        return PreferenceManager(config_dir=tmp_path)

    def test_get_trusted_domains(self, manager):
        domains = manager.get_trusted_domains()
        assert "chukai.io" in domains

    def test_add_trusted_domain(self, manager):
        manager.add_trusted_domain("example.com")
        assert "example.com" in manager.get_trusted_domains()

    def test_add_duplicate_domain(self, manager):
        manager.add_trusted_domain("chukai.io")
        assert manager.get_trusted_domains().count("chukai.io") == 1

    def test_remove_trusted_domain(self, manager):
        assert manager.remove_trusted_domain("chukai.io") is True
        assert "chukai.io" not in manager.get_trusted_domains()

    def test_remove_nonexistent_domain(self, manager):
        assert manager.remove_trusted_domain("nope.com") is False

    def test_persistence(self, tmp_path):
        m1 = PreferenceManager(config_dir=tmp_path)
        m1.add_trusted_domain("example.com")

        m2 = PreferenceManager(config_dir=tmp_path)
        assert "example.com" in m2.get_trusted_domains()


# ── ToolProcessor._should_confirm_tool integration ───────────────────────────


class TestToolProcessorTrustedDomain:
    """Test that _should_confirm_tool respects trusted domains."""

    def _make_processor(self):
        from mcp_cli.chat.tool_processor import ToolProcessor

        context = MagicMock()
        context.tool_manager = MagicMock()
        ui = MagicMock()
        return ToolProcessor(context, ui)

    @patch("mcp_cli.chat.tool_processor.get_preference_manager")
    def test_trusted_domain_skips_confirmation(self, mock_get_prefs):
        prefs = MagicMock()
        prefs.is_trusted_domain.return_value = True
        mock_get_prefs.return_value = prefs

        proc = self._make_processor()
        result = proc._should_confirm_tool("any_tool", "https://api.chukai.io/v1")

        assert result is False
        prefs.should_confirm_tool.assert_not_called()

    @patch("mcp_cli.chat.tool_processor.get_preference_manager")
    def test_untrusted_domain_checks_preferences(self, mock_get_prefs):
        prefs = MagicMock()
        prefs.is_trusted_domain.return_value = False
        prefs.should_confirm_tool.return_value = True
        mock_get_prefs.return_value = prefs

        proc = self._make_processor()
        result = proc._should_confirm_tool("delete_files", "https://evil.com/mcp")

        assert result is True
        prefs.should_confirm_tool.assert_called_once_with("delete_files")

    @patch("mcp_cli.chat.tool_processor.get_preference_manager")
    def test_no_url_checks_preferences(self, mock_get_prefs):
        prefs = MagicMock()
        prefs.should_confirm_tool.return_value = False
        mock_get_prefs.return_value = prefs

        proc = self._make_processor()
        result = proc._should_confirm_tool("get_data", None)

        assert result is False
        prefs.should_confirm_tool.assert_called_once_with("get_data")


# ── ToolProcessor._get_server_url_for_tool ────────────────────────────────────


class TestGetServerUrlForTool:
    def test_returns_url_for_http_server(self):
        from mcp_cli.chat.tool_processor import ToolProcessor

        server = MagicMock()
        server.namespace = "her"
        server.name = "her"
        server.url = "https://her.chukai.io/mcp"

        context = MagicMock()
        context.tool_manager = MagicMock()
        context.tool_to_server_map = {"her_search": "her"}
        context.server_info = [server]

        proc = ToolProcessor(context, MagicMock())
        assert (
            proc._get_server_url_for_tool("her_search") == "https://her.chukai.io/mcp"
        )

    def test_returns_none_for_unknown_tool(self):
        from mcp_cli.chat.tool_processor import ToolProcessor

        context = MagicMock()
        context.tool_manager = MagicMock()
        context.tool_to_server_map = {}
        context.server_info = []

        proc = ToolProcessor(context, MagicMock())
        assert proc._get_server_url_for_tool("unknown_tool") is None

    def test_returns_none_for_stdio_server(self):
        from mcp_cli.chat.tool_processor import ToolProcessor

        server = MagicMock()
        server.namespace = "sqlite"
        server.name = "sqlite"
        server.url = None

        context = MagicMock()
        context.tool_manager = MagicMock()
        context.tool_to_server_map = {"list_tables": "sqlite"}
        context.server_info = [server]

        proc = ToolProcessor(context, MagicMock())
        assert proc._get_server_url_for_tool("list_tables") is None
