# mcp_cli/apps/__init__.py
"""MCP Apps support for mcp-cli (SEP-1865).

Provides the ability to launch interactive HTML UIs from MCP tools
in the user's browser, with full JSON-RPC communication back to
the MCP server.

Install the optional dependency:
    pip install mcp-cli[apps]
"""

from __future__ import annotations

from mcp_cli.apps.models import AppInfo, AppState

__all__ = ["AppInfo", "AppState"]
