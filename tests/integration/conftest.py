# tests/integration/conftest.py
"""Fixtures for integration tests.

All tests in this directory require the @pytest.mark.integration marker
and are skipped by default. Run with:

    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import pytest

from mcp_cli.tools.manager import ToolManager


@pytest.fixture
async def tool_manager_sqlite(tmp_path):
    """Create a ToolManager connected to a real SQLite MCP server.

    Requires 'sqlite' server defined in server_config.json.
    Yields the initialized ToolManager, then closes it.
    """
    tm = ToolManager(
        config_file="server_config.json",
        servers=["sqlite"],
        initialization_timeout=30.0,
    )
    ok = await tm.initialize()
    if not ok:
        pytest.skip("Could not initialize sqlite server (not available)")
    try:
        yield tm
    finally:
        await tm.close()
