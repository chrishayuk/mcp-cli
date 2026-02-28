# mcp_cli/dashboard/launcher.py
"""Dashboard launch logic: port selection, HTTP server start, browser open."""

from __future__ import annotations

import logging
import webbrowser

from mcp_cli.dashboard.server import DashboardServer

logger = logging.getLogger(__name__)


async def launch_dashboard(
    port: int = 0,
    no_browser: bool = False,
) -> tuple[DashboardServer, int]:
    """Start the dashboard server and (optionally) open the browser.

    Args:
        port: Preferred port (0 = auto-select starting at DEFAULT_DASHBOARD_PORT_START).
        no_browser: If True, do not open the browser (URL is logged at INFO level).

    Returns:
        (server, bound_port) — the started server and its actual port.
    """
    server = DashboardServer()
    bound_port = await server.start(port)

    url = f"http://localhost:{bound_port}"
    logger.info("Dashboard available at %s", url)

    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Could not open browser: %s", exc)

    return server, bound_port
