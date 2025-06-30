# src/mcp_cli/commands/ping.py
"""
Ping every connected MCP server (or a filtered subset) and show latency.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

from rich.console import Console
from rich.table import Table
from rich.text import Text

# Updated import for new chuk-mcp APIs
from chuk_mcp.protocol.messages import send_ping
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────
def display_server_name(
    idx: int,
    explicit_map: Dict[int, str] | None,
    fallback_infos: list,
) -> str:
    """
    Resolve a human-readable server label.

    Precedence:
      1. `explicit_map` (passed in via CLI flags or ToolManager.server_names)
      2. The name reported by get_server_info()
      3. "server-{idx}"
    """
    if explicit_map and idx in explicit_map:
        return explicit_map[idx]
    if idx < len(fallback_infos):
        return fallback_infos[idx].name
    return f"server-{idx}"


async def _ping_one(
    idx: int,
    name: str,
    read_stream: Any,
    write_stream: Any,
    *,
    timeout: float = 5.0,
) -> Tuple[str, bool, float]:
    """Low-level ping for one stream pair."""
    start = time.perf_counter()
    try:
        ok = await asyncio.wait_for(send_ping(read_stream, write_stream), timeout)
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - start) * 1000
    return name, ok, latency_ms


# ──────────────────────────────────────────────────────────────────
# async (canonical) implementation
# ──────────────────────────────────────────────────────────────────
async def ping_action_async(
    tm: ToolManager,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
) -> bool:
    """
    Ping all (or filtered) servers.

    Returns **True** if at least one server was pinged.
    """
    streams = list(tm.get_streams())
    console = Console()

    # Pre-fetch server info once (await!)
    server_infos = await tm.get_server_info()

    tasks = []
    for idx, (r, w) in enumerate(streams):
        name = display_server_name(idx, server_names, server_infos)

        # filter if user passed explicit targets
        if targets and not any(t.lower() in (str(idx), name.lower()) for t in targets):
            continue

        tasks.append(asyncio.create_task(_ping_one(idx, name, r, w, timeout=5.0), name=name))

    if not tasks:
        console.print(
            "[red]No matching servers.[/red] "
            "Use `servers` to list names/indices."
        )
        return False

    console.print("[cyan]\nPinging servers…[/cyan]")
    results = await asyncio.gather(*tasks)

    # Render results
    table = Table(header_style="bold magenta")
    table.add_column("Server")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")

    for name, ok, ms in sorted(results, key=lambda x: x[0].lower()):
        status = Text("✓", style="green") if ok else Text("✗", style="red")
        latency = f"{ms:6.1f} ms" if ok else "-"
        table.add_row(name, status, latency)

    console.print(table)
    return True


# ──────────────────────────────────────────────────────────────────
# legacy sync wrapper
# ──────────────────────────────────────────────────────────────────
def ping_action(
    tm: ToolManager,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
) -> bool:
    """
    Synchronous helper for old call-sites.

    Raises if invoked from inside a running event-loop.
    """
    return run_blocking(ping_action_async(tm, server_names=server_names, targets=targets))