# src/mcp_cli/commands/ping.py
"""
Ping MCP servers (usable from both CLI and chat)
================================================

* Works cross-platform thanks to :pyfunc:`mcp_cli.utils.rich_helpers.get_console`
  (handles Windows ANSI quirks, pipes, narrow TTYs, …).
* Has both an **async** coroutine (*ping_action_async*) and a small synchronous
  wrapper (*ping_action*) for legacy call-sites.
* Lets callers pass a **timeout** so very-slow links (VPN, satellite) can still
  register a success.

Typical interactive usage
-------------------------
>>> /ping              # ping every server
>>> /ping 0 api        # ping the server with index 0 and the one named “api”
>>> /ping --timeout 10 # allow up to 10 s per round-trip
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, List, Sequence, Tuple
from rich.table import Table
from rich.text import Text

# mcp cli
from mcp_cli.utils.rich_helpers import get_console
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking

# mcp client
from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping

# logger
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# helper utilities
# ════════════════════════════════════════════════════════════════════════
def _display_server_name(
    idx: int,
    explicit_map: Dict[int, str] | None,
    fallback_infos: list,
) -> str:
    """
    Resolve a human-readable server label.

    Precedence
    ----------
    1. **explicit_map**  → from CLI flags or `ToolManager.server_names`
    2. `get_server_info()` name
    3. `"server-{idx}"`
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
    timeout: float,
) -> Tuple[str, bool, float]:
    """Measure round-trip latency to a single server."""
    start = time.perf_counter()
    try:
        ok = await asyncio.wait_for(send_ping(read_stream, write_stream), timeout)
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - start) * 1000
    return name, ok, latency_ms


# ════════════════════════════════════════════════════════════════════════
# async implementation (canonical)
# ════════════════════════════════════════════════════════════════════════
async def ping_action_async(
    tm: ToolManager,
    *,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
    timeout: float = 5.0,
) -> bool:
    """
    Ping all (or selected) MCP servers and render a Rich table.

    Parameters
    ----------
    tm
        The initialised :class:`~mcp_cli.tools.manager.ToolManager`.
    server_names
        Optional mapping *index → friendly-name* (overrides defaults).
    targets
        Optional list/tuple of filters (server index **or** name, case-insensitive).
    timeout
        Seconds to wait for each round-trip before marking it as “timed-out”.

    Returns
    -------
    bool
        ``True`` if at least one server was pinged.
    """
    console = get_console()
    streams = list(tm.get_streams())
    server_infos = await tm.get_server_info()      # single RPC

    tasks: List[asyncio.Task] = []
    for idx, (r, w) in enumerate(streams):
        name = _display_server_name(idx, server_names, server_infos)

        # Apply filter(s) if provided
        if targets and not any(t.lower() in (str(idx), name.lower()) for t in targets):
            continue

        tasks.append(
            asyncio.create_task(
                _ping_one(idx, name, r, w, timeout=timeout),
                name=name,
            )
        )

    if not tasks:
        console.print(
            "[red]No matching servers.[/red] "
            "Use [cyan]/servers[/cyan] to list names / indices."
        )
        return False

    console.print("[cyan]Pinging servers…[/cyan]")
    results = await asyncio.gather(*tasks)

    # ── render ───────────────────────────────────────────────────────
    table = Table(header_style="bold magenta")
    table.add_column("Server")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")

    for name, ok, ms in sorted(results, key=lambda x: x[0].lower()):
        status  = Text("✓", style="green") if ok else Text("✗", style="red")
        latency = f"{ms:6.1f} ms" if ok else "—"
        table.add_row(name, status, latency)

    console.print(table)
    return True


# ════════════════════════════════════════════════════════════════════════
# synchronous wrapper (legacy / CLI)
# ════════════════════════════════════════════════════════════════════════
def ping_action(
    tm: ToolManager,
    *,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
    timeout: float = 5.0,
) -> bool:
    """Blocking convenience wrapper around :pyfunc:`ping_action_async`."""
    return run_blocking(
        ping_action_async(
            tm,
            server_names=server_names,
            targets=targets,
            timeout=timeout,
        )
    )
