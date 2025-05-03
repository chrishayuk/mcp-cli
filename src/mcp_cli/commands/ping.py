# mcp_cli/commands/ping.py
"""
Pretty “/ping” command for the interactive CLI.

Highlights
----------
* Shows status and round-trip latency for each server.
* Accepts optional *targets* so you can `/ping sqlite` or `/ping 0`.
* Works with any StreamManager-like object that exposes ``get_streams()``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

# CHUK-MCP helper
from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping

app = typer.Typer(help="Ping MCP servers")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _display_name(idx: int, sm: Any, mapping: Dict[int, str] | None) -> str:
    """Prefer explicit mapping, then `sm.server_names`, then `sm.server_info`."""
    if mapping and idx in mapping:
        return mapping[idx]

    if getattr(sm, "server_names", None):
        name = sm.server_names.get(idx)  # type: ignore[attr-defined]
        if name:
            return name

    info = getattr(sm, "server_info", [])
    if isinstance(info, list) and idx < len(info):
        return info[idx].get("name") or f"server-{idx}"

    return f"server-{idx}"


async def _ping_one(
    idx: int,
    name: str,
    read_stream: Any,
    write_stream: Any,
    timeout: float = 5.0,
) -> Tuple[str, bool, float]:
    """Ping a single stream pair – return (name, ok, ms)."""
    start = time.perf_counter()
    try:
        ok = await asyncio.wait_for(send_ping(read_stream, write_stream), timeout)
    except Exception:  # noqa: BLE001
        ok = False
    duration_ms = (time.perf_counter() - start) * 1_000
    return (name, ok, duration_ms)


# --------------------------------------------------------------------------- #
# main entry-point                                                            #
# --------------------------------------------------------------------------- #
@app.command("run")
async def ping_run(
    stream_manager: Any,
    server_names: Dict[int, str] | None = None,
    *targets: str,
) -> None:
    """
    Ping all connected servers (or just the *targets* you specify).

    Examples
    --------
    `/ping` – ping everyone  
    `/ping sqlite` – ping server whose display-name is “sqlite”  
    `/ping 0` – ping server with index 0
    """
    if not hasattr(stream_manager, "get_streams"):
        raise TypeError(
            "ping_run expects a StreamManager-like object exposing 'get_streams()'"
        )

    streams: List[Tuple[Any, Any]] = list(stream_manager.get_streams())
    console = Console()

    # ── build ping tasks ────────────────────────────────────────────────────
    tasks = []
    for idx, (r, w) in enumerate(streams):
        name = _display_name(idx, stream_manager, server_names)

        if targets:
            # filter: keep if any target matches index or name (case-insensitive)
            if not any(
                t.lower() in (str(idx), name.lower())  # type: ignore[arg-type]
                for t in targets
            ):
                continue

        tasks.append(asyncio.create_task(_ping_one(idx, name, r, w), name=name))

    if not tasks:
        console.print(
            "[red]No matching servers.[/red]  "
            "Use `/servers` to see available names/indices."
        )
        return

    console.print("[cyan]\nPinging servers…[/cyan]")

    # ── gather results ─────────────────────────────────────────────────────
    results = await asyncio.gather(*tasks)

    # ── pretty table ───────────────────────────────────────────────────────
    table = Table(title=None, show_header=True, header_style="bold magenta")
    table.add_column("Server")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")

    for name, ok, dur in sorted(results, key=lambda r: r[0].lower()):
        status = Text("✓", style="green") if ok else Text("✗", style="red")
        latency = f"{dur:6.1f} ms" if ok else "-"
        table.add_row(name, status, latency)

    console.print(table)
