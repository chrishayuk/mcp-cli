# src/mcp_cli/commands/ping.py
import asyncio
import time
import logging
from typing import Any, Dict, Sequence, Tuple

from rich.console import Console
from rich.table import Table
from rich.text import Text

from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping
from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


def _display_name(
    idx: int,
    tm: ToolManager,
    mapping: Dict[int, str] | None
) -> str:
    if mapping and idx in mapping:
        return mapping[idx]
    if tm.server_names:
        name = tm.server_names.get(idx)
        if name:
            return name
    infos = tm.get_server_info()
    if idx < len(infos):
        return infos[idx].name
    return f"server-{idx}"


async def _ping_one(
    idx: int,
    name: str,
    read_stream: Any,
    write_stream: Any,
    timeout: float = 5.0,
) -> Tuple[str, bool, float]:
    start = time.perf_counter()
    try:
        ok = await asyncio.wait_for(send_ping(read_stream, write_stream), timeout)
    except Exception:
        ok = False
    latency = (time.perf_counter() - start) * 1000
    return name, ok, latency


async def ping_action(
    tm: ToolManager,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
) -> bool:
    """
    Ping all (or filtered) servers managed by `tm`.
    Returns True if we pinged at least one server, False otherwise.
    """
    streams = list(tm.get_streams())
    console = Console()

    tasks = []
    for idx, (r, w) in enumerate(streams):
        name = _display_name(idx, tm, server_names)
        if targets and not any(t.lower() in (str(idx), name.lower()) for t in targets):
            continue
        tasks.append(asyncio.create_task(_ping_one(idx, name, r, w), name=name))

    if not tasks:
        console.print("[red]No matching servers.[/red] Use `servers` to list names/indices.")
        return False

    console.print("[cyan]\nPinging servers…[/cyan]")
    results = await asyncio.gather(*tasks)

    table = Table(header_style="bold magenta")
    table.add_column("Server")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")

    for name, ok, ms in sorted(results, key=lambda x: x[0].lower()):
        status = Text("✓", style="green") if ok else Text("✗", style="red")
        lat = f"{ms:6.1f} ms" if ok else "-"
        table.add_row(name, status, lat)

    console.print(table)
    return True
