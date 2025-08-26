# src/mcp_cli/commands/ping.py
"""
Ping MCP servers to check connectivity and measure latency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

from chuk_mcp.protocol.messages import send_ping

from chuk_term.ui import output, format_table
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.async_utils import run_blocking

logger = logging.getLogger(__name__)


async def ping_action_async(
    tm: ToolManager,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
) -> bool:
    """
    Ping all or specified MCP servers.

    Args:
        tm: ToolManager instance
        server_names: Optional mapping of server indices to names
        targets: Specific servers to ping (empty = all)

    Returns:
        True if at least one server was pinged
    """
    streams = list(tm.get_streams())

    # Get server info
    server_infos = await tm.get_server_info()

    # Build ping tasks
    tasks = []
    for idx, (read_stream, write_stream) in enumerate(streams):
        name = _get_server_name(idx, server_names, server_infos)

        # Apply target filter if specified
        if targets and not _matches_target(idx, name, targets):
            continue

        task = asyncio.create_task(
            _ping_server(idx, name, read_stream, write_stream), name=name
        )
        tasks.append(task)

    # Check if we have servers to ping
    if not tasks:
        output.error("No matching servers found")
        output.hint("Use 'servers' command to list available servers")
        return False

    # Execute pings
    with output.loading(f"Pinging {len(tasks)} server(s)..."):
        results = await asyncio.gather(*tasks)

    # Display results
    _display_results(results)
    return True


async def _ping_server(
    idx: int,
    name: str,
    read_stream: Any,
    write_stream: Any,
    timeout: float = 5.0,
) -> Tuple[str, bool, float]:
    """
    Ping a single server and measure latency.

    Args:
        idx: Server index
        name: Server name
        read_stream: Read stream for server
        write_stream: Write stream for server
        timeout: Ping timeout in seconds

    Returns:
        Tuple of (name, success, latency_ms)
    """
    start = time.perf_counter()

    try:
        success = await asyncio.wait_for(send_ping(read_stream, write_stream), timeout)
    except asyncio.TimeoutError:
        logger.debug(f"Ping timeout for server {name}")
        success = False
    except Exception as e:
        logger.debug(f"Ping failed for server {name}: {e}")
        success = False

    latency_ms = (time.perf_counter() - start) * 1000
    return name, success, latency_ms


def _get_server_name(
    idx: int,
    explicit_names: Dict[int, str] | None,
    server_infos: list,
) -> str:
    """
    Get the display name for a server.

    Priority:
    1. Explicit name from server_names dict
    2. Name from server info
    3. Generic "server-{idx}"
    """
    if explicit_names and idx in explicit_names:
        return explicit_names[idx]

    if idx < len(server_infos):
        return server_infos[idx].name

    return f"server-{idx}"


def _matches_target(idx: int, name: str, targets: Sequence[str]) -> bool:
    """
    Check if a server matches any of the target filters.

    Args:
        idx: Server index
        name: Server name
        targets: Target filters

    Returns:
        True if server matches any target
    """
    for target in targets:
        target_lower = target.lower()
        if target_lower in (str(idx), name.lower()):
            return True
    return False


def _display_results(results: List[Tuple[str, bool, float]]) -> None:
    """
    Display ping results in a formatted table.

    Args:
        results: List of (name, success, latency_ms) tuples
    """
    # Sort results by server name
    sorted_results = sorted(results, key=lambda x: x[0].lower())

    # Build table data
    table_data = []
    successful_count = 0
    total_latency = 0.0

    for name, success, latency_ms in sorted_results:
        if success:
            status = "✓ Online"
            latency = f"{latency_ms:.1f} ms"
            successful_count += 1
            total_latency += latency_ms
        else:
            status = "✗ Offline"
            latency = "-"

        table_data.append({"Server": name, "Status": status, "Latency": latency})

    # Display table
    table = format_table(
        table_data, title="Server Ping Results", columns=["Server", "Status", "Latency"]
    )
    output.print_table(table)

    # Display summary
    output.print()
    if successful_count > 0:
        avg_latency = total_latency / successful_count
        output.success(f"{successful_count}/{len(results)} servers online")
        output.info(f"Average latency: {avg_latency:.1f} ms")
    else:
        output.error("All servers are offline")


def ping_action(
    tm: ToolManager,
    server_names: Dict[int, str] | None = None,
    targets: Sequence[str] = (),
) -> bool:
    """
    Synchronous wrapper for ping_action_async.

    Args:
        tm: ToolManager instance
        server_names: Optional mapping of server indices to names
        targets: Specific servers to ping

    Returns:
        True if at least one server was pinged
    """
    return run_blocking(
        ping_action_async(tm, server_names=server_names, targets=targets)
    )
