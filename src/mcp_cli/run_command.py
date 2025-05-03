# mcp_cli/run_command.py
"""
mcp_cli.run_command
===================

Centralised helpers that run *any* MCP-CLI command – be it

* a Typer **Command** object (has a ``.callback`` attribute),
* an **async** callable,
* a **sync** callable –

inside the right event-loop, and with a fully-initialised
:class:`mcp_cli.tools.manager.ToolManager` (plus underlying
``StreamManager``) injected when requested.

The helpers guarantee that **ToolManager.close() is always called** –
even when initialisation fails or the command raises.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger("mcp_cli.runner")


# ---------------------------------------------------------------------------- #
# Public sync wrapper
# ---------------------------------------------------------------------------- #
def run_command_sync(
    command_func: Any,                 # Typer Command OR async/sync callable
    config_file: str,
    servers: Iterable[str],
    *,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Synchronously run *command_func* in a (possibly new) asyncio loop.

    ``extra_params`` are forwarded verbatim; the helper *adds*
    ``tool_manager`` / ``stream_manager`` automatically when the target’s
    signature asks for them.
    """
    extra_params = extra_params or {}

    async def _runner() -> Any:
        from mcp_cli.run_command import run_command  # lazy import (avoid cycles)
        return await run_command(
            command_func,
            config_file=config_file,
            servers=servers,
            extra_params=extra_params,
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop yet → create one implicitly
        return asyncio.run(_runner())

    # A loop *exists* in this thread
    if loop.is_running():
        # Typical for pytest-asyncio – schedule task, return Future
        return asyncio.ensure_future(_runner())

    # Loop exists but not running (rare) – run it manually
    return loop.run_until_complete(_runner())


# ---------------------------------------------------------------------------- #
# Core async helper
# ---------------------------------------------------------------------------- #
async def run_command(
    command_func: Any,
    *,
    config_file: str,
    servers: Iterable[str],
    extra_params: Dict[str, Any],
) -> Any:
    """
    Execute *command_func* and return its result.

    Steps
    -----
    1.  Resolve the real callable (``.callback`` for Typer Commands)
    2.  Build **one** ToolManager & StreamManager
    3.  Inject them into the callable *iff* the parameter names appear
        in its signature
    4.  Run the callable (async or in executor)
    5.  **Always** call ``tool_manager.close()`` in a ``finally`` block
    """
    # 1️⃣  Resolve callable
    target = getattr(command_func, "callback", command_func)
    logger.info("Running command: %s", target.__name__)
    logger.debug("Servers: %s | Config: %s", list(servers), config_file)

    # 2️⃣  Build ToolManager
    from mcp_cli.tools.manager import ToolManager  # local import avoids cycles

    server_names = extra_params.get("server_names", {})  # may be empty
    tool_mgr = ToolManager(
        config_file=config_file,
        servers=list(servers),
        server_names=server_names,
    )

    init_ok = await tool_mgr.initialize()
    if not init_ok:
        # Even when init fails, we *must* close before raising
        try:
            await tool_mgr.close()
        finally:
            raise RuntimeError("Failed to initialise ToolManager – cannot run command.")

    # 3️⃣  Prepare kwargs for the target
    sig = inspect.signature(target).parameters
    call_kwargs = dict(extra_params)  # shallow copy

    if "tool_manager" in sig:
        call_kwargs["tool_manager"] = tool_mgr
    if "stream_manager" in sig:
        call_kwargs["stream_manager"] = tool_mgr.stream_manager

    # 4️⃣  Run the target – ensure cleanup in *finally*
    try:
        if asyncio.iscoroutinefunction(target):
            return await target(**call_kwargs)
        # Sync callable → run in default executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: target(**call_kwargs))

    finally:
        # 5️⃣  Always close the ToolManager
        try:
            await tool_mgr.close()
        except Exception:  # noqa: BLE001
            pass
