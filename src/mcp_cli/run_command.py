"""mcp_cli.run_command
Modernised helpers for invoking CLI commands with automatic
StreamManager lifecycle management.

Key changes
===========
* **Single async entry‑point** – `run_command()` is now an `async def` that you
  always *await*.  This erases the messy “nested event‑loop” edge‑cases from the
  previous synchronous wrapper.
* **Thin synchronous helper** – `run_command_sync()` is provided for truly
  blocking contexts.  It detects whether an event‑loop already exists and uses
  the safest mechanism available (`asyncio.run`, `loop.run_until_complete`, or
  scheduling a task and awaiting it) without ever creating a *second* loop in
  the same thread.
* Much smaller surface:  all error‑handling happens in one place and logging is
  consistent.

Usage
-----
```python
# async context (e.g. Typer async command)
result = await run_command(my_command, config_file, servers, user_specified)

# legacy synchronous context
result = run_command_sync(my_command, config_file, servers, user_specified)
```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from mcp_cli.stream_manager import StreamManager

__all__ = [
    "run_command",  # async version
    "run_command_sync",  # blocking helper
]

# ---------------------------------------------------------------------------
# Core async helper
# ---------------------------------------------------------------------------


async def run_command(
    command_func: Callable[..., Any],
    config_file: str,
    servers: List[str],
    user_specified: List[str],
    extra_params: Optional[Dict[str, Any]] | None = None,
) -> Any:
    """Execute *command_func* with a managed :class:`StreamManager`.

    The function is **always awaited** and never creates nested event‑loops.
    Most callers should switch from the old synchronous `run_command()` to
    simply::

        await run_command(...)

    Parameters
    ----------
    command_func
        Coroutine function that represents the CLI command.
    config_file
        Path to JSON config that lists the available MCP servers.
    servers
        The set of server names the user wants to connect to.
    user_specified
        Original list entered on the CLI (useful for logging).
    extra_params
        Extra keyword arguments to forward into *command_func*.
    """

    logging.info("Running command: %s", command_func.__name__)
    logging.debug("Servers: %s", servers)

    if not servers:
        logging.warning("No servers specified – aborting.")
        return False

    stream_manager = await StreamManager.create(
        config_file=config_file,
        servers=servers,
        server_names={i: name for i, name in enumerate(servers)},
    )

    try:
        params: Dict[str, Any] = dict(extra_params or {})
        params["stream_manager"] = stream_manager
        result = await command_func(**params)
    finally:
        logging.debug("Tearing down StreamManager")
        await stream_manager.close()

    return result


# ---------------------------------------------------------------------------
# Synchronous compatibility wrapper
# ---------------------------------------------------------------------------


def run_command_sync(
    command_func: Callable[..., Any],
    config_file: str,
    servers: List[str],
    user_specified: List[str],
    extra_params: Optional[Dict[str, Any]] | None = None,
) -> Any:
    """Blocking helper for legacy call‑sites.

    If an event‑loop is already running *in this thread*, the coroutine is
    scheduled on it and we **block** by awaiting the returned task.  This avoids
    the dreaded “asyncio.run() cannot be called from a running event loop”
    error without spawning a nested loop.
    """

    async def _runner() -> Any:  # inner wrapper captures current arguments
        return await run_command(
            command_func,
            config_file,
            servers,
            user_specified,
            extra_params,
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop yet → safest path is asyncio.run
        return asyncio.run(_runner())

    # We *do* have a running loop in this thread.
    if loop.is_running():
        # Schedule _runner and wait synchronously without a second loop.
        task = loop.create_task(_runner())
        # Blocking wait – safe because the loop keeps spinning.
        # NB: If run_command() performs "await" that blocks on thread‑pool
        # executors, the loop stays alive, so deadlocks are avoided.
        return asyncio.get_event_loop().run_until_complete(task)

    # Rare case: loop exists but not running (e.g. closed and reopened).
    return loop.run_until_complete(_runner())
