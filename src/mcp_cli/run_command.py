# mcp_cli/run_command.py - COMPLETE FIXED VERSION
"""
Main entry-point helpers for all CLI sub-commands.

These helpers encapsulate

* construction / cleanup of the shared **ToolManager**
* hand-off to individual command modules
* a thin synchronous wrapper so `uv run mcp-cli …` works

ENHANCED: Now properly handles namespace selection for HTTP vs STDIO servers.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.panel import Panel
from chuk_term.ui import output

from mcp_cli.tools.manager import set_tool_manager  # only the setter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# internal helpers / globals                                                  #
# --------------------------------------------------------------------------- #
_ALL_TM: List[Any] = []  # referenced by the unit-tests


# --------------------------------------------------------------------------- #
# Tool-manager helpers                                                        #
# --------------------------------------------------------------------------- #
async def _init_tool_manager(
    config_file: str,
    servers: List[str],
    server_names: Optional[Dict[int, str]] = None,
):
    """
    Dynamically import **ToolManager** (so monkey-patching works) and create it.
    ENHANCED: Automatically selects appropriate namespace based on server type.
    """
    tm_mod = importlib.import_module("mcp_cli.tools.manager")
    ToolManager = getattr(tm_mod, "ToolManager")  # patched in tests

    tm = ToolManager(config_file, servers, server_names)  # type: ignore[call-arg]

    # ENHANCED: Let ToolManager automatically select the namespace
    # It will use the server name for HTTP servers, "stdio" for STDIO servers
    ok = await tm.initialize()  # Remove the hardcoded namespace parameter

    if not ok:
        # Check if this is just because there are no servers
        if not servers:
            logger.info("No servers configured - continuing with empty tool manager")
            # Still record and return the manager for chat without tools
            set_tool_manager(tm)
            _ALL_TM.append(tm)
            return tm

        # record it for the tests
        _ALL_TM.append(tm)
        # ensure close() is still invoked
        try:
            await tm.close()
        finally:
            raise RuntimeError("Failed to initialise ToolManager")

    set_tool_manager(tm)
    _ALL_TM.append(tm)
    return tm


async def _safe_close(tm) -> None:
    """Close the ToolManager, swallowing any exception during shutdown."""
    try:
        await tm.close()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# command dispatch                                                            #
# --------------------------------------------------------------------------- #
async def run_command(
    async_command: Callable[..., Any],
    *,
    config_file: str,
    servers: List[str],
    extra_params: Optional[Dict[str, Any]],
) -> Any:
    """
    Initialise the ToolManager, then call *async_command(tool_manager, …)*.

    The *async_command* may itself be `async` **or** synchronous – both work.
    The ToolManager is always closed, even when the callable raises.
    """
    tm = None
    try:
        server_names = (extra_params or {}).get("server_names")

        # ------------------------------------------------------------------
        # build ToolManager  (patch-friendly, see helper)
        # ------------------------------------------------------------------
        tm = await _init_tool_manager(config_file, servers, server_names)

        # ------------------------------------------------------------------
        # special-case: interactive "app" object
        # ------------------------------------------------------------------
        name = getattr(async_command, "__name__", "")
        module = getattr(async_command, "__module__", "")
        if name == "app" and "interactive" in module:
            provider = (extra_params or {}).get("provider", "openai")
            model = (extra_params or {}).get("model", "gpt-4o-mini")
            result = await _enter_interactive_mode(tm, provider=provider, model=model)
            return result

        # ------------------------------------------------------------------
        # normal pathway
        # ------------------------------------------------------------------
        call_kwargs = (extra_params or {}).copy()
        call_kwargs["tool_manager"] = tm

        maybe_coro = async_command(**call_kwargs)

        # support sync callables transparently
        if asyncio.iscoroutine(maybe_coro):
            return await maybe_coro
        else:
            loop = asyncio.get_running_loop()
            # run in default executor so we don't block the event-loop
            return await loop.run_in_executor(None, lambda: maybe_coro)
    finally:
        if tm:
            await _safe_close(tm)


def run_command_sync(
    async_command: Callable[..., Any],
    config_file: str,
    servers: List[str],
    *,
    extra_params: Optional[Dict[str, Any]],
) -> Any:
    """
    Synchronous convenience wrapper (used by `mcp-cli` entry-point and tests).

    Spins up its own event-loop when necessary.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        run_command(
            async_command,
            config_file=config_file,
            servers=servers,
            extra_params=extra_params,
        )
    )


# --------------------------------------------------------------------------- #
# specialised helpers (chat / interactive)                                    #
# --------------------------------------------------------------------------- #
async def _enter_chat_mode(
    tool_manager,
    *,
    provider: str,
    model: str,
) -> bool:
    """Kick off the full-screen chat UI (banner printed by the handler)."""
    from mcp_cli.chat.chat_handler import handle_chat_mode

    return await handle_chat_mode(
        tool_manager,
        provider=provider,
        model=model,
    )


async def _enter_interactive_mode(
    tool_manager,
    *,
    provider: str,
    model: str,
) -> bool:
    """
    Start the interactive-shell UI.

    We pass in the *StreamManager* plus the ToolManager for parity with
    production code.
    """
    from mcp_cli.commands.interactive import interactive_mode

    return await interactive_mode(
        stream_manager=tool_manager.stream_manager,
        tool_manager=tool_manager,
        provider=provider,
        model=model,
    )


# --------------------------------------------------------------------------- #
# CLI entry-point (used by the `mcp-cli` wrapper script)                      #
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Master control programme 🙂")


@app.command("run")
def cli_entry(
    mode: str = typer.Argument("chat", help="chat | interactive"),
    config_file: str = typer.Option(
        "server_config.json", "--config", "-c", help="Server config file"
    ),
    server: List[str] = typer.Option(
        ["sqlite"], "--server", "-s", help="Server(s) to connect"
    ),
    provider: str = typer.Option("openai", help="LLM provider name"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model name"),
) -> None:
    """
    Thin wrapper so `uv run mcp-cli chat` (or `interactive`) is minimal.
    """

    async def _inner() -> None:
        if mode not in {"chat", "interactive"}:
            raise typer.BadParameter("mode must be 'chat' or 'interactive'")

        tm = await _init_tool_manager(config_file, server)

        try:
            if mode == "chat":
                ok = await _enter_chat_mode(tm, provider=provider, model=model)
            else:
                ok = await _enter_interactive_mode(tm, provider=provider, model=model)

            if not ok:
                raise RuntimeError("Command returned non-zero status")

        finally:
            await _safe_close(tm)

    try:
        asyncio.run(_inner())
    except Exception as exc:  # noqa: BLE001 – show nicely then exit
        output.print(Panel(str(exc), title="Fatal Error", style="bold red"))
        sys.exit(1)
