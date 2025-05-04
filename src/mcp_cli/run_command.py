# mcp_cli/run_command.py
"""
Main entry-point helpers for all CLI sub-commands.

These helpers encapsulate:

* construction / cleanup of the shared **ToolManager**
* hand-off to the individual command modules
* a thin synchronous wrapper so `uv run mcp-cli …` works
"""
from __future__ import annotations

import asyncio
import sys
from types import TracebackType
from typing import Any, Callable, Coroutine, Dict, List, Type

import typer
from rich.console import Console
from rich.panel import Panel

from mcp_cli.tools.manager import ToolManager, set_tool_manager

# --------------------------------------------------------------------------- #
# internal helpers / globals
# --------------------------------------------------------------------------- #
_ALL_TM: List[ToolManager] = []          # used by unit-tests


async def _init_tool_manager(
    config_file: str,
    servers: List[str],
    server_names: Dict[int, str] = None,
) -> ToolManager:
    """Create and initialise a *ToolManager* (stdio namespace)."""
    tm = ToolManager(config_file, servers, server_names)
    ok = await tm.initialize(namespace="stdio")
    if not ok:
        raise RuntimeError("Failed to initialise ToolManager")
    set_tool_manager(tm)          # make globally available
    _ALL_TM.append(tm)            # let tests assert close() calls
    return tm


async def _safe_close(tm: ToolManager) -> None:
    """Close and swallow *any* exceptions – never re-raise during shutdown."""
    try:
        await tm.close()
    except Exception:             # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# command dispatch
# --------------------------------------------------------------------------- #
async def run_command(
    async_command: Callable[..., Coroutine[Any, Any, Any]],
    *,
    config_file: str,
    servers: List[str],
    extra_params: Dict[str, Any] | None,
) -> Any:
    """
    Initialise the ToolManager, then call *async_command*(tool_manager, **extra).

    Always closes the ToolManager, even if the command raises.
    """
    tm: ToolManager | None = None
    try:
        # Extract server_names from extra_params if available
        server_names = extra_params.get("server_names") if extra_params else None
        
        # Initialize the tool manager with server_names
        tm = await _init_tool_manager(config_file, servers, server_names)
        
        # Check if the command being called is an interactive app
        command_name = getattr(async_command, "__name__", "")
        module_name = getattr(async_command, "__module__", "")
        
        # If it's the interactive app, use _enter_interactive_mode instead
        if command_name == "app" and "interactive" in module_name:
            # Extract provider and model from extra_params
            provider = extra_params.get("provider", "openai") if extra_params else "openai"
            model = extra_params.get("model", "gpt-4o-mini") if extra_params else "gpt-4o-mini"
            
            # Use the special helper function
            result = await _enter_interactive_mode(
                tm, 
                provider=provider, 
                model=model
            )
        else:
            # Normal case - pass tool_manager as keyword arg
            result = await async_command(tool_manager=tm, **(extra_params or {}))
            
        return result
    finally:
        if tm:
            await _safe_close(tm)


def run_command_sync(
    async_command: Callable[..., Coroutine[Any, Any, Any]],
    config_file: str,
    servers: List[str],
    extra_params: Dict[str, Any] | None,
) -> Any:
    """
    Synchronous wrapper for convenience scripts / unit-tests.

    Creates its own event-loop if necessary.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        run_command(async_command,
                    config_file=config_file,
                    servers=servers,
                    extra_params=extra_params),
    )


# --------------------------------------------------------------------------- #
# specialised helpers (chat / interactive)
# --------------------------------------------------------------------------- #
async def _enter_chat_mode(
    tool_manager: ToolManager,
    *,
    provider: str,
    model: str,
) -> bool:
    """
    Start the chat UI.

    NOTE: We removed the generic banner here – the *chat_handler*
    prints its own, richer welcome panel.
    """
    from mcp_cli.chat.chat_handler import handle_chat_mode

    return await handle_chat_mode(
        tool_manager,
        provider=provider,
        model=model,
    )


async def _enter_interactive_mode(
    tool_manager: ToolManager,
    *,
    provider: str,
    model: str,
) -> bool:
    """
    Start the interactive mode UI.
    
    This is a wrapper that extracts the stream_manager from tool_manager
    and calls the actual interactive_mode function.
    """
    from mcp_cli.commands.interactive import interactive_mode

    return await interactive_mode(
        stream_manager=tool_manager.stream_manager,
        tool_manager=tool_manager,
        provider=provider,
        model=model,
    )


# --------------------------------------------------------------------------- #
# CLI entry-point used by `mcp-cli` wrapper
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
    """Thin wrapper so `uv run mcp-cli …` is as short as possible."""
    console = Console()

    async def _inner() -> None:
        if mode not in {"chat", "interactive"}:
            raise typer.BadParameter("mode must be 'chat' or 'interactive'")

        tm = await _init_tool_manager(config_file, server)

        try:
            if mode == "chat":
                ok = await _enter_chat_mode(
                    tm, provider=provider, model=model
                )
            else:  # interactive
                ok = await _enter_interactive_mode(
                    tm, provider=provider, model=model
                )

            if not ok:
                raise RuntimeError("Command returned non-zero status")

        finally:
            await _safe_close(tm)

    try:
        asyncio.run(_inner())
    except Exception as exc:      # noqa: BLE001
        console.print(
            Panel(str(exc), title="Fatal Error", style="bold red")
        )
        sys.exit(1)