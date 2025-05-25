# src/mcp_cli/main.py - Fixed version with chat as default
"""Entry-point for the MCP CLI."""
from __future__ import annotations

import asyncio
import atexit
import gc
import logging
import os
import signal
import sys
from typing import Optional

import typer

# ──────────────────────────────────────────────────────────────────────────────
# local imports
# ──────────────────────────────────────────────────────────────────────────────
from mcp_cli.cli.commands import register_all_commands
from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.run_command import run_command_sync
from mcp_cli.ui.ui_helpers import restore_terminal
from mcp_cli.cli_options import process_options

# ──────────────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
logging.getLogger("chuk_tool_processor.span.inprocess_execution").setLevel(
    logging.WARNING
)

# ──────────────────────────────────────────────────────────────────────────────
# Typer root app
# ──────────────────────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False)


# ──────────────────────────────────────────────────────────────────────────────
# Default callback that handles no-subcommand case
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),  # ← CHANGED to Optional
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress server log output"),
) -> None:
    """MCP CLI - If no subcommand is given, start chat mode."""
    
    # If a subcommand was invoked, let it handle things
    if ctx.invoked_subcommand is not None:
        if quiet:
            logging.getLogger().setLevel(logging.WARNING)
            os.environ["CHUK_LOG_LEVEL"] = "WARNING"
        return
    
    # No subcommand - start chat mode (default behavior)
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        os.environ["CHUK_LOG_LEVEL"] = "WARNING"
    
    # Use ModelManager to get active provider/model if not specified
    from mcp_cli.model_manager import ModelManager
    model_manager = ModelManager()
    
    # Smart provider/model resolution:
    # 1. If both specified: use both
    # 2. If only provider specified: use provider + its default model
    # 3. If neither specified: use active provider + active model
    if provider and model:
        # Both specified explicitly
        effective_provider = provider
        effective_model = model
    elif provider and not model:
        # Provider specified, get its default model
        effective_provider = provider
        effective_model = model_manager.get_default_model(provider)
    elif not provider and model:
        # Model specified, use current provider
        effective_provider = model_manager.get_active_provider()
        effective_model = model
    else:
        # Neither specified, use active configuration
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()
    
    servers, _, server_names = process_options(
        server, disable_filesystem, effective_provider, effective_model, config_file
    )

    from mcp_cli.chat.chat_handler import handle_chat_mode
    from mcp_cli.tools.manager import ToolManager

    # Start chat mode directly
    async def _start_chat():
        tm = None
        try:
            from mcp_cli.run_command import _init_tool_manager
            tm = await _init_tool_manager(config_file, servers, server_names)
            
            success = await handle_chat_mode(
                tool_manager=tm,
                provider=effective_provider,  # Use effective values
                model=effective_model,        # Use effective values
                api_base=api_base,
                api_key=api_key
            )
            
        finally:
            if tm:
                from mcp_cli.run_command import _safe_close
                await _safe_close(tm)
    
    try:
        asyncio.run(_start_chat())
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        print(f"[red]Error:[/red] {e}")
    finally:
        restore_terminal()
        raise typer.Exit()


# ──────────────────────────────────────────────────────────────────────────────
# Built-in commands
# ──────────────────────────────────────────────────────────────────────────────
@app.command("interactive", help="Start interactive command mode.")
def _interactive_command(
    config_file: str = typer.Option("server_config.json", help="Configuration file path"),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(None, help="LLM provider name"),  # ← CHANGED to Optional
    model: Optional[str] = typer.Option(None, help="Model name"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress server log output"),
) -> None:
    """Start interactive command mode."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        os.environ["CHUK_LOG_LEVEL"] = "WARNING"
    
    # Use ModelManager to get active provider/model if not specified
    from mcp_cli.model_manager import ModelManager
    model_manager = ModelManager()
    
    # Smart provider/model resolution:
    # 1. If both specified: use both
    # 2. If only provider specified: use provider + its default model
    # 3. If neither specified: use active provider + active model
    if provider and model:
        # Both specified explicitly
        effective_provider = provider
        effective_model = model
    elif provider and not model:
        # Provider specified, get its default model
        effective_provider = provider
        effective_model = model_manager.get_default_model(provider)
    elif not provider and model:
        # Model specified, use current provider
        effective_provider = model_manager.get_active_provider()
        effective_model = model
    else:
        # Neither specified, use active configuration
        effective_provider = model_manager.get_active_provider()
        effective_model = model_manager.get_active_model()
    
    servers, _, server_names = process_options(
        server, disable_filesystem, effective_provider, effective_model, config_file
    )

    from mcp_cli.interactive.shell import interactive_mode

    run_command_sync(
        interactive_mode,
        config_file,
        servers,
        extra_params={
            "provider": effective_provider,  # Use effective values
            "model": effective_model,        # Use effective values
            "server_names": server_names,
            "api_base": api_base,
            "api_key": api_key,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Command registration - ONLY individual commands (no subgroups)
# ──────────────────────────────────────────────────────────────────────────────
def _register_simple_command(command_name: str) -> None:
    """Register a simple command from the registry."""
    cmd = CommandRegistry.get_command(command_name)
    if cmd:
        cmd.register(app, run_command_sync)


# Register all commands in the registry first
register_all_commands()

# Register individual commands that we know work
for command_name in ["chat", "provider", "cmd", "ping"]:
    _register_simple_command(command_name)

print("✓ MCP CLI ready - chat is default, or use: chat, interactive, provider, cmd, ping")

# ──────────────────────────────────────────────────────────────────────────────
# Signal handling
# ──────────────────────────────────────────────────────────────────────────────
def _setup_signal_handlers() -> None:
    """Setup signal handlers for clean shutdown."""
    def handler(sig, _frame):
        logging.debug(f"Received signal {sig}, shutting down")
        restore_terminal()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, handler)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _setup_signal_handlers()
    atexit.register(restore_terminal)
    
    try:
        app()
    finally:
        restore_terminal()
        gc.collect()