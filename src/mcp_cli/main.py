# src/mcp_cli/main.py
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

# --------------------------------------------------------------------------- #
# Local imports (kept as-is except for new log-level handling)
# --------------------------------------------------------------------------- #
from mcp_cli.cli.commands import register_all_commands
from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.run_command import run_command_sync
from mcp_cli.ui.ui_helpers import restore_terminal
from mcp_cli.cli_options import process_options
from mcp_cli.provider_config import ProviderConfig

# --------------------------------------------------------------------------- #
# Logging set-up
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# --------------------------------------------------------------------------- #
# Typer application + global “quiet” flag
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=False)
def main_callback(
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False,
        "-q", "--quiet",                # put the short flag first ✅
        help="Suppress server log output (sets CHUK_LOG_LEVEL=WARNING)",
        is_flag=True,
        show_default=False,
    ),
) -> None:
    """
    Executed before any sub-command.

    • Raises the root logger to WARNING when --quiet is given  
    • Exports CHUK_LOG_LEVEL so child servers are silent too
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        os.environ["CHUK_LOG_LEVEL"] = "WARNING"  # children read this
    # Typer still proceeds to sub-command normally.


# --------------------------------------------------------------------------- #
# Interactive Mode (unchanged except minor formatting)
# --------------------------------------------------------------------------- #
@app.command("interactive")
def _interactive_command(
    config_file: str = "server_config.json",
    server: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="API base URL for the provider"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for the provider"
    ),
    disable_filesystem: bool = False,
):
    """Start interactive command mode."""
    provider_cfg = ProviderConfig()
    actual_provider = provider or provider_cfg.get_active_provider()
    actual_model = model or provider_cfg.get_active_model()

    servers, _, server_names = process_options(
        server,
        disable_filesystem,
        actual_provider,
        actual_model,
        config_file,
    )

    from mcp_cli.interactive.shell import interactive_mode

    run_command_sync(
        interactive_mode,
        config_file,
        servers,
        extra_params={
            "provider": actual_provider,
            "model": actual_model,
            "server_names": server_names,
            "api_base": api_base,
            "api_key": api_key,
        },
    )


# --------------------------------------------------------------------------- #
# Register commands & groups (unchanged)
# --------------------------------------------------------------------------- #
register_all_commands()

CommandRegistry.create_subcommand_group(app, "tools",     ["list", "call"], run_command_sync)
CommandRegistry.create_subcommand_group(app, "resources", ["list"],         run_command_sync)
CommandRegistry.create_subcommand_group(app, "prompts",   ["list"],         run_command_sync)
CommandRegistry.create_subcommand_group(app, "servers",   ["list"],         run_command_sync)

for name in ("ping", "chat", "cmd", "provider"):
    cmd = CommandRegistry.get_command(name)
    if cmd:
        cmd.register(app, run_command_sync)

chat_cmd = CommandRegistry.get_command("chat")
if chat_cmd:
    chat_cmd.register_as_default(app, run_command_sync)


# --------------------------------------------------------------------------- #
# Clean shutdown helpers (unchanged)
# --------------------------------------------------------------------------- #
def _signal_handler(sig, _frame):
    logging.debug("Received signal %s, restoring terminal", sig)
    restore_terminal()
    sys.exit(0)


def _setup_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _setup_signal_handlers()
    try:
        app()  # Typer dispatch
    finally:
        restore_terminal()
        gc.collect()
