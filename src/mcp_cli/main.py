# mcp_cli/main.py
"""Entry-point for the MCP CLI.

Only minor changes were required after refactoring *run_command* to be
async-first:

* We now import **run_command_sync** (blocking helper) and pass it to
  `register_commands()`.  Down‑stream command handlers keep operating in a
  synchronous context without seeing any difference.
* Removed stale imports that were no longer referenced in this module.
"""
from __future__ import annotations

import asyncio
import atexit
import gc
import logging
import os
import signal
import sys

import typer

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from mcp_cli.cli_options import process_options
from mcp_cli.commands.register_commands import register_commands, chat_command
from mcp_cli.run_command import run_command_sync  # blocking helper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def restore_terminal() -> None:
    """Best‑effort attempt to reset the TTY and close the asyncio loop."""
    os.system("stty sane")

    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            return

        # Cancel outstanding tasks
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    except Exception as exc:  # pragma: no cover – best‑effort cleanup only
        logging.debug("Asyncio cleanup error: %s", exc)
    finally:
        gc.collect()


atexit.register(restore_terminal)

# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------
app = typer.Typer()
register_commands(app, process_options, run_command_sync)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(sig, _frame):
    logging.debug("Received signal %s", sig)
    restore_terminal()
    sys.exit(0)


def _setup_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def common_options(
    ctx: typer.Context,
    config_file: str = "server_config.json",
    server: str | None = None,
    provider: str = "openai",
    model: str | None = None,
    disable_filesystem: bool = True,
    logging_level: str = typer.Option(
        "WARNING",
        help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
):
    """Global CLI options.

    If the user does not specify a sub-command, we drop straight into chat
    mode (interactive REPL).
    """
    numeric_level = getattr(logging, logging_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Invalid logging level: {logging_level}")
    logging.getLogger().setLevel(numeric_level)
    logging.debug("Logging level set to %s", logging_level.upper())

    servers, user_specified, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )

    ctx.obj = {
        "config_file": config_file,
        "servers": servers,
        "user_specified": user_specified,
        "server_names": server_names,
    }

    if ctx.invoked_subcommand is None:
        _setup_signal_handlers()
        chat_command(
            config_file=config_file,
            server=server,
            provider=provider,
            model=model,
            disable_filesystem=disable_filesystem,
        )
        restore_terminal()
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Script entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        _setup_signal_handlers()
        app()
    except KeyboardInterrupt:
        logging.debug("KeyboardInterrupt received")
    except Exception as exc:  # pragma: no cover – top‑level catch‑all
        logging.error("Unhandled exception: %s", exc)
    finally:
        restore_terminal()
