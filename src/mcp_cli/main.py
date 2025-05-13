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
from typing import Optional, Callable

import typer

# ──────────────────────────────────────────────────────────────────────────────
# local imports
# ──────────────────────────────────────────────────────────────────────────────
from mcp_cli.cli.commands import register_all_commands
from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.run_command import run_command_sync
from mcp_cli.ui.ui_helpers import restore_terminal
from mcp_cli.cli_options import process_options
from mcp_cli.provider_config import ProviderConfig

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
# helper: generic wrapper builder
# ──────────────────────────────────────────────────────────────────────────────
def _register_command_with_help(
    app: typer.Typer,
    command_name: str,
    cmd,
    run_cmd: Callable,
) -> None:
    """Register *cmd* on *app* with a thin Typer wrapper keeping its help."""
    if not cmd:
        return

    help_text = getattr(cmd, "help_text", None) or getattr(cmd, "help", "") or (
        f"Execute the {command_name} command"
    )

    def make_wrapper(cmd_obj):
        _orig_exec = cmd_obj.wrapped_execute

        def wrapper(  # noqa: D401
            config_file: str = typer.Option(
                "server_config.json", help="Configuration file path"
            ),
            server: Optional[str] = typer.Option(
                None, help="Server to connect to"
            ),
            provider: str = typer.Option("openai", help="LLM provider name"),
            model: Optional[str] = typer.Option(None, help="Model name"),
            disable_filesystem: bool = typer.Option(
                False, help="Disable filesystem access"
            ),
            **kwargs,
        ):
            """Auto-generated CLI wrapper (help preserved)."""
            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model, config_file
            )
            extra_params = {
                "provider": provider,
                "model": model,
                "server_names": server_names,
                **kwargs,
            }
            run_cmd(_orig_exec, config_file, servers, extra_params=extra_params)

        wrapper.__doc__ = help_text
        wrapper.__name__ = f"_{command_name}_command"
        return wrapper

    wrapper = make_wrapper(cmd)
    app.command(command_name, help=help_text)(wrapper)


# ──────────────────────────────────────────────────────────────────────────────
# Typer root app + global “quiet” flag
# ──────────────────────────────────────────────────────────────────────────────
app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=False)
def main_callback(  # noqa: D401
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress server log output (sets CHUK_LOG_LEVEL=WARNING)",
        is_flag=True,
        show_default=False,
    ),
) -> None:
    """Common pre-command setup (handles --quiet)."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        os.environ["CHUK_LOG_LEVEL"] = "WARNING"


# ──────────────────────────────────────────────────────────────────────────────
# interactive (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@app.command("interactive", help="Start interactive command mode.")
def _interactive_command(  # noqa: D401
    config_file: str = typer.Option(
        "server_config.json", help="Configuration file path"
    ),
    server: Optional[str] = typer.Option(None, help="Server to connect to"),
    provider: Optional[str] = typer.Option(
        None, help="LLM provider name (defaults to active provider)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name (defaults to active model)"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="API base URL for the provider"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for the provider"
    ),
    disable_filesystem: bool = typer.Option(
        False, help="Disable filesystem access"
    ),
) -> None:
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


# ──────────────────────────────────────────────────────────────────────────────
# sub-command groups (tools/resources/…)
# ──────────────────────────────────────────────────────────────────────────────
def _create_enhanced_subcommand_group(
    app: typer.Typer,
    group_name: str,
    sub_commands: list[str],
    run_cmd: Callable,
) -> None:
    """Add a Typer sub-app exposing e.g. “tools list”."""
    group_title = f"{group_name.capitalize()} commands"
    subapp = typer.Typer(help=f"Commands for working with {group_name}")

    for sub_name in sub_commands:
        full_name = f"{group_name} {sub_name}"
        cmd_obj = CommandRegistry.get_command(full_name)
        if not cmd_obj:
            logging.warning("Command '%s' not found in registry", full_name)
            continue

        help_text = getattr(cmd_obj, "help_text", None) or getattr(
            cmd_obj, "help", ""
        )

        def make_wrapper(command):
            _orig_exec = command.wrapped_execute

            def wrapper(  # noqa: D401
                config_file: str = typer.Option(
                    "server_config.json", help="Configuration file path"
                ),
                server: Optional[str] = typer.Option(
                    None, help="Server to connect to"
                ),
                provider: str = typer.Option("openai", help="LLM provider name"),
                model: Optional[str] = typer.Option(None, help="Model name"),
                disable_filesystem: bool = typer.Option(
                    False, help="Disable filesystem access"
                ),
                **kwargs,
            ):
                servers, _, server_names = process_options(
                    server, disable_filesystem, provider, model, config_file
                )
                extra = {
                    "provider": provider,
                    "model": model,
                    "server_names": server_names,
                    **kwargs,
                }
                run_cmd(_orig_exec, config_file, servers, extra_params=extra)

            wrapper.__name__ = f"_{group_name}_{sub_name}_command"
            wrapper.__doc__ = help_text
            return wrapper

        wrapper = make_wrapper(cmd_obj)
        subapp.command(sub_name, help=help_text)(wrapper)

    app.add_typer(subapp, name=group_name, help=group_title)


# ──────────────────────────────────────────────────────────────────────────────
# command registration
# ──────────────────────────────────────────────────────────────────────────────
register_all_commands()

_create_enhanced_subcommand_group(
    app, "tools", ["list", "call"], run_command_sync
)
_create_enhanced_subcommand_group(
    app, "resources", ["list"], run_command_sync
)
_create_enhanced_subcommand_group(
    app, "prompts", ["list"], run_command_sync
)
_create_enhanced_subcommand_group(
    app, "servers", ["list"], run_command_sync
)

# commands that *don’t* supply their own Typer wrapper
for name in ("ping", "chat", "provider"):
    cmd = CommandRegistry.get_command(name)
    if cmd:
        _register_command_with_help(app, name, cmd, run_command_sync)

# let CmdCommand register its own rich wrapper (keeps --prompt etc.)
cmd_cmd = CommandRegistry.get_command("cmd")
if cmd_cmd:
    cmd_cmd.register(app, run_command_sync)

# make /chat the default when no sub-command is given
chat_cmd = CommandRegistry.get_command("chat")
if chat_cmd:
    chat_cmd.register_as_default(app, run_command_sync)


# ──────────────────────────────────────────────────────────────────────────────
# graceful shutdown
# ──────────────────────────────────────────────────────────────────────────────
def _signal_handler(sig, _frame):
    logging.debug("Received signal %s, restoring terminal", sig)
    restore_terminal()
    sys.exit(0)


def _setup_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _setup_signal_handlers()
    try:
        app()  # Typer dispatch
    finally:
        restore_terminal()
        gc.collect()
