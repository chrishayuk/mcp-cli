# mcp_cli/cli/commands/chat.py
"""Chat command implementation (no more ‘KWARGS’!)."""
from __future__ import annotations

import logging
import signal
from typing import Any, Callable, Optional

import typer

from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.provider_config import ProviderConfig        # ← NEW
from mcp_cli.tools.manager import ToolManager

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _default_model(provider: str, explicit: Optional[str]) -> str:
    """
    Return a guaranteed model string.

    Priority:
    1. Explicit --model from CLI
    2. ProviderConfig.default_model for that provider
    3. Library fallback (“gpt-4o-mini”)
    """
    if explicit:  # user passed --model
        return explicit

    cfg = ProviderConfig().get_provider_config(provider.lower())
    return cfg.get("default_model", "gpt-4o-mini")


def _set_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise typer.BadParameter(f"Invalid logging level: {level}")
    logging.getLogger().setLevel(numeric)


# ──────────────────────────────────────────────────────────────────────────────
# command
# ──────────────────────────────────────────────────────────────────────────────
class ChatCommand(BaseCommand):
    """Top-level `/chat` command (and default action)."""

    def __init__(self) -> None:
        super().__init__("chat", "Start interactive chat mode.")

    # ----------------------------------------------------------------- execute
    async def execute(self, tool_manager: ToolManager, **params) -> Any:
        """Spin up the chat UI with sane defaults."""
        from mcp_cli.chat.chat_handler import handle_chat_mode

        provider: str = params.get("provider", "openai")
        model: str = _default_model(provider, params.get("model"))

        log.debug("Starting chat (provider=%s model=%s)", provider, model)
        return await handle_chat_mode(tool_manager, provider=provider, model=model)

    # ----------------------------------------------------------- register (sub)
    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register `/chat` as an explicit sub-command (no KWARGS)."""

        @app.command(self.name, help=self.help)
        def _chat(                                   # noqa: D401
            config_file: str = "server_config.json",
            server: Optional[str] = None,
            provider: str = "openai",
            model: Optional[str] = None,
            disable_filesystem: bool = False,
            logging_level: str = typer.Option(
                "WARNING",
                help="Set logging level: DEBUG/INFO/WARNING/ERROR/CRITICAL",
            ),
        ) -> None:
            _set_logging(logging_level)
            model_fallback = _default_model(provider, model)

            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model_fallback, config_file
            )

            extra = {
                "provider": provider,
                "model": model_fallback,
                "server_names": server_names,
            }
            run_command_func(self.wrapped_execute, config_file, servers, extra_params=extra)

    # ------------------------------------------------------- register_default
    def register_as_default(self, app: typer.Typer, run_command_func: Callable) -> None:
        """
        Make `/chat` the action when the user runs plain `mcp-cli …`
        with no sub-command.
        """

        @app.callback(invoke_without_command=True)
        def _default(                                 # noqa: D401
            ctx: typer.Context,
            config_file: str = "server_config.json",
            server: Optional[str] = None,
            provider: str = "openai",
            model: Optional[str] = None,
            disable_filesystem: bool = False,
            logging_level: str = typer.Option(
                "WARNING",
                help="Set logging level: DEBUG/INFO/WARNING/ERROR/CRITICAL",
            ),
        ) -> None:
            _set_logging(logging_level)
            model_fallback = _default_model(provider, model)

            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model_fallback, config_file
            )
            ctx.obj = {
                "config_file": config_file,
                "servers": servers,
                "server_names": server_names,
            }

            # If the user typed *only* `mcp-cli` (no sub-command) …
            if ctx.invoked_subcommand is None:
                # Graceful Ctrl-C / SIGTERM handling
                def _sig_handler(sig, _frame):          # noqa: D401
                    log.debug("Received signal %s – exiting chat", sig)
                    from mcp_cli.ui.ui_helpers import restore_terminal

                    restore_terminal()
                    raise typer.Exit()

                signal.signal(signal.SIGINT, _sig_handler)
                signal.signal(signal.SIGTERM, _sig_handler)

                extra = {
                    "provider": provider,
                    "model": model_fallback,
                    "server_names": server_names,
                }
                run_command_func(
                    self.wrapped_execute,
                    config_file,
                    servers,
                    extra_params=extra,
                )

                from mcp_cli.ui.ui_helpers import restore_terminal

                restore_terminal()
                raise typer.Exit()
