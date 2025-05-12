# mcp_cli/cli/registry.py
"""
Command registry for MCP-CLI – central place to register & discover commands.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import typer

from mcp_cli.cli.commands.base import BaseCommand, CommandFunc, FunctionCommand

logger = logging.getLogger(__name__)


class CommandRegistry:
    """Central registry holding every CLI command object."""
    _commands: Dict[str, BaseCommand] = {}

    # ── registration helpers ─────────────────────────────────────────
    @classmethod
    def register(cls, command: BaseCommand) -> None:
        cls._commands[command.name] = command

    @classmethod
    def register_function(
        cls,
        name: str,
        func: CommandFunc,
        help_text: str = "",
    ) -> None:
        cls.register(FunctionCommand(name, func, help_text))

    # ── retrieval helpers ────────────────────────────────────────────
    @classmethod
    def get_command(cls, name: str) -> Optional[BaseCommand]:
        return cls._commands.get(name)

    @classmethod
    def get_all_commands(cls) -> List[BaseCommand]:
        return list(cls._commands.values())

    # ── bulk registration into a Typer app ───────────────────────────
    @classmethod
    def register_with_typer(cls, app: typer.Typer, run_cmd: Callable) -> None:
        for cmd in cls._commands.values():
            cmd.register(app, run_cmd)

    # ── create grouped sub-commands (e.g. “tools list”) ──────────────
    @classmethod
    def create_subcommand_group(
        cls,
        app: typer.Typer,
        group_name: str,
        sub_commands: List[str],
        run_cmd: Callable,
    ) -> None:
        """
        Attach a Typer sub-app that exposes commands like “tools list”.
        """
        subapp = typer.Typer(help=f"{group_name.capitalize()} commands")

        for sub_name in sub_commands:
            full_name = f"{group_name} {sub_name}"
            cmd_obj = cls.get_command(full_name)

            if not cmd_obj:
                logger.warning("Command '%s' not found in registry", full_name)
                continue

            # ------------------------------------------------------------------
            # Wrapper factory – binds variables at definition time to avoid the
            # classic late-binding-in-a-loop bug.
            # ------------------------------------------------------------------
            # create wrapper with variables bound at *definition* time
            # mcp_cli/cli/registry.py  — only the inner wrapper changed
            def _make_wrapper(command: BaseCommand) -> Callable[..., None]:
                _orig_exec = command.wrapped_execute  # bind once

                def command_wrapper(
                    *,
                    config_file: str = "server_config.json",
                    server: str | None = None,
                    provider: str = "openai",
                    model: str | None = None,
                    disable_filesystem: bool = False,
                ) -> None:
                    """Dynamically generated Typer wrapper."""
                    from mcp_cli.cli_options import process_options

                    servers, _, server_names = process_options(
                        server, disable_filesystem, provider, model, config_file
                    )

                    extra_params = {
                        "provider": provider,
                        "model": model,
                        "server_names": server_names,
                    }

                    run_cmd(
                        _orig_exec,
                        config_file,
                        servers,
                        extra_params=extra_params,
                    )

                # carry original help text safely
                help_str = getattr(command, "help_text", None) or getattr(command, "help", "")
                command_wrapper.__doc__ = help_str or "Sub-command wrapper"
                return command_wrapper



            subapp.command(sub_name)(_make_wrapper(cmd_obj))

        app.add_typer(subapp, name=group_name)
