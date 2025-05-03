# mcp_cli/commands/register_commands.py
"""Bind all high-level CLI sub-commands to a Typer app."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import typer

# concrete command modules
from mcp_cli.commands import (
    chat,
    cmd,
    interactive,
    ping,
    prompts,
    resources,
    tools,
)

from mcp_cli.cli_options import process_options
from mcp_cli.run_command import run_command_sync

__all__ = ["register_commands", "chat_command"]


# --------------------------------------------------------------------------- #
# small wrapper so tests can monkey-patch `process_options`
# --------------------------------------------------------------------------- #
def _process(
    process_opts: Callable[
        ...,
        Tuple[
            List[str],      # servers
            List[str],      # user_specified  (ignored here)
            Dict[int, str], # server_names
        ],
    ],
    server: Optional[str],
    disable_filesystem: bool,
    provider: str,
    model: Optional[str],
    config_file: str,
) -> Tuple[List[str], Dict[int, str]]:
    servers, _user_specified, server_names = process_opts(
        server, disable_filesystem, provider, model, config_file
    )
    return servers, server_names


# --------------------------------------------------------------------------- #
# main registration function
# --------------------------------------------------------------------------- #
def register_commands(
    app: typer.Typer,
    process_opts: Callable[
        ...,
        Tuple[List[str], List[str], Dict[int, str]],
    ],
    run_command_func: Callable[..., Any],   # injected in tests
) -> None:
    """Attach every sub-command to *app* using the supplied runner."""

    # ───────────────────────── ping ────────────────────────────
    @app.command("ping")
    def _ping(                                   # noqa: D401
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            ping.ping_run,                      # Typer Command
            config_file,
            servers,
            extra_params={"server_names": server_names},
        )

    # ───────────────────────── chat ────────────────────────────
    @app.command("chat")
    def _chat(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            chat.chat_run,
            config_file,
            servers,
            extra_params={"server_names": server_names},
        )

    # ─────────────────────── interactive ───────────────────────
    @app.command("interactive")
    def _interactive(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, _ = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(interactive.interactive_mode, config_file, servers)

    # ───────────────────────── prompts ─────────────────────────
    prompts_app = typer.Typer(help="Prompts commands")

    @prompts_app.command("list")
    def _prompts_list(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            prompts.prompts_list,
            config_file,
            servers,
            extra_params={"server_names": server_names},
        )

    app.add_typer(prompts_app, name="prompts")

    # ───────────────────────── tools ───────────────────────────
    tools_app = typer.Typer(help="Tools commands")

    @tools_app.command("list")
    def _tools_list(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            tools.tools_list,
            config_file,
            servers,
            extra_params={"server_names": server_names},
        )

    @tools_app.command("call")
    def _tools_call(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            tools.tools_call,
            config_file,
            servers,
            extra_params={"server_names": server_names},
        )

    app.add_typer(tools_app, name="tools")

    # ──────────────────────── resources ────────────────────────
    resources_app = typer.Typer(help="Resources commands")

    @resources_app.command("list")
    def _resources_list(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            resources.resources_list,
            config_file,
            servers,
            extra_params={"server_names": server_names},
        )

    app.add_typer(resources_app, name="resources")

    # ───────────────────────── cmd/run ─────────────────────────
    @app.command("cmd")
    def _cmd(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
        input: Optional[str] = None,      # noqa: A002
        prompt: Optional[str] = None,
        output: Optional[str] = None,
        raw: bool = False,
        tool: Optional[str] = None,
        tool_args: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        servers, server_names = _process(
            process_opts, server, disable_filesystem, provider, model, config_file
        )
        runner_kwargs: Dict[str, Any] = {
            "input": input,
            "prompt": prompt,
            "output": output,
            "raw": raw,
            "tool": tool,
            "tool_args": tool_args,
            "system_prompt": system_prompt,
            "server_names": server_names,
        }
        run_command_func(cmd.cmd_run, config_file, servers, extra_params=runner_kwargs)

    # ------------------------------------------------------------------ #
    # helper used by the package’s `python -m mcp_cli` entry-point
    # ------------------------------------------------------------------ #
def chat_command(
    config_file: str = "server_config.json",
    server: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    disable_filesystem: bool = False,
) -> None:
    """Start an interactive chat session (shortcut for `mcp-cli chat`)."""
    servers, _user_specified, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )
    run_command_sync(
        chat.chat_run,
        config_file,
        servers,
        extra_params={"server_names": server_names},
    )

