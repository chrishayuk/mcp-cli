# mcp_cli/commands/register_commands.py
"""Register Typer sub‑commands for the MCP CLI in a runner‑agnostic way.

Sub‑commands are bound to the provided `run_command_func`, which handles
lifecycle of StreamManager and event‑loop execution.
"""
from __future__ import annotations

import typer
from typing import Any, Callable, Dict, List, Optional, Tuple

# implementations of each sub‑command
from mcp_cli.commands import ping, chat, prompts, tools, resources, interactive, cmd
from mcp_cli.run_command import run_command_sync
from mcp_cli.cli_options import process_options

__all__ = ["register_commands", "chat_command"]


def register_commands(
    app: typer.Typer,
    process_opts: Callable[..., Tuple[List[str], List[str], Dict[int, str]]],
    run_command_func: Callable[..., Any],
) -> None:
    """Attach sub‑commands to the Typer app using the given runner."""

    @app.command("ping")
    def ping_command(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            ping.ping_run,
            config_file,
            servers,
            user_specified,
            {"server_names": server_names},
        )
        return 0

    @app.command("chat")
    def _chat(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            chat.chat_run,
            config_file,
            servers,
            user_specified,
            {"server_names": server_names},
        )
        return 0

    @app.command("interactive")
    def interactive_command(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            interactive.interactive_mode,
            config_file,
            servers,
            user_specified,
        )
        return 0

    # Prompts
    prompts_app = typer.Typer(help="Prompts commands")

    @prompts_app.command("list")
    def prompts_list(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            prompts.prompts_list,
            config_file,
            servers,
            user_specified,
            {"server_names": server_names},
        )
        return 0

    app.add_typer(prompts_app, name="prompts")

    # Tools
    tools_app = typer.Typer(help="Tools commands")

    @tools_app.command("list")
    def tools_list(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            tools.tools_list,
            config_file,
            servers,
            user_specified,
            {"server_names": server_names},
        )
        return 0

    @tools_app.command("call")
    def tools_call(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            tools.tools_call,
            config_file,
            servers,
            user_specified,
            {"server_names": server_names},
        )
        return 0

    app.add_typer(tools_app, name="tools")

    # Resources
    resources_app = typer.Typer(help="Resources commands")

    @resources_app.command("list")
    def resources_list(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        run_command_func(
            resources.resources_list,
            config_file,
            servers,
            user_specified,
            {"server_names": server_names},
        )
        return 0

    app.add_typer(resources_app, name="resources")

    # Cmd
    @app.command("cmd")
    def cmd_command(
        config_file: str = "server_config.json",
        server: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        disable_filesystem: bool = False,
        input: Optional[str] = None,
        prompt: Optional[str] = None,
        output: Optional[str] = None,
        raw: bool = False,
        tool: Optional[str] = None,
        tool_args: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> int:
        servers, user_specified, server_names = process_opts(
            server, disable_filesystem, provider, model, config_file
        )
        extra_params: Dict[str, Any] = {
            "input": input,
            "prompt": prompt,
            "output": output,
            "raw": raw,
            "tool": tool,
            "tool_args": tool_args,
            "system_prompt": system_prompt,
            "server_names": server_names,
        }
        run_command_func(
            cmd.cmd_run,
            config_file,
            servers,
            user_specified,
            extra_params,
        )
        return 0


def chat_command(
    config_file: str = "server_config.json",
    server: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    disable_filesystem: bool = False,
) -> int:
    """Start an interactive chat session (for default entry‑point)."""
    servers, user_specified, server_names = process_options(
        server, disable_filesystem, provider, model, config_file
    )
    run_command_sync(
        chat.chat_run,
        config_file,
        servers,
        user_specified,
        {"server_names": server_names},
    )
    return 0
