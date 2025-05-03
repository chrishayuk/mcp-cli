# mcp_cli/commands/interactive.py
"""
Interactive CLI loop for MCP.

It glues together the individual sub-commands (/ping, /tools, …) and
presents a small REPL so the user can work comfortably from a single
prompt.

Only the *current* command APIs are supported – all legacy shims were
removed.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any, Callable, Dict

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# prompt-toolkit (for slash-command completions)
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
from mcp_cli.commands import chat, ping, prompts, resources
from mcp_cli.chat.commands.tools import tools_command
from mcp_cli.chat.command_completer import ChatCommandCompleter
from mcp_cli.ui.ui_helpers import clear_screen

# --------------------------------------------------------------------------- #
# Typer application (only used if the file is “cli-invoked”)
# --------------------------------------------------------------------------- #
app = typer.Typer(help="Interactive mode")


# --------------------------------------------------------------------------- #
# Public entry-point
# --------------------------------------------------------------------------- #
async def interactive_mode(
    stream_manager: Any,
    tool_manager: Any | None = None,
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> bool:
    """
    Run the interactive prompt.

    Parameters
    ----------
    stream_manager
        Object exposing at least:

        * `get_all_tools()` – list of tool specs
        * `get_server_info()` – list with metadata per server
        * `tool_to_server_map` – mapping *tool → server*

    tool_manager
        Passed straight through to `chat.chat_run`.  If *None* we re-use
        *stream_manager*.

    provider / model
        Display-only metadata for the banner.
    """
    if tool_manager is None:
        tool_manager = stream_manager

    console = Console()

    # Build context object that other helpers expect
    context: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "tools": stream_manager.get_all_tools(),
        "server_info": stream_manager.get_server_info(),
        "tool_to_server_map": stream_manager.tool_to_server_map,
        "stream_manager": stream_manager,
        "tool_manager": tool_manager,
        "server_names": getattr(stream_manager, "server_names", {}),
    }

    _banner(console, context)

    # --------------------------------------------------------------------- #
    # Command handlers (functions or coroutines)
    # --------------------------------------------------------------------- #
    async def cmd_ping(_arg: str = ""):
        await ping.ping_run(stream_manager=stream_manager)

    async def cmd_prompts(_arg: str = ""):
        await prompts.prompts_list(
            stream_manager=getattr(tool_manager, "stream_manager", stream_manager)
        )

    async def cmd_resources(_arg: str = ""):
        await resources.resources_list(
            stream_manager=getattr(tool_manager, "stream_manager", stream_manager)
        )

    async def cmd_tools(_arg: str = ""):
        await tools_command([], context)

    async def cmd_tools_all(_arg: str = ""):
        await tools_command(["--all"], context)

    async def cmd_tools_raw(_arg: str = ""):
        await tools_command(["--raw"], context)

    async def cmd_chat(_arg: str = ""):
        await chat.chat_run(tool_manager=tool_manager)

    def cmd_servers(_arg: str = ""):
        _servers(console, context)

    def cmd_cls(_arg: str = ""):
        clear_screen()

    def cmd_clear(_arg: str = ""):
        clear_screen()
        _banner(console, context)

    def cmd_exit(_arg: str = ""):
        return "exit"

    def cmd_help(_arg: str = ""):
        _help(console)

    # Map prompt → handler  (signature handler(argstr:str)->Any)
    commands: Dict[str, Callable[[str], Any]] = {
        "/ping": cmd_ping,
        "/prompts": cmd_prompts,
        "/resources": cmd_resources,
        "/tools": cmd_tools,
        "/tools-all": cmd_tools_all,
        "/tools-raw": cmd_tools_raw,
        "/chat": cmd_chat,
        "/servers": cmd_servers,
        "/s": cmd_servers,
        "/cls": cmd_cls,
        "/clear": cmd_clear,
        "/help": cmd_help,
        "/exit": cmd_exit,
        "/quit": cmd_exit,
    }

    # --------------------------------------------------------------------- #
    # Prompt-toolkit session (with slash completions)
    # --------------------------------------------------------------------- #
    style = Style.from_dict(
        {
            "completion-menu": "bg:default",
            "completion-menu.completion": "bg:default fg:goldenrod",
            "completion-menu.completion.current": "bg:default fg:goldenrod bold",
            "auto-suggestion": "fg:ansibrightblack",
        }
    )

    history_file = os.path.expanduser("~/.mcp_interactive_history")
    pt_session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=ChatCommandCompleter(context),
        complete_while_typing=True,
        style=style,
        message="> ",
    )

    # --------------------------------------------------------------------- #
    # Main REPL loop
    # --------------------------------------------------------------------- #
    try:
        while True:
            try:
                user_in = await pt_session.prompt_async()
                user_in = user_in.strip()
                if not user_in:
                    continue

                # Plain 'exit' / 'quit' without slash
                if user_in.lower() in ("exit", "quit"):
                    console.print("\n[bold red]Goodbye![/bold red]")
                    return True

                cmd, *rest = user_in.split(maxsplit=1)
                arg_str = rest[0] if rest else ""

                # ------------------------------------------------------------------
                # Built-in /model and /provider (simple local switches)
                # ------------------------------------------------------------------
                if cmd == "/model":
                    if arg_str:
                        context["model"] = arg_str
                        console.print(f"[green]Switched model → {arg_str}[/green]")
                    else:
                        console.print(f"[yellow]Current model:[/yellow] {context['model']}")
                    continue

                if cmd == "/provider":
                    if arg_str:
                        context["provider"] = arg_str
                        console.print(
                            f"[green]Switched provider → {arg_str}[/green]"
                        )
                    else:
                        console.print(
                            f"[yellow]Current provider:[/yellow] {context['provider']}"
                        )
                    continue

                # ------------------------------------------------------------------
                # Normal command dispatch
                # ------------------------------------------------------------------
                if cmd in commands:
                    handler = commands[cmd]
                    if inspect.iscoroutinefunction(handler):
                        await handler(arg_str)  # type: ignore[arg-type]
                    else:
                        if handler(arg_str) == "exit":
                            console.print("\n[bold red]Goodbye![/bold red]")
                            return True
                else:
                    console.print(
                        f"[red]Unknown command:[/red] {cmd}\n"
                        "[yellow]Type '/help' for a list of commands.[/yellow]"
                    )
            except EOFError:
                return True
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted – Ctrl-D to quit.[/yellow]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"\n[red]Error:[/red] {exc}")
    finally:
        # Any cleanup for stream_manager is handled by the caller
        pass


# --------------------------------------------------------------------------- #
# UI helper functions
# --------------------------------------------------------------------------- #
def _banner(console: Console, ctx: Dict[str, Any]) -> None:
    """Nice welcome panel + tool count."""
    console.print(
        Panel(
            f"Welcome to the Interactive Mode!\n\n"
            f"Provider: [bold]{ctx['provider']}[/bold]  |  "
            f"Model: [bold]{ctx['model']}[/bold]\n\n"
            "Type '/help' for available commands or 'exit' to quit.",
            title="Interactive Mode",
            border_style="yellow",
            expand=True,
        )
    )

    if ctx["tools"]:
        console.print(f"[green]Loaded {len(ctx['tools'])} tools.[/green]")
        if ctx["server_info"]:
            console.print("[yellow]Type '/servers' to list servers.[/yellow]")


def _servers(console: Console, ctx: Dict[str, Any]) -> None:
    info = ctx["server_info"]
    if not info:
        console.print("[yellow]No servers connected.[/yellow]")
        return

    table = Table(title="Connected Servers", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Tools", style="cyan")
    table.add_column("Status", style="green")

    for s in info:
        table.add_row(
            str(s.get("id", "?")),
            s.get("name", "Unknown"),
            str(s.get("tools", 0)),
            s.get("status", "Unknown"),
        )

    console.print(table)


def _help(console: Console) -> None:
    help_md = """# Commands
- **/ping**         check server reachability  
- **/prompts**      list prompts  
- **/resources**    list resources  
- **/tools**        list tools  
- **/tools-all**    detailed tool info  
- **/tools-raw**    raw JSON definitions  
- **/chat**         enter chat mode  
- **/servers**      show connected servers (/s)  
- **/model**        show or switch model  
- **/provider**     show or switch provider  
- **/cls**          clear screen  
- **/clear**        clear + banner  
- **/help**         this help  
- **/exit**         exit the program (/quit, *exit*, *quit*)"""
    console.print(Panel(Markdown(help_md), title="Help", style="yellow"))


# --------------------------------------------------------------------------- #
# Typer wrapper – so “python -m … interactive run” still works
# --------------------------------------------------------------------------- #
@app.command("run")
def _run_via_cli() -> None:  # pragma: no cover
    Console().print("[red]Interactive mode must be started from the main MCP CLI.[/red]")
    raise typer.Exit(1)
