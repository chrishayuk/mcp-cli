# mcp_cli/commands/help.py
"""
Unified help command for both interactive and CLI modes.
"""
from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Prefer interactive registry, but gracefully fall back to CLI registry
try:
    from mcp_cli.interactive.registry import InteractiveCommandRegistry as _Reg
except ImportError:  # not in interactive mode
    from mcp_cli.cli.registry import CommandRegistry as _Reg  # type: ignore


def _get_commands() -> Dict[str, object]:
    """Return the mapping of command-name → command-object."""
    return _Reg.get_all_commands() if hasattr(_Reg, "get_all_commands") else {}


# ──────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────
def help_action(command_name: Optional[str] = None, *, console: Console | None = None) -> None:
    """
    Print help for *all* commands, or a specific command if `command_name`
    is supplied.

    Parameters
    ----------
    command_name
        Optional – the command to describe in detail.
    console
        Optional – Rich Console; one is created automatically if omitted.
    """
    console = console or Console()
    commands = _get_commands()

    # ── detailed help for one command ────────────────────────────────
    if command_name:
        cmd = commands.get(command_name)
        if not cmd:
            console.print(f"[red]Unknown command:[/red] {command_name}")
            return

        md = Markdown(f"## `{cmd.name}`\n\n{cmd.help or '_No description provided._'}")
        console.print(
            Panel(
                md,
                title="Command Help",
                border_style="cyan",
            )
        )
        if getattr(cmd, "aliases", None):
            aliases = ", ".join(cmd.aliases)
            console.print(f"[dim]Aliases:[/dim] {aliases}")
        return

    # ── full list ────────────────────────────────────────────────────
    table = Table(title="Available Commands")
    table.add_column("Command", style="green")
    table.add_column("Aliases", style="cyan")
    table.add_column("Description")

    for name, cmd in sorted(commands.items()):
        desc = (cmd.help or "").split("\n", 1)[0]  # first line
        alias_str = ", ".join(cmd.aliases) if getattr(cmd, "aliases", None) else "-"
        table.add_row(name, alias_str, desc or "-")

    console.print(table)
    console.print(
        "[dim]Type 'help &lt;command&gt;' for detailed info on a specific command.[/dim]"
    )
