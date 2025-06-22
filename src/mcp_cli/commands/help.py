# mcp_cli/commands/help.py
"""
Human-friendly *help* output for both chat-mode and command-line
================================================================

This helper renders either:

* **A single command's details** when a *command_name* is given.
* **A summary table of *all* commands** otherwise.

It first tries the *interactive* registry (chat-mode); if that is not
importable we fall back to the CLI registry instead.

Highlights
----------
* **Cross-platform console** - via
  :pyfunc:`mcp_cli.utils.rich_helpers.get_console` so colours work on
  Windows terminals and disappear automatically when output is redirected.
* **Doc-string parsing** - the first non-empty line that *doesn't* start
  with “usage” becomes the one-liner in the command table.
* **Alias column** - shows “-” when a command has no aliases, keeping the
  table tidy.
"""
from __future__ import annotations
from typing import Dict, Optional
from rich.markdown import Markdown
from rich.panel     import Panel
from rich.table     import Table

# mcp cli
from mcp_cli.utils.rich_helpers import get_console

# Prefer interactive registry, gracefully fall back to CLI registry
try:
    from mcp_cli.interactive.registry import InteractiveCommandRegistry as _Reg
except ImportError:  # not in interactive mode
    from mcp_cli.cli.registry import CommandRegistry as _Reg  # type: ignore


def _get_commands() -> Dict[str, object]:
    """Return *name → Command* mapping from whichever registry is available."""
    return _Reg.get_all_commands() if hasattr(_Reg, "get_all_commands") else {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def help_action(
    command_name: Optional[str] = None,
    *,
    console=None,
) -> None:
    """
    Render help for one command or for the whole command set.

    Parameters
    ----------
    command_name:
        Show detailed help for *this* command only.  If *None* (default)
        a table of every command is produced.
    console:
        Optional :class:`rich.console.Console` instance.  If omitted a
        cross-platform console is created automatically via
        :pyfunc:`mcp_cli.utils.rich_helpers.get_console`.
    """
    console = console or get_console()
    commands = _get_commands()

    # ── detailed view ────────────────────────────────────────────────────
    if command_name:
        cmd = commands.get(command_name)
        if cmd is None:
            console.print(f"[red]Unknown command:[/red] {command_name}")
            return

        md = Markdown(
            f"## `{cmd.name}`\n\n{cmd.help or '*No description provided.*'}"
        )
        console.print(Panel(md, title="Command Help", border_style="cyan"))

        if getattr(cmd, "aliases", None):
            console.print(
                f"[dim]Aliases:[/dim] {', '.join(cmd.aliases)}", justify="right"
            )
        return

    # ── summary table ────────────────────────────────────────────────────
    tbl = Table(title="Available Commands")
    tbl.add_column("Command",   style="green",  no_wrap=True)
    tbl.add_column("Aliases",   style="cyan",   no_wrap=True)
    tbl.add_column("Description")

    for name, cmd in sorted(commands.items()):
        # First non-blank line that isn't a "usage" header
        lines = [
            ln.strip()
            for ln in (cmd.help or "").splitlines()
            if ln.strip() and not ln.strip().lower().startswith("usage")
        ]
        desc = lines[0] if lines else "No description"
        aliases = ", ".join(cmd.aliases) if getattr(cmd, "aliases", None) else "-"
        tbl.add_row(name, aliases, desc)

    console.print(tbl)
    console.print(
        "[dim]Type 'help &lt;command&gt;' for detailed information on a specific "
        "command.[/dim]"
    )
