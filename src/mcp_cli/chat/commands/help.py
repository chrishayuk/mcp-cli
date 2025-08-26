# mcp_cli/chat/commands/help.py
"""
Chat-mode “/help” commands for MCP-CLI
======================================

This module implements two closely-related chat commands:

* **/help** - an in-session manual that either shows a concise **table of
  every slash-command** or a **detailed panel** for a single command.
* **/quickhelp** (alias **/qh**) - a *very* short crib-sheet of the half-dozen
  commands new users need most often.

Internally the code introspects the central **chat command registry**
(`mcp_cli.chat.commands`) so it always stays up-to-date—no hard-coded lists.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Chat-command registry
from mcp_cli.chat.commands import (
    register_command,
    _COMMAND_HANDLERS,
    _COMMAND_COMPLETIONS,
)

# Optional grouped help text
from mcp_cli.chat.commands.help_text import (
    TOOL_COMMANDS_HELP,
    CONVERSATION_COMMANDS_HELP,
    UI_COMMANDS_HELP,
)


# ════════════════════════════════════════════════════════════════════════════
# /help  ── contextual manual
# ════════════════════════════════════════════════════════════════════════════
async def cmd_help(cmd_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Show contextual help inside chat.

    • `/help` → overview table of **all** slash-commands.
    • `/help <command>` → detailed panel for one command.
    • `/help tools` → grouped help for tool-related commands.
    • `/help conversation` → grouped help for conversation/history commands.
    """
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    # ── individual command help (check first) ──────────────────────────────
    name = None
    if args:
        name = args[0] if args[0].startswith("/") else f"/{args[0]}"

    if name and name in _COMMAND_HANDLERS:
        handler = _COMMAND_HANDLERS[name]
        doc = (handler.__doc__ or "No detailed help available.").strip()
        text = f"## {name}\n\n{doc}"
        if name in _COMMAND_COMPLETIONS:
            comps = ", ".join(_COMMAND_COMPLETIONS[name])
            text += f"\n\n**Completions:** {comps}"
        output.print(Panel(Markdown(text), title=f"Help: {name}", style="cyan"))
        return True

    # ── grouped topical help (if not a command) ────────────────────────────
    if args and args[0].lower() in {"tools"}:
        output.print(
            Panel(Markdown(TOOL_COMMANDS_HELP), title="Tool Commands", style="cyan")
        )
        return True

    if args and args[0].lower() in {"conversation", "ch"}:
        output.print(
            Panel(
                Markdown(CONVERSATION_COMMANDS_HELP),
                title="Conversation-History Commands",
                style="cyan",
            )
        )
        return True

    if args and args[0].lower() in {"ui", "preferences"}:
        output.print(
            Panel(
                Markdown(UI_COMMANDS_HELP),
                title="UI & Preference Commands",
                style="cyan",
            )
        )
        return True

    # ── fallback: list all commands ────────────────────────────────────────
    table = Table(title=f"{len(_COMMAND_HANDLERS)} Available Commands")
    table.add_column("Command", style="green")
    table.add_column("Description")

    for cmd, handler in sorted(_COMMAND_HANDLERS.items()):
        # first non-empty line *not* starting with “Usage”
        lines = [
            ln.strip()
            for ln in (handler.__doc__ or "").splitlines()
            if ln.strip() and not ln.strip().lower().startswith("usage")
        ]
        desc = lines[0] if lines else "No description"
        table.add_row(cmd, desc)

    output.print(table)
    output.print("\nType [green]/help <command>[/green] for details.")
    return True


# ════════════════════════════════════════════════════════════════════════════
# /quickhelp  ── cheat-sheet
# ════════════════════════════════════════════════════════════════════════════
async def display_quick_help(cmd_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    Display a short cheat-sheet of the most common commands.
    """

    quick_tbl = Table(title="Quick Command Reference")
    quick_tbl.add_column("Command", style="green")
    quick_tbl.add_column("Description")

    for cmd, desc in [
        ("/help", "Show the full manual"),
        ("/theme", "Choose UI color scheme"),
        ("/tools", "List available tools"),
        ("/toolhistory", "Show history of tool calls"),
        ("/conversation", "Show conversation history"),
        ("/clear", "Reset screen & history"),
        ("/interrupt", "Cancel running tools"),
        ("/exit", "Leave chat"),
    ]:
        quick_tbl.add_row(cmd, desc)

    output.print(quick_tbl)
    output.print("\nType [green]/help[/green] for the complete list.")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Register the commands
# ════════════════════════════════════════════════════════════════════════════
register_command("/help", cmd_help)
register_command("/qh", display_quick_help)
