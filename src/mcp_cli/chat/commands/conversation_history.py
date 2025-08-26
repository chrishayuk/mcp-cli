# mcp_cli/chat/commands/conversation_history.py
"""
Inspect the current chat history with */conversation*
=====================================================

The */conversation* (alias */ch*) command lets you browse or export the
messages exchanged in this session:

* **/conversation** - show the whole history as a Rich table
* **/conversation -n 5** - table of the last five messages
* **/conversation --json** - dump everything to JSON
* **/conversation <row>** - pretty-print a single message
* **/conversation <row>; --json** - same but as raw JSON

Column widths are capped so the view stays readable even on narrow
terminals.  When a message contains tool calls the table shows a
placeholder; the full list is included in the single-row view.

The command is completely read-only and never mutates the chat context,
so you can invoke it as often as you like without side-effects.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Cross-platform Rich console
from chuk_term.ui import output
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box  # noqa: T201 - used deliberately

from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Handler
# ════════════════════════════════════════════════════════════════════════════
async def conversation_history_command(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Browse or export the in-memory conversation history."""
    history = ctx.get("conversation_history", []) or []

    if not history:
        output.print(
            "[italic yellow]No conversation history available.[/italic yellow]"
        )
        return True

    args = parts[1:]
    show_json = "--json" in args
    limit = None
    single_row = None

    # Row index?
    if args and args[0].isdigit():
        single_row = int(args[0])
        if not (1 <= single_row <= len(history)):
            output.print(f"[red]Invalid row. Must be 1-{len(history)}[/red]")
            return True

    # -n limit?
    if "-n" in args:
        try:
            idx = args.index("-n")
            limit = int(args[idx + 1])
        except Exception:
            output.print("[red]Invalid -n value; showing all[/red]")

    # Slice history
    if single_row is not None:
        selection = [history[single_row - 1]]
    elif limit and limit > 0:
        selection = history[-limit:]
    else:
        selection = history

    # ── JSON output ─────────────────────────────────────────────────────────
    if show_json:
        payload = selection[0] if single_row else selection
        title = (
            f"Message #{single_row} (JSON)"
            if single_row
            else "Conversation History (JSON)"
        )
        output.print(
            Panel(
                Syntax(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    "json",
                    word_wrap=True,
                ),
                title=title,
                box=box.ROUNDED,
                border_style="cyan",
                expand=True,
                padding=(1, 2),
            )
        )
        return True

    # ── single-message pretty panel ─────────────────────────────────────────
    if single_row is not None:
        msg = selection[0]
        role = msg.get("role", "")
        name = msg.get("name", "")
        label = f"{role} ({name})" if name else role

        content = msg.get("content") or ""
        if content is None and msg.get("tool_calls"):
            fnames = [
                tc["function"]["name"] for tc in msg["tool_calls"] if "function" in tc
            ]
            content = f"[Tool call: {', '.join(fnames)}]"

        from rich.text import Text

        details = Text.from_markup(content)
        if msg.get("tool_calls"):
            details.append("\n\nTool Calls:\n")
            for idx, tc in enumerate(msg["tool_calls"], 1):
                fn = tc["function"]
                details.append(f"  {idx}. {fn['name']} args={fn['arguments']}\n")

        output.print(
            Panel(
                details,
                title=f"Message #{single_row} — {label}",
                box=box.ROUNDED,
                border_style="cyan",
                expand=True,
                padding=(1, 2),
            )
        )
        return True

    # ── tabular list view ───────────────────────────────────────────────────
    table = Table(title=f"Conversation History ({len(selection)} messages)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", style="cyan", width=12)
    table.add_column("Content", style="white")

    for msg in selection:
        idx = history.index(msg) + 1
        role = msg.get("role", "")
        name = msg.get("name", "")
        label = f"{role} ({name})" if name else role
        content = msg.get("content") or ""
        if content is None and msg.get("tool_calls"):
            fnames = [
                tc["function"]["name"] for tc in msg["tool_calls"] if "function" in tc
            ]
            content = f"[Tool call: {', '.join(fnames)}]"
        if len(content) > 100:
            content = content[:97] + "…"
        table.add_row(str(idx), label, content)

    output.print(table)
    output.print("\nType [green]/conversation <row>[/green] for full message details.")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/conversation", conversation_history_command)
register_command("/ch", conversation_history_command)
