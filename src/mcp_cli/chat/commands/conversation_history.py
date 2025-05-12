# mcp_cli/chat/commands/conversation_history.py
"""
Chat-mode `/conversation` command – inspect current session history.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box, print

from mcp_cli.chat.commands import register_command
from mcp_cli.ui.ui_helpers import display_welcome_banner, clear_screen


# ──────────────────────────────────────────────────────────────────
# core handler
# ──────────────────────────────────────────────────────────────────
async def conversation_history_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Usage
    -----
      /conversation                Show history table
      /conversation -n 5           Show last 5 messages
      /conversation --json         Dump history as JSON
      /conversation <row> [--json] Show one message (# starts at 1)
    """
    console = Console()
    history = ctx.get("conversation_history", []) or []

    if not history:
        console.print("[italic yellow]No conversation history available.[/italic yellow]")
        return True

    args        = parts[1:]
    show_json   = "--json" in args
    limit       = None
    single_row  = None

    # row index?
    if args and args[0].isdigit():
        single_row = int(args[0])
        if not (1 <= single_row <= len(history)):
            console.print(f"[red]Invalid row. Must be 1–{len(history)}[/red]")
            return True

    # -n limit?
    if "-n" in args:
        try:
            idx = args.index("-n")
            limit = int(args[idx + 1])
        except Exception:
            console.print("[red]Invalid -n value; showing all[/red]")

    # slice history
    if single_row is not None:
        selection = [history[single_row - 1]]
    elif limit and limit > 0:
        selection = history[-limit:]
    else:
        selection = history

    # ── JSON output ─────────────────────────────────────────────────
    if show_json:
        payload = selection[0] if single_row else selection
        title   = f"Message #{single_row} (JSON)" if single_row else "Conversation History (JSON)"
        console.print(
            Panel(
                Syntax(json.dumps(payload, indent=2, ensure_ascii=False), "json", word_wrap=True),
                title=title,
                box=box.ROUNDED,
                border_style="cyan",
                expand=True,
                padding=(1, 2),
            )
        )
        return True

    # ── single-message pretty panel ────────────────────────────────
    if single_row is not None:
        msg   = selection[0]
        role  = msg.get("role", "")
        name  = msg.get("name", "")
        label = f"{role} ({name})" if name else role

        content = msg.get("content") or ""
        if content is None and msg.get("tool_calls"):
            fnames  = [tc["function"]["name"] for tc in msg["tool_calls"] if "function" in tc]
            content = f"[Tool call: {', '.join(fnames)}]"

        from rich.text import Text
        details = Text.from_markup(content)
        if msg.get("tool_calls"):
            details.append("\n\nTool Calls:\n")
            for idx, tc in enumerate(msg["tool_calls"], 1):
                fn = tc["function"]
                details.append(f"  {idx}. {fn['name']} args={fn['arguments']}\n")

        console.print(
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

    # ── tabular list view ──────────────────────────────────────────
    table = Table(title=f"Conversation History ({len(selection)} messages)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", style="cyan", width=12)
    table.add_column("Content", style="white")

    for msg in selection:
        idx   = history.index(msg) + 1
        role  = msg.get("role", "")
        name  = msg.get("name", "")
        label = f"{role} ({name})" if name else role
        content = msg.get("content") or ""
        if content is None and msg.get("tool_calls"):
            fnames  = [tc["function"]["name"] for tc in msg["tool_calls"] if "function" in tc]
            content = f"[Tool call: {', '.join(fnames)}]"
        if len(content) > 100:
            content = content[:97] + "…"
        table.add_row(str(idx), label, content)

    console.print(table)
    console.print("\nType [green]/conversation &lt;row&gt;[/green] for full message details.")
    return True


# ──────────────────────────────────────────────────────────────────
# registration
# ──────────────────────────────────────────────────────────────────
register_command("/conversation", conversation_history_command)
register_command("/ch",           conversation_history_command)
