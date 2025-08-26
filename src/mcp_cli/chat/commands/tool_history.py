# mcp_cli/chat/commands/tool_history.py
"""
Inspect the history of tool calls executed during this chat session.

The **/toolhistory** slash-command (alias **/th**) lets you audit every
function or tool invocation the assistant has issued so far.

Key Features
------------
* **Tabular overview** - default view lists call-number, tool‐name, and the
  (truncated) argument JSON.
* **Row drill-down**  - pass a *row-number* to see the full JSON payload for
  that specific call.
* **Quick filters**    - `-n N` limits output to the last *N* entries,
  `--json` emits a machine-readable dump of the whole list.

Examples
--------
  /toolhistory              → table of all calls
  /toolhistory -n 5         → last five calls only
  /toolhistory 3            → full JSON for call #3
  /toolhistory --json       → raw JSON dump

Usage
-----
  /toolhistory              - show all calls in a table
  /toolhistory -n 10        - last ten calls only
  /toolhistory <row>        - detailed view of one call
  /toolhistory --json       - JSON dump of all calls
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Cross-platform Rich console helper (handles Windows quirks, piping, etc.)
from chuk_term.ui import output

# Chat-command registry
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Command handler
# ════════════════════════════════════════════════════════════════════════════
async def tool_history_command(cmd_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Inspect the history of tool calls executed during this chat session."""

    history = ctx.get("conversation_history", []) or []

    # ── gather all tool-calls from assistant messages ───────────────────────
    tool_calls: List[Dict[str, Any]] = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "unknown")
            raw_args = fn.get("arguments", {})
            # decode JSON if stored as string
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({"name": name, "args": raw_args})

    if not tool_calls:
        output.print(
            "[italic yellow]No tool calls recorded this session.[/italic yellow]"
        )
        return True  # command handled

    # ── parse flags / positional args ────────────────────────────────────────
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    # 1️⃣  single-row detail
    if args and args[0].isdigit():
        row = int(args[0])
        if 1 <= row <= len(tool_calls):
            entry = tool_calls[row - 1]
            output.print(
                Panel(
                    Syntax(
                        json.dumps(entry, indent=2, ensure_ascii=False),
                        "json",
                        line_numbers=False,
                    ),
                    title=f"Tool Call #{row} Details",
                    style="cyan",
                )
            )
        else:
            output.print(f"[red]Invalid index - choose 1-{len(tool_calls)}[/red]")
        return True

    # 2️⃣  full JSON dump
    if "--json" in args:
        output.print(
            Syntax(
                json.dumps(tool_calls, indent=2, ensure_ascii=False),
                "json",
                line_numbers=False,
            )
        )
        return True

    # 3️⃣  -n limit
    limit = None
    if "-n" in args:
        try:
            idx = args.index("-n")
            limit = int(args[idx + 1])
        except Exception:
            output.print("[red]Invalid value after -n; showing all rows[/red]")

    display = tool_calls[-limit:] if limit and limit > 0 else tool_calls

    # ── render table ─────────────────────────────────────────────────────────
    table = Table(title=f"Tool Call History ({len(display)} calls)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool", style="green")
    table.add_column("Arguments", style="yellow")

    start = len(tool_calls) - len(display) + 1
    for i, call in enumerate(display, start=start):
        arg_repr = json.dumps(call["args"])
        if len(arg_repr) > 80:
            arg_repr = arg_repr[:77] + "…"
        table.add_row(str(i), call["name"], arg_repr)

    output.print(table)
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/toolhistory", tool_history_command, ["-n", "--json"])
register_command("/th", tool_history_command, ["-n", "--json"])
