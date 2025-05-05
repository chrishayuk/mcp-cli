# mcp_cli/chat/commands/conversation_history.py
"""
Chat command for displaying the conversation history of the current session.
Delegates to the shared conversation logic in the interactive shell.
"""
import json
import traceback
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich import box
from rich.text import Text

# Chat registry
from mcp_cli.chat.commands import register_command

async def conversation_history_command(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display the conversation history of the current chat session.

    Usage:
      /conversation         — Show history in a table
      /conversation -n 5    — Show only the last 5 messages
      /conversation --json  — Dump history as JSON
      /conversation <row>   — Show full details for that message
      /conversation <row> --json — Show message #<row> as JSON
    """
    console = Console()
    history = context.get("conversation_history", []) or []

    if not history:
        console.print("[italic yellow]No conversation history available.[/italic yellow]")
        return True

    args = cmd_parts[1:] if len(cmd_parts) > 1 else []
    show_json = "--json" in args
    limit = None
    row = None

    # Detect row index
    if args and args[0].isdigit():
        row = int(args[0])
        if not (1 <= row <= len(history)):
            console.print(f"[red]Invalid row. Must be 1–{len(history)}.[/red]")
            return True

    # Detect limit flag
    if "-n" in args:
        try:
            idx = args.index("-n")
            limit = int(args[idx + 1])
        except Exception:
            console.print("[red]Invalid -n value; showing all[/red]")

    # Filter messages
    if row is not None:
        selected = [history[row - 1]]
    elif limit and limit > 0:
        selected = history[-limit:]
    else:
        selected = history

    # JSON mode
    if show_json:
        if row is not None:
            msg = selected[0]
            console.print(
                Panel(
                    Syntax(json.dumps(msg, indent=2, ensure_ascii=False), "json", word_wrap=True),
                    title=f"Message #{row} (JSON)",
                    box=box.ROUNDED,
                    border_style="cyan",
                    expand=True,
                    padding=(1,2)
                )
            )
        else:
            console.print(
                Panel(
                    Syntax(json.dumps(selected, indent=2, ensure_ascii=False), "json", word_wrap=True),
                    title="Conversation History (JSON)",
                    box=box.ROUNDED,
                    border_style="cyan",
                    expand=True,
                    padding=(1,2)
                )
            )
        return True

    # Single‐message panel view
    if row is not None:
        msg = selected[0]
        role = msg.get("role", "")
        name = msg.get("name", "")
        label = f"{role} ({name})" if name else role
        content = msg.get("content") or ""
        # Handle tool_calls
        if content is None and msg.get("tool_calls"):
            names = [tc["function"]["name"] for tc in msg["tool_calls"] if "function" in tc]
            content = f"[Tool call: {', '.join(names)}]"

        details = Text.from_markup(content)
        # Append tool call details
        if msg.get("tool_calls"):
            details.append("\n\nTool Calls:\n")
            for idx, tc in enumerate(msg["tool_calls"], 1):
                fn = tc["function"]
                details.append(f"  {idx}. {fn['name']} args={fn['arguments']}\n")

        console.print(
            Panel(
                details,
                title=f"Message #{row} — {label}",
                box=box.ROUNDED,
                border_style="cyan",
                expand=True,
                padding=(1,2)
            )
        )
        return True

    # Tabular view for multiple messages
    table = Table(title=f"Conversation History ({len(selected)} messages)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", style="cyan", width=12)
    table.add_column("Content", style="white")

    for msg in selected:
        idx = history.index(msg) + 1
        role = msg.get("role", "")
        name = msg.get("name", "")
        label = f"{role} ({name})" if name else role
        content = msg.get("content") or ""
        if content is None and msg.get("tool_calls"):
            names = [tc["function"]["name"] for tc in msg["tool_calls"] if "function" in tc]
            content = f"[Tool call: {', '.join(names)}]"
        # Truncate for table
        if len(content) > 100:
            content = content[:97] + "..."
        table.add_row(str(idx), label, content)

    console.print(table)
    return True

# Register commands
register_command("/conversation", conversation_history_command, ["-n", "--json"])
register_command("/ch",           conversation_history_command, ["-n", "--json"])
