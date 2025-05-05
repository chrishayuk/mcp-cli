# mcp_cli/chat/commands/tool_history.py
"""
Tool history command module for displaying executed tool calls in the current session,
refactored to follow the same registration and structure pattern.
"""
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
import json
import traceback

# Chat registry
from mcp_cli.chat.commands import register_command

async def tool_history_command(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display history of executed tool calls in the current chat session.

    Usage:
      /toolhistory         - Show all tool calls
      /toolhistory -n 5    - Show only the last 5 tool calls
      /toolhistory --json  - Show in JSON format
      /toolhistory <row>   - Show full details for a specific call
    """
    console = Console()
    history = context.get("conversation_history", []) or []

    # Extract all tool calls from assistant messages
    tool_calls = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "unknown")
            args = fn.get("arguments", {})
            # parse JSON string if needed
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({"name": name, "args": args})

    if not tool_calls:
        console.print("[italic yellow]No tool calls recorded this session.[/italic yellow]")
        return True

    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    # Row-specific view
    if args and args[0].isdigit():
        idx = int(args[0])
        if 1 <= idx <= len(tool_calls):
            entry = tool_calls[idx - 1]
            console.print(
                Panel(
                    Syntax(json.dumps(entry, indent=2), "json", line_numbers=True),
                    title=f"Tool Call #{idx} Details",
                    style="cyan"
                )
            )
        else:
            console.print(f"[red]Invalid index; choose 1–{len(tool_calls)}[/red]")
        return True

    # JSON dump?
    if "--json" in args:
        console.print(Syntax(json.dumps(tool_calls, indent=2), "json", line_numbers=True))
        return True

    # Limit
    limit = None
    if "-n" in args:
        try:
            i = args.index("-n")
            limit = int(args[i+1])
        except Exception:
            console.print("[red]Invalid -n value; showing all[/red]")
    display = tool_calls[-limit:] if limit and limit > 0 else tool_calls

    # Summary table
    table = Table(title=f"Tool Call History ({len(display)} calls)")
    table.add_column("#", style="dim")
    table.add_column("Tool", style="green")
    table.add_column("Arguments", style="yellow")

    start = len(tool_calls) - len(display) + 1
    for i, call in enumerate(display, start=start):
        args_str = json.dumps(call["args"])
        if len(args_str) > 80:
            args_str = args_str[:77] + "..."
        table.add_row(str(i), call["name"], args_str)

    console.print(table)
    return True

# Register under /toolhistory and /th
register_command("/toolhistory", tool_history_command, ["-n", "--json"])
register_command("/th",           tool_history_command, ["-n", "--json"])
