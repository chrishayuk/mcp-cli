# mcp_cli/chat/commands/conversation.py
"""
Conversation-history commands for MCP-CLI chat
==============================================

This file wires four convenience commands that let you tidy up or persist the
current chat history without leaving the session:

* **/cls**       - clear the terminal window but *keep* the conversation.  
* **/clear**     - clear *both* the screen *and* the in-memory history
  (system prompt is preserved).  
* **/compact**   - ask the LLM to summarise the conversation so far and
  replace the full history with that concise summary.  
* **/save** _file_ - dump the history (minus the system prompt) to a JSON file
  on disk.

All commands are *read-only* w.r.t. external state; they operate solely on the
in-memory context that the chat UI passes in.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Rich helpers
from mcp_cli.utils.rich_helpers import get_console
from rich.panel import Panel
from rich.markdown import Markdown

# Shared UI helpers
from mcp_cli.ui.ui_helpers import display_welcome_banner, clear_screen

# Chat registry
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# /cls  - clear screen, keep history
# ════════════════════════════════════════════════════════════════════════════
async def cmd_cls(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Clear the terminal window but *preserve* the conversation history."""
    console = get_console()
    clear_screen()
    display_welcome_banner(ctx)
    console.print("[green]Screen cleared. Conversation history preserved.[/green]")
    return True


# ════════════════════════════════════════════════════════════════════════════
# /clear - clear screen *and* history
# ════════════════════════════════════════════════════════════════════════════
async def cmd_clear(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Clear the screen *and* reset the in-memory history."""
    console = get_console()
    clear_screen()

    history = ctx.get("conversation_history", [])
    if history and history[0].get("role") == "system":
        system_prompt = history[0]["content"]
        history.clear()
        history.append({"role": "system", "content": system_prompt})

    display_welcome_banner(ctx)
    console.print("[green]Screen cleared and conversation history reset.[/green]")
    return True


# ════════════════════════════════════════════════════════════════════════════
# /compact - summarise conversation
# ════════════════════════════════════════════════════════════════════════════
async def cmd_compact(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Replace lengthy history with a compact LLM-generated summary."""
    console = get_console()
    history = ctx.get("conversation_history", [])

    if len(history) <= 1:
        console.print("[yellow]Nothing to compact.[/yellow]")
        return True

    system_prompt = history[0]["content"]
    summary_prompt = {
        "role": "user",
        "content": "Please summarise our conversation so far, concisely.",
    }

    with console.status("[cyan]Generating summary…[/cyan]", spinner="dots"):
        try:
            result = await ctx["client"].create_completion(
                messages=history + [summary_prompt]
            )
            summary = result.get("response", "No summary available.")
        except Exception as exc:  # pragma: no cover
            console.print(f"[red]Error summarising conversation: {exc}[/red]")
            summary = "Failed to generate summary."

    # Reset history
    clear_screen()
    ctx["conversation_history"][:] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"**Summary:**\n\n{summary}"},
    ]

    display_welcome_banner(ctx)
    console.print("[green]Conversation compacted.[/green]")
    console.print(
        Panel(
            Markdown(f"**Summary:**\n\n{summary}"),
            title="Conversation Summary",
            style="cyan",
        )
    )
    return True


# ════════════════════════════════════════════════════════════════════════════
# /save  - write history to disk
# ════════════════════════════════════════════════════════════════════════════
async def cmd_save(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Persist the conversation history to a JSON file on disk."""
    console = get_console()

    if len(parts) < 2:
        console.print("[yellow]Usage: /save <filename>[/yellow]")
        return True

    filename = parts[1]
    if not filename.endswith(".json"):
        filename += ".json"

    history = ctx.get("conversation_history", [])[1:]  # skip system prompt
    try:
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2, ensure_ascii=False)
        console.print(f"[green]Conversation saved to {filename}[/green]")
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Failed to save conversation: {exc}[/red]")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/cls",     cmd_cls)
register_command("/clear",   cmd_clear)
register_command("/compact", cmd_compact)
register_command("/save",    cmd_save, ["<filename>"])
