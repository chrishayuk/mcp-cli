# mcp_cli/chat/commands/conversation.py
"""
Chat-mode commands for managing conversation history and context.
"""
import json
from typing import List, Dict, Any
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console

# Shared UI helpers
from mcp_cli.ui.ui_helpers import display_welcome_banner, clear_screen
# Chat registry
from mcp_cli.chat.commands import register_command

async def cmd_cls(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Clear the terminal screen only, preserving conversation history.

    Usage: /cls

    Clears your view but keeps the in‐memory conversation intact.
    """
    clear_screen()
    display_welcome_banner(context)
    print("[green]Screen cleared. Conversation history preserved.[/green]")
    return True

async def cmd_clear(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Clear screen and reset conversation history.

    Usage: /clear

    Wipes all past messages (except the system prompt) and clears your view.
    """
    clear_screen()
    history = context.get("conversation_history", [])
    if history and history[0].get("role") == "system":
        system_prompt = history[0]["content"]
        history.clear()
        history.append({"role": "system", "content": system_prompt})
    display_welcome_banner(context)
    print("[green]Screen cleared and conversation history reset.[/green]")
    return True

async def cmd_compact(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Compact conversation history by summarizing it.

    Usage: /compact

    Replaces your history (beyond the system prompt) with a brief summary
    to free up context while preserving the gist.
    """
    history = context.get("conversation_history", [])
    if len(history) <= 1:
        print("[yellow]Nothing to compact.[/yellow]")
        return True

    system_prompt = history[0]["content"]
    # Prepare a summary request
    summary_req = {"role": "user", "content": "Please summarize our conversation so far concisely."}
    convo_for_summary = history + [summary_req]

    console = Console()
    with console.status("[cyan]Generating summary…[/cyan]", spinner="dots"):
        try:
            # Note: client.create_completion is async
            result = await context["client"].create_completion(
                messages=convo_for_summary
            )
            summary = result.get("response", "No summary available.")
        except Exception as e:
            print(f"[red]Error summarizing conversation: {e}[/red]")
            summary = "Failed to generate summary."

    # Reset history with system prompt + summary
    clear_screen()
    new_history = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"**Summary:**\n\n{summary}"}
    ]
    context["conversation_history"][:] = new_history  # mutate in‐place
    display_welcome_banner(context)
    print("[green]Conversation compacted.[/green]")
    print(Panel(Markdown(f"**Summary:**\n\n{summary}"), title="Conversation Summary", style="cyan"))
    return True

async def cmd_save(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Save conversation history to a JSON file.

    Usage: /save <filename>

    Persists all messages (excluding the system prompt) to disk.
    """
    if len(cmd_parts) < 2:
        print("[yellow]Please specify a filename: /save <filename>[/yellow]")
        return True

    filename = cmd_parts[1]
    if not filename.endswith(".json"):
        filename += ".json"

    history = context.get("conversation_history", [])[1:]  # skip system prompt
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"[green]Conversation saved to {filename}[/green]")
    except Exception as e:
        print(f"[red]Failed to save conversation: {e}[/red]")
    return True

# Register commands and their aliases
register_command("/cls",     cmd_cls)
register_command("/clear",   cmd_clear)
register_command("/compact", cmd_compact)
register_command("/save",    cmd_save, ["<filename>"])
