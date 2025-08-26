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

# NEW: Use the new UI module instead of rich directly
from chuk_term.ui import (
    output,
    clear_screen,
    display_chat_banner,
)

# Chat registry
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# /cls  - clear screen, keep history
# ════════════════════════════════════════════════════════════════════════════
async def cmd_cls(_parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Clear the terminal window but *preserve* the conversation history."""
    clear_screen()

    # Re-display the chat banner
    display_chat_banner(
        provider=ctx.get("provider", "unknown"), model=ctx.get("model", "unknown")
    )

    output.success("Screen cleared. Conversation history preserved.")
    return True


# ════════════════════════════════════════════════════════════════════════════
# /clear - clear screen *and* history
# ════════════════════════════════════════════════════════════════════════════
async def cmd_clear(_parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Clear the screen *and* reset the in-memory history."""
    clear_screen()

    history = ctx.get("conversation_history", [])
    if history and history[0].get("role") == "system":
        system_prompt = history[0]["content"]
        history.clear()
        history.append({"role": "system", "content": system_prompt})

    # Re-display the chat banner
    display_chat_banner(
        provider=ctx.get("provider", "unknown"), model=ctx.get("model", "unknown")
    )

    output.success("Screen cleared and conversation history reset.")
    return True


# ════════════════════════════════════════════════════════════════════════════
# /compact - summarise conversation
# ════════════════════════════════════════════════════════════════════════════
async def cmd_compact(_parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Replace lengthy history with a compact LLM-generated summary."""
    history = ctx.get("conversation_history", [])

    if len(history) <= 1:
        output.warning("Nothing to compact.")
        return True

    system_prompt = history[0]["content"]
    summary_prompt = {
        "role": "user",
        "content": "Please summarise our conversation so far, concisely.",
    }

    with output.loading("Generating summary..."):
        try:
            result = await ctx["client"].create_completion(
                messages=history + [summary_prompt]
            )
            summary = result.get("response", "No summary available.")
        except Exception as exc:
            output.error(f"Error summarising conversation: {exc}")
            summary = "Failed to generate summary."

    # Reset history
    clear_screen()
    ctx["conversation_history"][:] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"**Summary:**\n\n{summary}"},
    ]

    # Re-display the chat banner
    display_chat_banner(
        provider=ctx.get("provider", "unknown"), model=ctx.get("model", "unknown")
    )

    output.success("Conversation compacted.")

    # Display the summary in a panel
    output.panel(
        f"**Summary:**\n\n{summary}", title="Conversation Summary", style="cyan"
    )

    return True


# ════════════════════════════════════════════════════════════════════════════
# /save  - write history to disk
# ════════════════════════════════════════════════════════════════════════════
async def cmd_save(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """Persist the conversation history to a JSON file on disk."""
    if len(parts) < 2:
        output.warning("Usage: /save <filename>")
        return True

    filename = parts[1]
    if not filename.endswith(".json"):
        filename += ".json"

    history = ctx.get("conversation_history", [])[1:]  # skip system prompt
    try:
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2, ensure_ascii=False)
        output.success(f"Conversation saved to {filename}")
    except Exception as exc:
        output.error(f"Failed to save conversation: {exc}")

    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/cls", cmd_cls)
register_command("/clear", cmd_clear)
register_command("/compact", cmd_compact)
register_command("/save", cmd_save, ["<filename>"])
