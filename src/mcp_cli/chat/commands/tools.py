# mcp_cli/chat/commands/tools.py
from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from mcp_cli.utils.rich_helpers import get_console

# Shared helpers
from mcp_cli.commands.tools import tools_action_async
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Command handler
# ════════════════════════════════════════════════════════════════════════════
async def tools_command(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    List available tools (or call one interactively).

    This chat-command shows every server-side tool exposed by the connected
    MCP servers and can also launch a mini-wizard that walks you through
    executing a tool with JSON arguments.

    Usage
    -----
    /tools              - list tools  
    /tools --all        - include parameter schemas  
    /tools --raw        - dump raw JSON definitions  
    /tools call         - interactive “call tool” helper  
    /t                  - short alias
    """
    console = get_console()

    tm: ToolManager | None = ctx.get("tool_manager")
    if tm is None:
        console.print("[red]Error:[/red] ToolManager not available.")
        return True   # command handled

    args = parts[1:]  # drop the command itself

    # ── Interactive call helper ────────────────────────────────────────────
    if args and args[0].lower() == "call":
        await tools_call_action(tm)
        return True

    # ── Tool listing ───────────────────────────────────────────────────────
    show_details = "--all" in args
    show_raw     = "--raw" in args

    await tools_action_async(
        tm,
        show_details=show_details,
        show_raw=show_raw,
    )
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/tools", tools_command)

