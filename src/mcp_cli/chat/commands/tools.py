# mcp_cli/chat/commands/tools.py
"""
Chat-mode `/tools` command – list tools or open the interactive call helper.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from mcp_cli.commands.tools import tools_action_async
from mcp_cli.commands.tools_call import tools_call_action
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def tools_command(parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Usage
    -----
      /tools            List tools
      /tools --all      Show parameter details
      /tools --raw      Show raw JSON definitions
      /tools call       Open interactive tool-call UI
      /t                Alias for /tools
    """
    console = Console()
    tm: ToolManager | None = ctx.get("tool_manager")

    if not tm:
        console.print("[red]Error:[/red] ToolManager not available.")
        return True

    args = parts[1:]  # strip the command word itself

    # ── interactive call helper ─────────────────────────────────────
    if args and args[0].lower() == "call":
        await tools_call_action(tm)
        return True

    # ── list tools ───────────────────────────────────────────────────
    show_details = "--all" in args
    show_raw     = "--raw" in args

    await tools_action_async(
        tm,
        show_details=show_details,
        show_raw=show_raw,
    )
    return True


# Register command and short alias
register_command("/tools", tools_command)
register_command("/t", tools_command)
