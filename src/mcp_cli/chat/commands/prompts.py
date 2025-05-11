# mcp_cli/chat/commands/prompts.py
"""
Interactive `/prompts` command – lists stored prompts on every
connected MCP server.

It re-uses the canonical async implementation from
`mcp_cli.commands.prompts`.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console

from mcp_cli.commands.prompts import prompts_action_cmd  # ← async helper
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


async def cmd_prompts(_parts: List[str], ctx: Dict[str, Any]) -> bool:
    """
    Usage
    -----
      /prompts    List prompts
      /p          Alias
    """
    console = Console()
    tm: ToolManager | None = ctx.get("tool_manager")

    if not tm:
        console.print("[red]Error:[/red] ToolManager not available.")
        return True

    await prompts_action_cmd(tm)
    return True


# Register command + short alias
register_command("/prompts", cmd_prompts)
register_command("/p", cmd_prompts)
