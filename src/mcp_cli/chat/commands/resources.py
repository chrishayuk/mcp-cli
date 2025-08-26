# mcp_cli/chat/commands/resources.py
"""
Chat-mode “/resources” command for MCP-CLI
=========================================

The **/resources** slash-command shows every *resource* currently recorded
by the connected MCP server(s) - things like uploaded files, database
snapshots, or any other artefact that a tool has stored.

Why it exists
-------------
* Quickly verify that an upload/tool-execution succeeded.
* Check MIME-type and size before attempting to download or process a
  resource.
* Discover orphaned artefacts you may want to clean up.

Implementation highlights
-------------------------
* Delegates all heavy lifting to
  :pyfunc:`mcp_cli.commands.resources.resources_action_async`, ensuring one
  source of truth for table formatting across CLI, interactive shell and
  chat modes.
* Read-only - does **not** mutate the chat context, therefore safe to
  hot-reload.

Example
-------
>>> /resources
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Server   ┃ URI                                        ┃ Size  ┃ MIME-type    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━┩
│ sqlite   │ /tmp/report_2025-05-26T00-01-03.csv        │ 4 KB  │ text/csv     │
│ sqlite   │ /tmp/raw_dump_2025-05-25T23-59-10.parquet  │ 12 MB │ application… │
└──────────┴────────────────────────────────────────────┴───────┴──────────────┘
"""

from __future__ import annotations

from typing import Any, Dict, List

# Cross-platform Rich console helper
from chuk_term.ui import output

# Shared async helper
from mcp_cli.commands.resources import resources_action_async
from mcp_cli.tools.manager import ToolManager
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# Command handler
# ════════════════════════════════════════════════════════════════════════════
async def cmd_resources(_parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    List all recorded resources across connected servers.

    Usage
    -----
      /resources      - show resources
    """

    tm: ToolManager | None = ctx.get("tool_manager")
    if tm is None:
        output.print("[red]Error:[/red] ToolManager not available.")
        return True  # command handled

    # Delegate to the canonical async implementation
    await resources_action_async(tm)
    return True


# ════════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════════
register_command("/resources", cmd_resources)
