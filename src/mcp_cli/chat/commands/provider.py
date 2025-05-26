# mcp_cli/chat/commands/provider.py
"""
Chat-mode `/provider` command for MCP-CLI
========================================

Gives you full control over **LLM providers** without leaving the chat
session.

At a glance
-----------
* `/provider`                      - show current provider & model
* `/provider list`                 - list available providers
* `/provider config`               - dump full provider configs
* `/provider diagnostic`           - ping each provider with a tiny prompt
* `/provider set <prov> <k> <v>`   - change one config value (e.g. API key)
* `/provider <prov>  [model]`      - switch provider (and optional model)

All heavy lifting is delegated to
:meth:`mcp_cli.commands.provider.provider_action_async`, which performs
safety probes before committing any switch.

Features
--------
* **Cross-platform Rich console** - via
  :pyfunc:`mcp_cli.utils.rich_helpers.get_console`.
* **Graceful error surfacing** - unexpected exceptions are caught and printed
  as red error messages instead of exploding the event-loop.
"""

from __future__ import annotations
from typing import Any, Dict, List

# Cross-platform Rich console helper
from mcp_cli.utils.rich_helpers import get_console

# Shared implementation
from mcp_cli.commands.provider import provider_action_async
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# /provider entry-point
# ════════════════════════════════════════════════════════════════════════════
async def cmd_provider(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """Handle the `/provider` slash-command inside chat."""
    console = get_console()

    try:
        # Forward everything after the command itself to the shared helper
        await provider_action_async(parts[1:], context=ctx)
    except Exception as exc:  # pragma: no cover – unexpected edge cases
        console.print(f"[red]Provider command failed:[/red] {exc}")

    return True


# ────────────────────────────────────────────────────────────────────────────
# registration
# ────────────────────────────────────────────────────────────────────────────
register_command("/provider", cmd_provider)
