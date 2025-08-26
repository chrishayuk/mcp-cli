# mcp_cli/chat/commands/model.py
"""
Chat-mode `/model` command for MCP-CLI
======================================

Allows users to *inspect* or *change* the current LLM model straight from the
chat session.

Shortcuts
---------
* `/model`                 - show current provider & model
* `/model list`            - list models for the active provider
* `/model <name>`          - switch to *<name>* (probe-tests first)

The heavy-lifting is delegated to
:meth:`mcp_cli.commands.model.model_action_async`, which pings the target model
to ensure it responds before committing the switch.
"""

from __future__ import annotations
from typing import Any, Dict, List

# Cross-platform Rich console (handles colour fallback on Windows / pipes)
from chuk_term.ui import output

from mcp_cli.commands.model import model_action_async
from mcp_cli.chat.commands import register_command


# ════════════════════════════════════════════════════════════════════════════
# /model entry-point
# ════════════════════════════════════════════════════════════════════════════
async def cmd_model(parts: List[str], ctx: Dict[str, Any]) -> bool:  # noqa: D401
    """
    View or change the active LLM model.

    * `/model`          - show current provider & model
    * `/model list`     - list available models for the active provider
    * `/model <name>`   - attempt to switch to **<name>** (probe first)

    The command passes its arguments verbatim to the shared helper and prints
    any errors in a user-friendly way.
    """

    try:
        await model_action_async(parts[1:], context=ctx)
    except Exception as exc:  # pragma: no cover  - unexpected
        output.print(f"[red]Model command failed:[/red] {exc}")
    return True


# ────────────────────────────────────────────────────────────────────────────
# registration
# ────────────────────────────────────────────────────────────────────────────
register_command("/model", cmd_model)
