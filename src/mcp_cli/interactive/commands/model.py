# mcp_cli/interactive/commands/model.py
"""
Interactive **model** command - view or change the active LLM model.

Usage
-----
  model                 → show current provider / model
  model list            → list models for the active provider
  model <name>          → switch to <name> (probe first)
  model <provider> <model?>  → switch provider (and optional model)
  m …                   → short alias
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from chuk_term.ui import output
from mcp_cli.commands.model import model_action_async
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ModelCommand(InteractiveCommand):
    """Inspect or change the current LLM model inside the interactive shell."""

    def __init__(self) -> None:
        super().__init__(
            name="model",
            aliases=["m"],
            help_text="Show / switch the active LLM model (see `/help model`).",
        )

    # ------------------------------------------------------------------
    async def execute(  # noqa: D401
        self,
        args: List[str],
        tool_manager: Any = None,  # unused, kept for signature parity
        **ctx: Dict[str, Any],
    ) -> None:
        """
        Delegate to :func:`mcp_cli.commands.model.model_action_async`.

        *args* is everything after the command word.
        """

        # Basic sanity-check: the shared helper expects a ModelManager
        if "model_manager" not in ctx:
            log.debug("No model_manager in context - model command may misbehave.")
            output.print(
                "[yellow]Warning:[/yellow] internal ModelManager missing; "
                "results may be incomplete."
            )

        await model_action_async(args, context=ctx)
