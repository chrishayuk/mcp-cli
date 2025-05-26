# mcp_cli/interactive/commands/provider.py
"""
Interactive **provider** command - inspect or switch the active LLM provider
(and optionally the default model) from inside the shell.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from mcp_cli.utils.rich_helpers import get_console           # ← NEW
from mcp_cli.commands.provider import provider_action_async  # shared logic
from .base import InteractiveCommand

log = logging.getLogger(__name__)


class ProviderCommand(InteractiveCommand):
    """Show / switch providers, tweak config, run diagnostics."""

    def __init__(self) -> None:
        super().__init__(
            name="provider",
            aliases=["p"],
            help_text=(
                "Manage LLM providers.\n\n"
                "  provider                          Show current provider/model\n"
                "  provider list                     List available providers\n"
                "  provider config                   Show provider configuration\n"
                "  provider diagnostic [prov]        Probe provider(s) health\n"
                "  provider set <prov> <key> <val>   Update one config key\n"
                "  provider <prov> [model]           Switch provider (and model)\n"
            ),
        )

    # ------------------------------------------------------------------
    async def execute(                      # noqa: D401  (simple entry-point)
        self,
        args: List[str],
        tool_manager: Any = None,           # kept for API parity (unused)
        **ctx: Dict[str, Any],
    ) -> None:
        """
        Delegate to :func:`provider_action_async`.

        *args* arrive without the leading command word, exactly as the
        shared helper expects.
        """
        console = get_console()

        # The provider command does not require ToolManager, but log if absent
        if tool_manager is None:
            log.debug("ProviderCommand executed without ToolManager – OK for now.")

        try:
            await provider_action_async(args, context=ctx)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Provider command failed:[/red] {exc}")
            log.exception("ProviderCommand error")
