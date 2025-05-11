# mcp_cli/interactive/commands/provider.py
"""
Interactive “provider” command that re-uses the shared provider_action_async.
"""
from __future__ import annotations

from typing import Any, List

from mcp_cli.commands.provider import provider_action_async
from .base import InteractiveCommand


class ProviderCommand(InteractiveCommand):
    """Manage active LLM provider / model (interactive shell)."""

    def __init__(self) -> None:
        super().__init__(
            name="provider",
            help_text=(
                "Manage LLM providers.\n\n"
                "  provider                         Show current provider/model\n"
                "  provider list                    List providers\n"
                "  provider config                  Show provider config\n"
                "  provider set <prov> <k> <v>      Update one setting\n"
                "  provider <prov> [model]          Switch provider (and model)\n"
            ),
            aliases=["p"],
        )

    # InteractiveCommand interface ------------------------------------
    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,   # unused but kept for parity
        **ctx: Any,
    ) -> None:
        """
        Delegate to the shared async helper.

        *args* excludes the command word itself, exactly what
        provider_action_async expects.
        """
        await provider_action_async(args, context=ctx)
