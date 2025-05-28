# src/mcp_cli/cli/commands/provider.py
"""
CLI binding for "provider" management commands.

All heavy-lifting is delegated to the shared helper:
    mcp_cli.commands.provider.provider_action_async
"""
from __future__ import annotations

import logging
from typing import Any

import typer
from rich.console import Console

from mcp_cli.commands.provider import provider_action_async
from mcp_cli.model_manager import ModelManager  # ← CHANGED
from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.tools.manager import get_tool_manager
from mcp_cli.utils.async_utils import run_blocking

log = logging.getLogger(__name__)

# ─── Typer sub-app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Manage LLM provider configuration")


def _call_shared_helper(argv: list[str]) -> None:
    """Parse argv (after 'provider') and run shared async helper."""
    # Build a transient context dict (the shared helper expects it)
    ctx: dict[str, Any] = {
        "model_manager": ModelManager(),  # ← CHANGED
        # The CLI path has no session client – we omit "client"
    }
    run_blocking(provider_action_async(argv, context=ctx))


@app.command("show", help="Show current provider & model")
def provider_show() -> None:
    _call_shared_helper([])


@app.command("list", help="List configured providers")
def provider_list() -> None:
    _call_shared_helper(["list"])


@app.command("config", help="Display full provider config")
def provider_config() -> None:
    _call_shared_helper(["config"])


@app.command("set", help="Set one configuration value")
def provider_set(
    provider_name: str = typer.Argument(...),
    key: str = typer.Argument(...),
    value: str = typer.Argument(...),
) -> None:
    _call_shared_helper(["set", provider_name, key, value])


# ─── In-process command for CommandRegistry ──────────────────────────────────
class ProviderCommand(BaseCommand):
    """`provider` command usable from interactive or scripting contexts."""

    def __init__(self) -> None:
        super().__init__(
            name="provider",
            help_text="Manage LLM providers (show/list/config/set/switch).",
        )

    async def execute(self, tool_manager: Any, **params: Any) -> None:  # noqa: D401
        """
        Forward params to *provider_action_async*.

        Expected keys in **params:
          • subcommand (str)  – list | config | set | show
          • provider_name, key, value (for 'set')
          • model / provider (optional overrides)
        """
        argv: list[str] = []

        sub = params.get("subcommand", "show")
        argv.append(sub)

        if sub == "set":
            for arg in ("provider_name", "key", "value"):
                val = params.get(arg)
                if val is None:
                    Console().print(f"[red]Missing {arg} for 'set'[/red]")
                    return
                argv.append(str(val))
        elif sub not in {"show", "list", "config"}:
            # treat it as "switch provider [model]"
            argv = [sub]  # sub is actually provider name
            maybe_model = params.get("model")
            if maybe_model:
                argv.append(maybe_model)

        context: dict[str, Any] = {
            "model_manager": ModelManager(),  # ← CHANGED
        }
        await provider_action_async(argv, context=context)