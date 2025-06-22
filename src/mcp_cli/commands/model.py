# src/mcp_cli/commands/model.py
"""
Model-management command for MCP-CLI.

Inside chat / interactive mode
------------------------------
  /model                → show current model & provider
  /model list           → list models for the active provider
  /model <name>         → probe & switch model (auto-rollback on failure)
"""
from __future__ import annotations
from typing import Any, Dict, List
from rich.table import Table

# mcp cli
from mcp_cli.model_manager import ModelManager
from mcp_cli.utils.rich_helpers import get_console
from mcp_cli.utils.async_utils import run_blocking
from mcp_cli.utils.llm_probe import LLMProbe


# ════════════════════════════════════════════════════════════════════════
# Async implementation (core logic)
# ════════════════════════════════════════════════════════════════════════
async def model_action_async(args: List[str], *, context: Dict[str, Any]) -> None:
    console = get_console()

    # Re-use (or lazily create) a ModelManager kept in context
    model_manager: ModelManager = context.get("model_manager") or ModelManager()
    context.setdefault("model_manager", model_manager)

    provider      = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()

    # ── no arguments → just display current state ───────────────────────
    if not args:
        _print_status(console, current_model, provider)
        return

    # ── "/model list" helper ────────────────────────────────────────────
    if args[0].lower() == "list":
        _print_model_list(console, model_manager, provider)
        return

    # ── attempt model switch ────────────────────────────────────────────
    new_model = args[0]
    console.print(f"[dim]Probing model '{new_model}'…[/dim]")

    async with LLMProbe(model_manager, suppress_logging=True) as probe:
        result = await probe.test_model(new_model)

    if not result.success:
        msg = (
            f"provider error: {result.error_message}"
            if result.error_message
            else "unknown error"
        )
        console.print(f"[red]Model switch failed:[/red] {msg}")
        console.print(f"[yellow]Keeping current model:[/yellow] {current_model}")
        return

    # Success - commit the change
    model_manager.set_active_model(new_model)
    context["model"]         = new_model
    context["client"]        = result.client
    context["model_manager"] = model_manager
    console.print(f"[green]Switched to model:[/green] {new_model}")


# ════════════════════════════════════════════════════════════════════════
# Sync wrapper for non-async code-paths
# ════════════════════════════════════════════════════════════════════════
def model_action(args: List[str], *, context: Dict[str, Any]) -> None:
    """Thin synchronous facade around *model_action_async*."""
    run_blocking(model_action_async(args, context=context))


# ════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════
def _print_status(console, model: str, provider: str) -> None:
    console.print(f"[cyan]Current model:[/cyan] {model}")
    console.print(f"[cyan]Provider     :[/cyan] {provider}")
    console.print("[dim]/model <name> to switch  |  /model list to list[/dim]")


def _print_model_list(console, model_manager: ModelManager, provider: str) -> None:
    table = Table(title=f"Models for provider '{provider}'")
    table.add_column("Type",  style="cyan", width=10)
    table.add_column("Model", style="green")

    table.add_row("default", model_manager.get_default_model(provider))
    console.print(table)
